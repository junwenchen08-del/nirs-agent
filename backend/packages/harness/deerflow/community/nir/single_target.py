"""Single-target model-family fitting and selection primitives."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.resources import ResourceLimitError

from deerflow.tools.types import Runtime

from ._common import (
    _bind_model_metrics,
    _err,
    _json_default,
    _load_npz_safely,
    _model_runtime_preflight_error,
    _ok,
    _parse_pipeline_step,
    _resolve,
    _sha256_file,
    _write_trusted_model_artifact,
)
from ._knowledge_hint import _build_knowledge_hint
from ._resources import budget_for_runtime, resource_error
from ._science_gate import (
    dataset_science_gate,
    reproducibility_evidence,
    science_gate_error,
    split_science_gate,
)
from .artifacts import _build_model_artifact, _write_plots_and_report
from .candidate_selection import (
    _choose_model_candidate,
    _choose_wavelength_candidate,
    _decide_autonomous_model_selection,
    _decide_autonomous_wavelength_selection,
)

_WAVELENGTH_SELECTION_METHODS = ("none", "cars", "spa", "manual")


def _resolve_training_path(runtime: Runtime, path: str, *, read_only: bool) -> str:
    """Honor the historical modeling-facade resolver for compatible callers."""
    facade = sys.modules.get("deerflow.community.nir.modeling")
    resolver = getattr(facade, "_resolve", _resolve) if facade is not None else _resolve
    return resolver(runtime, path, read_only=read_only)


def _score_model_candidate(
    method: str,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    max_components: int,
    cv_folds: int,
) -> dict:
    """Fit/tune one model on calibration only and score it on tuning."""
    model, best_n, cv_results, predict_fn = _train_one_model(
        method,
        X_cal,
        y_cal,
        max_components=max_components,
        cv_folds=cv_folds,
        cv_strategy="auto",
    )
    prediction = np.asarray(predict_fn(model, X_tune), dtype=float).ravel()
    rmse = float(np.sqrt(np.mean((prediction - np.asarray(y_tune, dtype=float).ravel()) ** 2)))
    return {
        "method": method,
        "n_components": best_n,
        "RMSE_tuning": rmse,
        "RMSECV": _extract_rmsecv(cv_results, best_n),
        "cv_results": cv_results,
        "_model": model,
        "_predict_fn": predict_fn,
        "_tuning_prediction": prediction,
    }


def _evaluate_model_candidates(
    methods: list[str],
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    max_components: int,
    cv_folds: int,
) -> tuple[list[dict], list[dict]]:
    """Evaluate bounded candidates; auto alternatives may fail without losing PLS."""
    results: list[dict] = []
    failures: list[dict] = []
    for method in methods:
        try:
            results.append(
                _score_model_candidate(
                    method,
                    X_cal,
                    y_cal,
                    X_tune,
                    y_tune,
                    max_components=max_components,
                    cv_folds=cv_folds,
                )
            )
        except Exception as exc:  # noqa: BLE001 - auto alternatives must degrade safely
            if len(methods) == 1 or method == "pls":
                raise
            failures.append({"method": method, "error": f"{type(exc).__name__}: {exc}"})
            warnings.warn(
                f"Autonomous {method} evaluation failed and was skipped: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return results, failures


def _select_model_family(
    requested_method: str | None,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    max_components: int,
    cv_folds: int,
    max_svr_samples: int,
    max_tree_samples: int,
    min_relative_improvement: float,
) -> tuple[dict, dict, list[dict]]:
    """Run the PLS-gated model-family comparison and return the chosen fit."""
    requested = (requested_method or "auto").strip().lower()
    pls_baseline = _score_model_candidate(
        "pls",
        X_cal,
        y_cal,
        X_tune,
        y_tune,
        max_components=max_components,
        cv_folds=cv_folds,
    )
    decision = _decide_autonomous_model_selection(
        requested_method=requested,
        n_calibration=X_cal.shape[0],
        n_features=X_cal.shape[1],
        baseline_rmse=pls_baseline["RMSE_tuning"],
        y_tuning=y_tune,
        max_svr_samples=max_svr_samples,
        max_tree_samples=max_tree_samples,
    )
    if requested == "auto":
        results = [pls_baseline]
        alternatives, failures = _evaluate_model_candidates(
            decision["candidate_methods"][1:],
            X_cal,
            y_cal,
            X_tune,
            y_tune,
            max_components=max_components,
            cv_folds=cv_folds,
        )
        results.extend(alternatives)
    elif requested == "pls":
        results = [pls_baseline]
        failures = []
    else:
        results, failures = _evaluate_model_candidates(
            [requested],
            X_cal,
            y_cal,
            X_tune,
            y_tune,
            max_components=max_components,
            cv_folds=cv_folds,
        )

    chosen, adoption = _choose_model_candidate(
        results,
        min_relative_improvement=min_relative_improvement,
    )
    decision["adoption"] = adoption
    decision["selected_method"] = chosen["method"]
    decision["failed_candidates"] = failures
    return chosen, decision, results


# ---------------------------------------------------------------------------
# Shared helper: train one model with a given method and return predictions.
# ---------------------------------------------------------------------------


def _train_one_model(method: str, X_tr, y_tr, *, max_components: int, cv_folds: int, cv_strategy: str = "auto"):
    """Train a model with the given method on the provided training set.

    Supports all available modelling methods:
    - Linear: pls, pcr, ridge, lasso, elasticnet
    - Tree-based: rf (Random Forest), et (Extra Trees), gbm (Gradient Boosting)
    - Instance-based: svr, knn
    - Deep learning: mlp (neural network), cnn (1D-CNN, requires PyTorch)

    Returns ``(model, best_n, cv_results, predict_fn)`` — callers do the
    actual prediction with the returned model to keep this function pure-ish.

    For PLS/PCR (which pre-date cv_strategy), we fall back transparently
    on TypeError for backward compatibility.
    """
    # ------------------------------------------------------------------
    # Classic chemometric methods (PLS / PCR) — have n_components search.
    # ------------------------------------------------------------------
    if method == "pls":
        from nir_core.model.pls import predict_pls, train_pls

        try:
            model, best_n, cv_results = train_pls(
                X_tr,
                y_tr,
                n_components=None,
                max_components=max_components,
                cv_folds=cv_folds,
                cv_strategy=cv_strategy,
                random_state=42,
            )
        except TypeError:
            model, best_n, cv_results = train_pls(
                X_tr,
                y_tr,
                n_components=None,
                max_components=max_components,
                cv_folds=cv_folds,
                random_state=42,
            )
        return model, best_n, cv_results, predict_pls
    if method == "pcr":
        from nir_core.model.pcr import predict_pcr, train_pcr

        try:
            model, best_n, cv_results = train_pcr(
                X_tr,
                y_tr,
                n_components=None,
                max_components=max_components,
                cv_folds=cv_folds,
                cv_strategy=cv_strategy,
                random_state=42,
            )
        except TypeError:
            model, best_n, cv_results = train_pcr(
                X_tr,
                y_tr,
                n_components=None,
                max_components=max_components,
                cv_folds=cv_folds,
                random_state=42,
            )
        return model, best_n, cv_results, predict_pcr

    # ------------------------------------------------------------------
    # SVR — grid-searched C/gamma.
    # ------------------------------------------------------------------
    if method == "svr":
        from nir_core.model.svr import predict_svr, train_svr

        try:
            model, cv_results = train_svr(
                X_tr,
                y_tr,
                cv_folds=cv_folds,
                cv_strategy=cv_strategy,
                random_state=42,
            )
        except TypeError:
            model, cv_results = train_svr(
                X_tr,
                y_tr,
                cv_folds=cv_folds,
                random_state=42,
            )
        return model, None, cv_results, predict_svr

    # ------------------------------------------------------------------
    # Tree-based ensembles (RF / ET / GBM) — grid-searched, scale-invariant.
    # ------------------------------------------------------------------
    if method == "rf":
        from nir_core.model.rf import predict_rf, train_rf

        model, cv_results = train_rf(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_rf
    if method == "et":
        from nir_core.model.rf import predict_et, train_et

        model, cv_results = train_et(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_et
    if method == "gbm":
        from nir_core.model.gbm import predict_gbm, train_gbm

        model, cv_results = train_gbm(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_gbm

    # ------------------------------------------------------------------
    # Regularized linear (Ridge / Lasso / ElasticNet).
    # ------------------------------------------------------------------
    if method == "ridge":
        from nir_core.model.linear_reg import predict_ridge, train_ridge

        model, cv_results = train_ridge(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_ridge
    if method == "lasso":
        from nir_core.model.linear_reg import predict_lasso, train_lasso

        model, cv_results = train_lasso(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_lasso
    if method == "elasticnet":
        from nir_core.model.linear_reg import predict_elasticnet, train_elasticnet

        model, cv_results = train_elasticnet(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_elasticnet

    # ------------------------------------------------------------------
    # KNN — distance-based, standardised internally.
    # ------------------------------------------------------------------
    if method == "knn":
        from nir_core.model.knn import predict_knn, train_knn

        model, cv_results = train_knn(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_knn

    # ------------------------------------------------------------------
    # Deep learning: MLP (sklearn) and CNN (PyTorch).
    # ------------------------------------------------------------------
    if method == "mlp":
        from nir_core.model.mlp import predict_mlp, train_mlp

        model, cv_results = train_mlp(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_mlp
    if method == "cnn":
        from nir_core.model.cnn import predict_cnn, train_cnn

        model, cv_results = train_cnn(
            X_tr,
            y_tr,
            cv_folds=cv_folds,
            cv_strategy=cv_strategy,
            random_state=42,
        )
        return model, None, cv_results, predict_cnn

    raise ValueError(f"Unknown method {method!r}; use one of: pls/pcr/svr/rf/et/gbm/ridge/lasso/elasticnet/knn/mlp/cnn")


def _extract_rmsecv(cv_results, best_n) -> float | None:
    """Pull the RMSECV corresponding to ``best_n`` out of cv_results.

    Handles three cv_results layouts:
    - PLS/PCR: ``mean_rmse_cv`` is a list indexed by ``n_components``.
    - CNN: ``mean_rmse_cv`` is a float.
    - RF/GBM/Ridge/Lasso/KNN/MLP/SVR: ``best_rmse`` is a float (no
      component search), so we fall back to it.
    """
    if not isinstance(cv_results, dict):
        return None
    _rmse_cv = cv_results.get("mean_rmse_cv")
    if isinstance(_rmse_cv, list) and _rmse_cv:
        _nc_list = cv_results.get("n_components", [])
        if best_n in _nc_list:
            return float(_rmse_cv[_nc_list.index(best_n)])
        return float(min(_rmse_cv))
    if isinstance(_rmse_cv, (int, float)):
        return float(_rmse_cv)
    # Fallback: methods with grid search store best_rmse.
    _best_rmse = cv_results.get("best_rmse")
    if isinstance(_best_rmse, (int, float)) and np.isfinite(_best_rmse):
        return float(_best_rmse)
    return None


def _parse_wavelength_selection_params(params: str | dict | None) -> dict:
    """Parse optional wavelength-selection params from JSON or a dict."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    if isinstance(params, str):
        stripped = params.strip()
        if not stripped:
            return {}
        import json

        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("wavelength_selection_params must be a JSON object")
        return parsed
    raise ValueError("wavelength_selection_params must be None, a dict, or a JSON object string")


def _normalise_wavelength_selection_method(method: str | None) -> str:
    selected = (method or "none").strip().lower()
    if selected in {"", "all", "full", "full_wavelength", "full-wavelength"}:
        return "none"
    if selected not in _WAVELENGTH_SELECTION_METHODS:
        raise ValueError(f"Unknown wavelength_selection {method!r}; use one of: {', '.join(_WAVELENGTH_SELECTION_METHODS)}")
    return selected


def _coerce_index_list(values, *, n_wavelengths: int) -> list[int]:
    indices = sorted(set(int(v) for v in values))
    if not indices:
        raise ValueError("manual wavelength selection produced no indices")
    bad = [i for i in indices if i < 0 or i >= n_wavelengths]
    if bad:
        raise ValueError(f"manual wavelength indices out of range: {bad[:10]}")
    return indices


def _parse_wavelength_range_string(value: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for chunk in value.split(","):
        token = chunk.strip()
        if not token:
            continue
        if ":" in token:
            left, right = token.split(":", 1)
        elif "-" in token:
            left, right = token.split("-", 1)
        else:
            raise ValueError(f"Invalid wavelength range {token!r}; expected 'start-end' or 'start:end'")
        lo = float(left.strip())
        hi = float(right.strip())
        ranges.append((min(lo, hi), max(lo, hi)))
    return ranges


def _normalise_wavelength_ranges(raw_ranges) -> list[tuple[float, float]]:
    if raw_ranges is None:
        return []
    if isinstance(raw_ranges, str):
        return _parse_wavelength_range_string(raw_ranges)
    ranges: list[tuple[float, float]] = []
    for item in raw_ranges:
        if isinstance(item, str):
            ranges.extend(_parse_wavelength_range_string(item))
        elif isinstance(item, dict):
            lo = item.get("min", item.get("start", item.get("from")))
            hi = item.get("max", item.get("end", item.get("to")))
            if lo is None or hi is None:
                raise ValueError(f"Invalid wavelength range object {item!r}")
            ranges.append((min(float(lo), float(hi)), max(float(lo), float(hi))))
        else:
            if len(item) != 2:
                raise ValueError(f"Invalid wavelength range {item!r}")
            lo, hi = float(item[0]), float(item[1])
            ranges.append((min(lo, hi), max(lo, hi)))
    return ranges


def _manual_wavelength_indices(params: dict, *, n_wavelengths: int, wv: np.ndarray | None) -> list[int]:
    if "indices" in params:
        return _coerce_index_list(params["indices"], n_wavelengths=n_wavelengths)

    raw_ranges = params.get("ranges", params.get("wavelength_ranges", params.get("range")))
    ranges = _normalise_wavelength_ranges(raw_ranges)
    if not ranges:
        raise ValueError("manual wavelength selection requires 'indices' or wavelength 'ranges'")
    if wv is None:
        raise ValueError("manual wavelength ranges require wavelength vector 'wv' in the input data")

    wv_arr = np.asarray(wv, dtype=float).ravel()
    selected: list[int] = []
    for lo, hi in ranges:
        selected.extend(int(i) for i in np.where((wv_arr >= lo) & (wv_arr <= hi))[0])
    return _coerce_index_list(selected, n_wavelengths=n_wavelengths)


def _selection_metadata(method: str, params: dict, indices: list[int], *, n_original: int, wv: np.ndarray | None) -> dict:
    selected_wv = None
    selected_indices = None if method == "none" else [int(i) for i in indices]
    if method != "none" and wv is not None:
        wv_arr = np.asarray(wv, dtype=float).ravel()
        selected_wv = [float(wv_arr[i]) for i in indices]
    return {
        "method": method,
        "params": params,
        "n_original": int(n_original),
        "n_selected": int(len(indices)),
        "selected_indices": selected_indices,
        "selected_wavelengths": selected_wv,
    }


def _apply_wavelength_selection(
    X_tr,
    X_val,
    X_te,
    y_tr,
    *,
    wv: np.ndarray | None,
    wavelength_selection: str | None,
    wavelength_selection_params: str | dict | None,
    cv_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Fit wavelength selection on train only, then slice all splits."""
    method = _normalise_wavelength_selection_method(wavelength_selection)
    params = _parse_wavelength_selection_params(wavelength_selection_params)

    X_tr = np.asarray(X_tr, dtype=float)
    X_val = np.asarray(X_val, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    n_original = int(X_tr.shape[1])

    if method == "none":
        all_indices = list(range(n_original))
        return X_tr, X_val, X_te, wv, _selection_metadata("none", {}, all_indices, n_original=n_original, wv=wv)

    if method == "cars":
        from nir_core.model.selection import cars_wavelength_selection

        _, selected_indices = cars_wavelength_selection(
            X_tr,
            y_tr,
            n_mc_samples=int(params.get("n_mc_samples", 50)),
            n_folds=int(params.get("n_folds", cv_folds)),
            random_state=int(params.get("random_state", 42)),
        )
    elif method == "spa":
        from nir_core.model.selection import spa_wavelength_selection

        n_max = params.get("n_max")
        _, selected_indices = spa_wavelength_selection(
            X_tr,
            y_tr,
            n_min=int(params.get("n_min", 1)),
            n_max=None if n_max is None else int(n_max),
        )
    else:
        selected_indices = _manual_wavelength_indices(params, n_wavelengths=n_original, wv=wv)

    selected_indices = _coerce_index_list(selected_indices, n_wavelengths=n_original)
    selected_wv_arr = np.asarray(wv, dtype=float).ravel()[selected_indices] if wv is not None else None
    metadata = _selection_metadata(method, params, selected_indices, n_original=n_original, wv=wv)
    return X_tr[:, selected_indices], X_val[:, selected_indices], X_te[:, selected_indices], selected_wv_arr, metadata


def _target_rank_selection_indices(y: np.ndarray, limit: int) -> np.ndarray:
    """Choose a deterministic, target-covering calibration subset for CARS."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size <= limit:
        return np.arange(y.size, dtype=int)
    ordered = np.argsort(y, kind="mergesort")
    positions = np.linspace(0, y.size - 1, int(limit), dtype=int)
    return np.sort(ordered[positions])


def _score_pls_wavelength_candidate(
    candidate: dict,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    *,
    max_components: int,
) -> dict:
    """Select latent variables for one wavelength candidate on tuning only."""
    from nir_core.model.pls import predict_pls, train_pls

    indices = [int(index) for index in candidate["indices"]]
    upper = max(1, min(int(max_components), X_cal.shape[0] - 1, len(indices)))
    best_rmse = float("inf")
    best_components = 1
    for components in range(1, upper + 1):
        model, _, _ = train_pls(X_cal[:, indices], y_cal, n_components=components)
        prediction = predict_pls(model, X_tune[:, indices])
        score = float(np.sqrt(np.mean((prediction - y_tune) ** 2)))
        if score < best_rmse:
            best_rmse = score
            best_components = components
    return {
        **candidate,
        "indices": indices,
        "n_selected": int(len(indices)),
        "n_components": int(best_components),
        "RMSE_tuning": float(best_rmse),
    }


def _autonomous_pls_wavelength_selection(
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    X_test: np.ndarray,
    *,
    wv: np.ndarray | None,
    wavelength_selection_params: str | dict | None,
    max_components: int,
    cv_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict, dict, list[dict]]:
    """Compare full-spectrum PLS with CARS using calibration/tuning only."""
    from nir_core.model.selection import cars_wavelength_selection

    params = _parse_wavelength_selection_params(wavelength_selection_params)
    n_original = int(X_cal.shape[1])
    full_result = _score_pls_wavelength_candidate(
        {"method": "none", "indices": list(range(n_original)), "params": {}},
        X_cal,
        y_cal,
        X_tune,
        y_tune,
        max_components=max_components,
    )
    max_selection_samples = int(params.get("max_selection_samples", params.get("max_cars_samples", 750)))
    decision = _decide_autonomous_wavelength_selection(
        compare_cars=None,
        n_calibration=X_cal.shape[0],
        n_wavelengths=n_original,
        baseline_rmse=full_result["RMSE_tuning"],
        y_tuning=y_tune,
        max_selection_samples=max_selection_samples,
    )
    candidate_results = [full_result]
    if decision["evaluate_cars"]:
        cars_params = {
            "n_mc_samples": int(params.get("n_mc_samples", 20)),
            "n_folds": int(params.get("n_folds", cv_folds)),
            "random_state": int(params.get("random_state", 42)),
        }
        selection_rows = _target_rank_selection_indices(y_cal, decision["cars_fit_samples"])
        try:
            _, cars_indices = cars_wavelength_selection(
                X_cal[selection_rows],
                y_cal[selection_rows],
                **cars_params,
            )
            candidate_results.append(
                _score_pls_wavelength_candidate(
                    {
                        "method": "cars",
                        "indices": cars_indices,
                        "params": cars_params,
                        "selection_fit_samples": int(selection_rows.size),
                    },
                    X_cal,
                    y_cal,
                    X_tune,
                    y_tune,
                    max_components=max_components,
                )
            )
        except Exception as exc:  # noqa: BLE001 - auto mode must retain a safe baseline
            decision["selection_error"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(
                f"Autonomous CARS evaluation failed; retaining full spectrum: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    chosen, adoption = _choose_wavelength_candidate(
        candidate_results,
        min_relative_improvement=float(params.get("min_relative_improvement", 0.005)),
    )
    if decision.get("selection_error"):
        adoption = {
            "reason_code": "cars_evaluation_failed",
            "relative_RMSE_improvement": None,
        }
    decision["adoption"] = adoption
    decision["selected_method"] = chosen["method"]
    indices = [int(index) for index in chosen["indices"]]
    model_wv = np.asarray(wv, dtype=float).ravel()[indices] if wv is not None else None
    metadata = _selection_metadata(
        chosen["method"],
        chosen["params"],
        indices,
        n_original=n_original,
        wv=wv,
    )
    return (
        X_cal[:, indices],
        X_tune[:, indices],
        X_test[:, indices],
        model_wv,
        metadata,
        decision,
        candidate_results,
    )


@tool("nir_train_model", parse_docstring=True)
def nir_train_model_tool(
    runtime: Runtime,
    input_path: str,
    method: str = "pls",
    pipeline_steps: str | None = None,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    max_components: int = 20,
    cv_folds: int = 10,
    cv_strategy: str = "auto",
    wavelength_selection: str = "none",
    wavelength_selection_params: str | None = None,
    model_output: str = "/mnt/user-data/outputs/model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/metrics.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train a chemometric / ML / DL calibration model.

    Supports 12 modelling methods:
    - Classic: ``pls`` / ``pcr`` / ``svr``
    - Tree-based: ``rf`` (Random Forest) / ``et`` (Extra Trees) / ``gbm`` (Gradient Boosting)
    - Regularized linear: ``ridge`` / ``lasso`` / ``elasticnet``
    - Instance-based: ``knn``
    - Deep learning: ``mlp`` (neural network) / ``cnn`` (1D-CNN, requires PyTorch)

    ★ v3 改进: 接受可选的 ``pipeline_steps`` JSON 参数实现防泄露预处理。
    当提供 ``pipeline_steps`` 时，工具内部执行：划分数据 → 仅在训练集
    拟合预处理参数 → 变换全部集合 → 建模。避免了分步模式中先全局预处理
    后划分导致的数据泄露问题。

    自动执行: 三集分离（可选内置防泄露预处理）→ 内部CV选成分数/超参数 →
    全指标评估（RMSEC/RMSECV/RMSEP, R², RPD, bias, slope）→ VIP
    可解释性输出 → 模型序列化 → 指标JSON。

    Args:
        input_path: Virtual path to the .npz file (must contain ``X`` and
            ``y``). Can be raw data or pre-preprocessed data.
        method: Modelling method: ``pls`` / ``pcr`` / ``svr`` / ``rf`` /
            ``et`` / ``gbm`` / ``ridge`` / ``lasso`` / ``elasticnet`` /
            ``knn`` / ``mlp`` / ``cnn``.
        pipeline_steps: ★ v3: JSON array of preprocessing steps. Each step
            can be a plain method name string (``"snv"``) or a dict with
            a method name and optional hyper-parameters dict. When provided,
            the tool internally splits data first, then fits the pipeline on
            train only (leakage-safe), then transforms all sets.
            If None, assumes input is already preprocessed.
        test_ratio: Fraction of data reserved as the independent test set.
        val_ratio: Fraction reserved as validation.
        max_components: Upper bound for the PLS/PCR component search.
        cv_folds: Number of CV folds for component selection.
        cv_strategy: ★ v3: ``"auto"`` (default, adapts to sample size),
            ``"loocv"`` (Leave-One-Out), or ``"fixed"`` (use cv_folds).
        wavelength_selection: Optional train-only wavelength selection:
            ``"none"`` (default), ``"cars"``, ``"spa"``, or ``"manual"``.
        wavelength_selection_params: Optional JSON object with method params.
            CARS accepts ``n_mc_samples``, ``n_folds``, ``random_state``; SPA
            accepts ``n_min``/``n_max``; manual accepts ``indices`` or
            wavelength ``ranges``.
        model_output: Virtual path for the serialised model (.pkl).
        metrics_output: Virtual path for the metrics JSON file.
        domain: Application domain for quality-gate thresholds.

    Returns:
        JSON with method, n_components, R2_val, RPD, RMSEC, RMSECV, RMSEP,
        VIP summary, coefficient summary (including the raw-space PLS
        intercept), model_path, metrics_path, quality assessment, and
        knowledge_hint.

        ★ knowledge_hint: When non-null (unknown domain or R²_val < 0.7),
        the LLM SHOULD call ``nir_search_knowledge(query=hint['query'])``
        before deciding whether to retry with a different pipeline or report
        the result to the user.
    """
    try:
        import json

        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        from nir_core.model.evaluation import compute_metrics, split_dataset
        from nir_core.utils.metrics import evaluate_quality

        # Defensive: newer nir_core exposes PLS interpretability helpers;
        # older sandboxes may not. Import lazily and tolerate ImportError so the
        # tool still runs end-to-end on the older wheel.
        try:
            from nir_core.model.pls import (
                compute_vip,
                get_regression_coefficients,
                get_regression_intercept,
            )
        except ImportError:
            compute_vip = None
            get_regression_coefficients = None
            get_regression_intercept = None

        budget = budget_for_runtime(runtime)
        real_in = _resolve_training_path(runtime, input_path, read_only=True)
        budget.check_file(real_in, stage="model_input_preflight")
        data_dict = _load_npz_safely(real_in)
        X = np.asarray(data_dict["X"], dtype=float)
        y = np.asarray(data_dict["y"], dtype=float).ravel()
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
        budget.check_matrix_shape(
            X.shape,
            dtype=X.dtype,
            target_count=1,
            peak_multiplier=8.0,
            stage="model_matrix_preflight",
        )
        dataset_validation = dataset_science_gate(X, y, wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)

        budget.checkpoint("data_split")
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            random_state=42,
        )
        split_validation = split_science_gate(
            calibration=X_tr,
            tuning=X_val,
            holdout=X_te,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        # ★ v3: Leakage-safe inline preprocessing when pipeline_steps given.
        best_pipe = None
        preprocessing_desc = "none"
        steps_list: list = []
        if pipeline_steps is not None:
            budget.checkpoint("preprocessing")
            try:
                from nir_core.models import PreprocessingStep  # noqa: F401
                from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
            except ImportError:
                return _err("nir_core V3 features (PreprocessingPipeline) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core, or remove pipeline_steps and train without inline preprocessing.")

            try:
                steps_list = json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
            except (ValueError, TypeError):
                return _err(f"Invalid pipeline_steps JSON: {pipeline_steps!r}")

            steps = [_parse_pipeline_step(m) for m in steps_list]

            # ★ V3.6: Validate pipeline before training (flexible guardrail).
            is_valid, vreason = validate_pipeline(steps)
            if not is_valid:
                return _err(f"流水线非法: {vreason}")

            best_pipe = PreprocessingPipeline(steps=steps) if steps else None
            if best_pipe is not None:
                # Fit on TRAIN ONLY, then transform all sets.
                best_pipe.fit(X_tr, wv)
                X_tr = best_pipe.transform(X_tr, wv)
                X_val = best_pipe.transform(X_val, wv)
                X_te = best_pipe.transform(X_te, wv)
                preprocessing_desc = best_pipe.description()

        budget.checkpoint("wavelength_selection")
        X_tr, X_val, X_te, model_wv, wavelength_selection_meta = _apply_wavelength_selection(
            X_tr,
            X_val,
            X_te,
            y_tr,
            wv=wv,
            wavelength_selection=wavelength_selection,
            wavelength_selection_params=wavelength_selection_params,
            cv_folds=cv_folds,
        )

        budget.checkpoint("model_fit")
        model, best_n, cv_results, predict_fn = _train_one_model(method, X_tr, y_tr, max_components=max_components, cv_folds=cv_folds, cv_strategy=cv_strategy)
        y_pred_tr = predict_fn(model, X_tr)
        y_pred_val = predict_fn(model, X_val)
        y_pred_te = predict_fn(model, X_te)

        training_data_hash = _sha256_file(real_in)
        metrics = {
            "training_data_hash": training_data_hash,
            "scientific_validation": {
                "schema_version": 1,
                "passed": True,
                "dataset": dataset_validation,
                "partition_separation": split_validation,
            },
            "reproducibility": reproducibility_evidence(
                random_state=42,
                protocol="random_three_way_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "test_ratio": float(test_ratio),
                    "val_ratio": float(val_ratio),
                    "cv_folds": int(cv_folds),
                    "cv_strategy": cv_strategy,
                    "method": method,
                    "max_components": int(max_components),
                    "wavelength_selection": wavelength_selection,
                },
            ),
            "method": method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(wavelength_selection_meta["n_original"]),
            "n_wavelengths_model": int(wavelength_selection_meta["n_selected"]),
            "preprocessing": preprocessing_desc,
            "preprocessing_steps": steps_list if pipeline_steps is not None else [],
            "wavelength_selection": wavelength_selection_meta,
            "train": compute_metrics(y_tr, y_pred_tr),
            "val": compute_metrics(y_val, y_pred_val),
            "test": compute_metrics(y_te, y_pred_te),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        metrics["RMSECV"] = _extract_rmsecv(cv_results, best_n)
        metrics["RMSEP"] = metrics["test"]["RMSE"]

        # ★ v3: Interpretability — VIP and regression coefficients (PLS only).
        vip_summary = None
        coef_summary = None
        if method == "pls":
            try:
                vip_scores = compute_vip(model, X_tr, y_tr)
                top_local = [int(i) for i in np.argsort(vip_scores)[-10:][::-1]]
                selected_indices = wavelength_selection_meta.get("selected_indices") or list(range(len(vip_scores)))
                top_original = [int(selected_indices[i]) for i in top_local if i < len(selected_indices)]
                vip_summary = {
                    "max": float(np.max(vip_scores)),
                    "mean": float(np.mean(vip_scores)),
                    "n_above_1": int(np.sum(vip_scores > 1.0)),
                    "top_10_indices": top_original,
                    "top_10_model_indices": top_local,
                }
                coef = get_regression_coefficients(model)
                coef_summary = {
                    "max_abs": float(np.max(np.abs(coef))),
                    "mean_abs": float(np.mean(np.abs(coef))),
                    "intercept": get_regression_intercept(model),
                }
                metrics["vip_summary"] = vip_summary
                metrics["coef_summary"] = coef_summary
            except Exception as exc:
                warnings.warn(f"VIP/coefficient computation failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

        # Quality assessment.
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(X.shape[0]))
        metrics["quality"] = quality

        # ★ V3.6: Residual diagnostics for LLM-driven preprocessing decisions.
        try:
            from nir_core.diagnostics import compute_residual_diagnostics

            val_diag = compute_residual_diagnostics(y_val, y_pred_val)
            metrics["diagnostics"] = val_diag
        except Exception as exc:
            warnings.warn(f"Residual diagnostics failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

        # Serialise model + metrics.
        budget.checkpoint("persist_artifacts")
        real_model = _resolve_training_path(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        _write_trusted_model_artifact(
            _build_model_artifact(
                model,
                method=method,
                preprocessing_pipeline=best_pipe,
                preprocessing_desc=preprocessing_desc,
                wavelength_selection=wavelength_selection_meta,
                X_reference=X_tr,
            ),
            real_model,
        )

        real_metrics = _resolve_training_path(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(
            real_model,
            real_metrics,
            training_data_hash=training_data_hash,
        )

        # Generate plots + report.
        from nir_core.models import SpectralData

        spec_data = SpectralData(X=X, y=y, wv=wv)

        out_dir = os.path.dirname(real_model)
        raw_b64, pred_b64, resid_b64, cv_b64, vip_b64, coef_b64 = _write_plots_and_report(
            out_dir=out_dir,
            spec_data=spec_data,
            metrics=metrics,
            quality=quality,
            best_pipe=best_pipe,
            y_te=y_te,
            y_pred_te=y_pred_te,
            cv_results=cv_results,
            best_n=best_n,
            method=method,
            model=model,
            X_tr=X_tr,
            y_tr=y_tr,
            wv=model_wv,
        )

        out_virtual = os.path.dirname(model_output)

        # ★ Knowledge-base hint: when the domain is non-standard or R² is low,
        # suggest a concrete retrieval query so the LLM can pull relevant
        # paper sections before deciding next steps (retry / report).
        knowledge_hint = _build_knowledge_hint(
            domain=domain,
            grade=quality.get("grade"),
            passed=quality.get("passed"),
            r2_val=metrics.get("R2_val"),
            diagnostics=metrics.get("diagnostics"),
        )

        return _ok(
            {
                "status": "ok",
                "method": method,
                "n_components": best_n,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": wavelength_selection_meta,
                "R2_val": round(metrics["R2_val"], 4),
                "RPD": round(metrics["RPD"], 4),
                "RMSEC": round(metrics["RMSEC"], 4),
                "RMSECV": (round(metrics["RMSECV"], 4) if metrics["RMSECV"] is not None else None),
                "RMSEP": round(metrics["RMSEP"], 4),
                "vip_summary": vip_summary,
                "coef_summary": coef_summary,
                "cv_strategy": cv_results.get("cv_strategy", "unknown"),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "thresholds_used": quality["thresholds_used"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "resource_budget": budget.evidence(),
                "report": out_virtual + "/report.md",
                "knowledge_hint": knowledge_hint,
                "plots": {
                    "raw_spectra": out_virtual + "/raw_spectra.png",
                    "predicted_vs_reference": out_virtual + "/predicted_vs_reference.png",
                    "residuals": out_virtual + "/residuals.png",
                    "cv_curve": (out_virtual + "/cv_curve.png") if cv_b64 else None,
                    "vip_scores": (out_virtual + "/vip_scores.png") if vip_b64 else None,
                    "regression_coefficients": (out_virtual + "/regression_coefficients.png") if coef_b64 else None,
                },
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except MemoryError:
        return _err("内存不足: 数据集过大或预处理候选过多。请减少候选流水线数量或使用更小子集。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
