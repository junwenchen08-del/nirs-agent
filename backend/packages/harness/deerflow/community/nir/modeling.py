"""NIR modeling tools: train / analyze / compare / register.

This module contains the model-building tools that share a common
pattern: load data → optional leakage-safe preprocessing → train PLS/PCR/SVR
→ full evaluation → quality gate → plots + report → knowledge_hint.

Keeping them together (rather than splitting per-tool) avoids duplicating
the shared train+evaluate helper code, while still separating them from
I/O, preprocessing-only, and reflection tools.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.sniffers import inspect_file

from deerflow.tools.types import Runtime

from ._common import _err, _json_default, _ok, _parse_pipeline_step, _resolve, _resolve_writable_dir
from ._knowledge_hint import _build_knowledge_hint
from ._report import _build_report

_WAVELENGTH_SELECTION_METHODS = ("none", "cars", "spa", "manual")
_MODEL_METHODS = ("pls", "pcr", "svr", "rf", "et", "gbm", "ridge", "lasso", "elasticnet", "knn", "mlp", "cnn")


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


def _build_model_artifact(model, *, method: str, preprocessing_pipeline, preprocessing_desc: str, wavelength_selection: dict):
    """Return a serialisable model artifact, preserving old plain-model files when possible."""
    if preprocessing_pipeline is None and wavelength_selection.get("method") == "none":
        return model
    return {
        "format": "nir_model_artifact",
        "version": 2,
        "model": model,
        "method": method,
        "preprocessing": {
            "description": preprocessing_desc,
            "pipeline": preprocessing_pipeline,
            "apply_on_predict": preprocessing_pipeline is not None,
        },
        "wavelength_selection": wavelength_selection,
    }


def _decide_autonomous_wavelength_selection(
    *,
    compare_cars: bool | None,
    n_calibration: int,
    n_wavelengths: int,
    baseline_rmse: float,
    y_tuning: np.ndarray,
    max_selection_samples: int = 750,
) -> dict:
    """Decide whether CARS is worth evaluating without consulting test data."""
    n_calibration = int(n_calibration)
    n_wavelengths = int(n_wavelengths)
    max_selection_samples = max(30, int(max_selection_samples))
    y_tuning = np.asarray(y_tuning, dtype=float).ravel()
    tuning_variance = float(np.var(y_tuning)) if y_tuning.size else 0.0
    baseline_r2 = float(1.0 - float(baseline_rmse) ** 2 / tuning_variance) if tuning_variance > 0 else None
    ratio = float(n_wavelengths / max(1, n_calibration))
    cars_fit_samples = min(n_calibration, max_selection_samples)
    common = {
        "n_calibration": n_calibration,
        "n_wavelengths": n_wavelengths,
        "wavelength_to_sample_ratio": ratio,
        "baseline_RMSE_tuning": float(baseline_rmse),
        "baseline_R2_tuning": baseline_r2,
        "cars_fit_samples": cars_fit_samples,
        "sample_cap_applied": cars_fit_samples < n_calibration,
        "max_selection_samples": max_selection_samples,
    }

    if compare_cars is False:
        return {
            **common,
            "mode": "disabled",
            "evaluate_cars": False,
            "reason_code": "explicitly_disabled",
            "reason": "CARS comparison was explicitly disabled by the caller.",
            "signals": [],
        }
    if compare_cars is True:
        return {
            **common,
            "mode": "forced",
            "evaluate_cars": True,
            "reason_code": "explicitly_enabled",
            "reason": "CARS comparison was explicitly requested by the caller.",
            "signals": ["explicit_request"],
        }
    if n_calibration < 30:
        return {
            **common,
            "mode": "auto",
            "evaluate_cars": False,
            "reason_code": "insufficient_calibration_samples",
            "reason": "Fewer than 30 calibration samples make autonomous CARS unstable.",
            "signals": [],
        }
    if n_wavelengths < 40:
        return {
            **common,
            "mode": "auto",
            "evaluate_cars": False,
            "reason_code": "insufficient_wavelengths",
            "reason": "Fewer than 40 wavelengths provide little scope for useful variable reduction.",
            "signals": [],
        }

    signals: list[str] = []
    if n_wavelengths >= 150:
        signals.append("many_wavelengths")
    if ratio >= 0.5:
        signals.append("high_wavelength_to_sample_ratio")
    if baseline_r2 is not None and baseline_r2 < 0.85:
        signals.append("weak_full_spectrum_tuning_fit")
    evaluate = bool(signals)
    return {
        **common,
        "mode": "auto",
        "evaluate_cars": evaluate,
        "reason_code": "selection_signals_detected" if evaluate else "full_spectrum_sufficient",
        "reason": ("CARS will be compared because: " + ", ".join(signals) + "." if evaluate else "The full-spectrum problem is small and well-conditioned; CARS is unlikely to justify its cost."),
        "signals": signals,
    }


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


def _choose_wavelength_candidate(
    candidate_results: list[dict],
    *,
    min_relative_improvement: float = 0.005,
) -> tuple[dict, dict]:
    """Adopt CARS only when it materially improves tuning RMSE."""
    full = next((item for item in candidate_results if item["method"] == "none"), None)
    cars = next((item for item in candidate_results if item["method"] == "cars"), None)
    if full is None:
        chosen = min(candidate_results, key=lambda item: item["RMSE_tuning"])
        return chosen, {"reason_code": "no_full_spectrum_candidate"}
    if cars is None:
        return full, {
            "reason_code": "cars_not_evaluated",
            "relative_RMSE_improvement": None,
        }
    denominator = max(abs(float(full["RMSE_tuning"])), np.finfo(float).eps)
    improvement = (float(full["RMSE_tuning"]) - float(cars["RMSE_tuning"])) / denominator
    if improvement >= float(min_relative_improvement):
        return cars, {
            "reason_code": "cars_improved_tuning_rmse",
            "relative_RMSE_improvement": float(improvement),
            "minimum_required_improvement": float(min_relative_improvement),
        }
    return full, {
        "reason_code": "cars_improvement_below_threshold",
        "relative_RMSE_improvement": float(improvement),
        "minimum_required_improvement": float(min_relative_improvement),
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


def _wavelength_candidate_summary(candidate_results: list[dict]) -> list[dict]:
    """Remove large index arrays while retaining auditable tuning evidence."""
    return [
        {
            "method": item["method"],
            "n_selected": int(item["n_selected"]),
            "n_components": int(item["n_components"]),
            "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
        }
        for item in candidate_results
    ]


def _decide_autonomous_model_selection(
    *,
    requested_method: str | None,
    n_calibration: int,
    n_features: int,
    baseline_rmse: float,
    y_tuning: np.ndarray,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
) -> dict:
    """Choose a bounded set of model families without consulting test data."""
    requested = (requested_method or "auto").strip().lower()
    if requested != "auto" and requested not in _MODEL_METHODS:
        raise ValueError(f"Unknown method {requested_method!r}; use auto or one of: {', '.join(_MODEL_METHODS)}")

    n_calibration = int(n_calibration)
    n_features = int(n_features)
    max_svr_samples = max(30, int(max_svr_samples))
    max_tree_samples = max(80, int(max_tree_samples))
    y_tuning = np.asarray(y_tuning, dtype=float).ravel()
    tuning_variance = float(np.var(y_tuning)) if y_tuning.size else 0.0
    baseline_r2 = float(1.0 - float(baseline_rmse) ** 2 / tuning_variance) if tuning_variance > 0 else None
    feature_ratio = float(n_features / max(1, n_calibration))
    common = {
        "n_calibration": n_calibration,
        "n_features": n_features,
        "feature_to_sample_ratio": feature_ratio,
        "pls_baseline_RMSE_tuning": float(baseline_rmse),
        "pls_baseline_R2_tuning": baseline_r2,
        "max_svr_samples": max_svr_samples,
        "max_tree_samples": max_tree_samples,
    }
    if requested != "auto":
        return {
            **common,
            "mode": "forced",
            "candidate_methods": [requested],
            "reason_code": "explicit_method",
            "reason": f"The caller explicitly requested method={requested!r}.",
            "signals": ["explicit_request"],
            "runtime_caps_applied": [],
        }
    if n_calibration < 30 or baseline_r2 is None:
        return {
            **common,
            "mode": "auto",
            "candidate_methods": ["pls"],
            "reason_code": "insufficient_model_selection_evidence",
            "reason": "The calibration/tuning data are insufficient for stable model-family comparison.",
            "signals": [],
            "runtime_caps_applied": [],
        }

    signals: list[str] = []
    if feature_ratio >= 0.5:
        signals.append("high_feature_to_sample_ratio")
    if baseline_r2 < 0.9:
        signals.append("weak_pls_tuning_fit")
    if baseline_r2 < 0.75:
        signals.append("possible_nonlinearity")

    candidates = ["pls"]
    if "high_feature_to_sample_ratio" in signals or "weak_pls_tuning_fit" in signals:
        candidates.append("ridge")

    runtime_caps: list[str] = []
    if "weak_pls_tuning_fit" in signals:
        if n_calibration <= max_svr_samples:
            candidates.append("svr")
        else:
            runtime_caps.append("svr")
    if "possible_nonlinearity" in signals and n_calibration >= 80:
        if n_calibration <= max_tree_samples:
            candidates.append("et")
        else:
            runtime_caps.append("et")

    return {
        **common,
        "mode": "auto",
        "candidate_methods": candidates,
        "reason_code": "model_comparison_signals_detected" if len(candidates) > 1 else "pls_baseline_sufficient",
        "reason": ("Alternative model families will be compared because: " + ", ".join(signals) + "." if len(candidates) > 1 else "The compact, strong PLS baseline does not justify additional model-family searches."),
        "signals": signals,
        "runtime_caps_applied": runtime_caps,
    }


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


def _choose_model_candidate(
    candidate_results: list[dict],
    *,
    min_relative_improvement: float = 0.01,
) -> tuple[dict, dict]:
    """Prefer PLS unless an alternative materially improves tuning RMSE."""
    if not candidate_results:
        raise ValueError("No successful model candidates are available")
    pls = next((item for item in candidate_results if item["method"] == "pls"), None)
    if pls is None:
        chosen = min(candidate_results, key=lambda item: float(item["RMSE_tuning"]))
        return chosen, {
            "reason_code": "explicit_or_no_pls_baseline",
            "relative_RMSE_improvement": None,
            "minimum_required_improvement": float(min_relative_improvement),
        }
    alternatives = [item for item in candidate_results if item["method"] != "pls"]
    if not alternatives:
        return pls, {
            "reason_code": "pls_only_candidate",
            "relative_RMSE_improvement": None,
            "minimum_required_improvement": float(min_relative_improvement),
        }
    best_alternative = min(alternatives, key=lambda item: float(item["RMSE_tuning"]))
    denominator = max(abs(float(pls["RMSE_tuning"])), np.finfo(float).eps)
    improvement = (float(pls["RMSE_tuning"]) - float(best_alternative["RMSE_tuning"])) / denominator
    if improvement >= float(min_relative_improvement):
        return best_alternative, {
            "reason_code": "alternative_improved_tuning_rmse",
            "relative_RMSE_improvement": float(improvement),
            "minimum_required_improvement": float(min_relative_improvement),
        }
    return pls, {
        "reason_code": "alternative_improvement_below_threshold",
        "relative_RMSE_improvement": float(improvement),
        "minimum_required_improvement": float(min_relative_improvement),
    }


def _model_candidate_summary(candidate_results: list[dict]) -> list[dict]:
    """Return JSON-safe, decision-grade model comparison evidence."""
    return [
        {
            "method": item["method"],
            "n_components": item.get("n_components"),
            "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            "RMSECV": None if item.get("RMSECV") is None else round(float(item["RMSECV"]), 5),
        }
        for item in candidate_results
    ]


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


def _write_plots_and_report(
    *,
    out_dir: str,
    spec_data,
    metrics: dict,
    quality: dict,
    best_pipe,
    y_te,
    y_pred_te,
    cv_results,
    best_n,
    method: str,
    model,
    X_tr,
    y_tr,
    wv,
) -> tuple[str, str, str, str, str, str]:
    """Generate plots + Markdown report under ``out_dir``.

    Returns ``(raw_b64, pred_b64, resid_b64, cv_b64, vip_b64, coef_b64)``
    so the caller can decide which paths to surface in the JSON payload.
    """
    import base64

    from nir_core.plotting.model_diag import (
        plot_cv_curve,
        plot_predicted_vs_reference,
        plot_residuals,
    )
    from nir_core.plotting.spectra import plot_raw_spectra

    pred_b64 = plot_predicted_vs_reference(y_te, y_pred_te)
    resid_b64 = plot_residuals(y_te, y_pred_te)
    raw_b64 = plot_raw_spectra(spec_data, n_highlight=5)

    cv_b64 = ""
    if isinstance(cv_results, dict) and cv_results.get("n_components"):
        cv_b64 = plot_cv_curve(
            cv_results["n_components"],
            cv_results["mean_rmse_cv"],
            best_n,
            cv_results.get("std_rmse_cv"),
        )

    for fname, b64 in [
        ("predicted_vs_reference.png", pred_b64),
        ("residuals.png", resid_b64),
        ("raw_spectra.png", raw_b64),
        ("cv_curve.png", cv_b64),
    ]:
        if b64:
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(base64.b64decode(b64))

    # ★ v3: VIP and regression coefficient plots (PLS only).
    vip_b64 = ""
    coef_b64 = ""
    if method == "pls":
        try:
            from nir_core.model.pls import compute_vip, get_regression_coefficients
            from nir_core.plotting.model_diag import plot_regression_coefficients, plot_vip

            vip_scores = compute_vip(model, X_tr, y_tr)
            coef = get_regression_coefficients(model)
            vip_b64 = plot_vip(vip_scores, wv=wv)
            coef_b64 = plot_regression_coefficients(coef, wv=wv)
            for fname, b64 in [("vip_scores.png", vip_b64), ("regression_coefficients.png", coef_b64)]:
                if b64:
                    with open(os.path.join(out_dir, fname), "wb") as f:
                        f.write(base64.b64decode(b64))
        except Exception as exc:
            warnings.warn(f"VIP plot generation failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    report = _build_report(
        spec_data,
        metrics,
        quality,
        best_pipe,
        raw_spectra_b64=raw_b64,
        predicted_vs_reference_b64=pred_b64,
        residuals_b64=resid_b64,
        cv_curve_b64=cv_b64,
    )
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return raw_b64, pred_b64, resid_b64, cv_b64, vip_b64, coef_b64


# ---------------------------------------------------------------------------
# nir_train_model
# ---------------------------------------------------------------------------


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

        import joblib
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

        real_in = _resolve(runtime, input_path, read_only=True)
        data_dict = dict(np.load(real_in, allow_pickle=True))
        X = np.asarray(data_dict["X"], dtype=float)
        y = np.asarray(data_dict["y"], dtype=float).ravel()
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")

        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            random_state=42,
        )

        # ★ v3: Leakage-safe inline preprocessing when pipeline_steps given.
        best_pipe = None
        preprocessing_desc = "none"
        steps_list: list = []
        if pipeline_steps is not None:
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

        model, best_n, cv_results, predict_fn = _train_one_model(method, X_tr, y_tr, max_components=max_components, cv_folds=cv_folds, cv_strategy=cv_strategy)
        y_pred_tr = predict_fn(model, X_tr)
        y_pred_val = predict_fn(model, X_val)
        y_pred_te = predict_fn(model, X_te)

        metrics = {
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
        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        joblib.dump(
            _build_model_artifact(
                model,
                method=method,
                preprocessing_pipeline=best_pipe,
                preprocessing_desc=preprocessing_desc,
                wavelength_selection=wavelength_selection_meta,
            ),
            real_model,
        )

        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)

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
    except MemoryError:
        return _err("内存不足: 数据集过大或预处理候选过多。请减少候选流水线数量或使用更小子集。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Fast single-component CSV protocols
# ---------------------------------------------------------------------------


_GROUP_FIELD_PATTERNS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (100, ("instrument", "spectrometer", "device", "仪器", "设备")),
    (95, ("batch", "lot", "批次", "批号")),
    (90, ("season", "year", "季节", "年份")),
    (85, ("region", "origin", "site", "farm", "orchard", "产地", "地区", "站点", "农场", "果园")),
    (80, ("cultivar", "variety", "population", "品种", "种类", "群体")),
)


def _split_counts(n_samples: int, *, tuning_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """Validate ratios and return exact calibration/tuning/test counts."""
    if n_samples < 6:
        raise ValueError("At least 6 samples are required for a three-way split")
    if not (0.0 < tuning_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("tuning_ratio and test_ratio must both be in (0, 1)")
    if tuning_ratio + test_ratio >= 1.0:
        raise ValueError("tuning_ratio + test_ratio must be less than 1")

    n_tuning = max(1, int(round(n_samples * tuning_ratio)))
    n_test = max(1, int(round(n_samples * test_ratio)))
    n_calibration = n_samples - n_tuning - n_test
    if n_calibration < 3:
        raise ValueError("The requested split leaves fewer than 3 calibration samples")
    return n_calibration, n_tuning, n_test


def _allocate_ordered_indices(
    order: np.ndarray,
    *,
    n_calibration: int,
    n_tuning: int,
    n_test: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distribute a representative sample order with weighted round-robin."""
    targets = np.asarray([n_calibration, n_tuning, n_test], dtype=int)
    current = np.zeros(3, dtype=int)
    buckets: list[list[int]] = [[], [], []]
    n_total = int(targets.sum())
    if len(order) != n_total:
        raise ValueError(f"Split order has {len(order)} rows, expected {n_total}")

    for position, raw_index in enumerate(np.asarray(order, dtype=int)):
        progress = float(position + 1) / float(n_total)
        deficits = targets * progress - current
        deficits[current >= targets] = -np.inf
        destination = int(np.argmax(deficits))
        buckets[destination].append(int(raw_index))
        current[destination] += 1

    return tuple(np.sort(np.asarray(bucket, dtype=int)) for bucket in buckets)  # type: ignore[return-value]


def _deterministic_holdout_indices(
    n_samples: int,
    *,
    tuning_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return reproducible calibration/tuning/test row indices."""
    n_calibration, n_tuning, n_test = _split_counts(
        n_samples,
        tuning_ratio=tuning_ratio,
        test_ratio=test_ratio,
    )
    order = np.random.RandomState(random_state).permutation(n_samples)
    return _allocate_ordered_indices(
        order,
        n_calibration=n_calibration,
        n_tuning=n_tuning,
        n_test=n_test,
    )


def _y_stratified_holdout_indices(
    y: np.ndarray,
    *,
    tuning_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spread adjacent target ranks across all partitions without bin failures."""
    y = np.asarray(y, dtype=float).ravel()
    n_calibration, n_tuning, n_test = _split_counts(len(y), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    sorted_indices = np.argsort(y, kind="mergesort")
    rng = np.random.RandomState(random_state)
    blocks: list[np.ndarray] = []
    for start in range(0, len(y), 20):
        block = sorted_indices[start : start + 20].copy()
        rng.shuffle(block)
        blocks.append(block)
    order = np.concatenate(blocks)
    return _allocate_ordered_indices(
        order,
        n_calibration=n_calibration,
        n_tuning=n_tuning,
        n_test=n_test,
    )


def _spxy_holdout_indices(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tuning_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a joint spectral/target maximin order and distribute it evenly."""
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_calibration, n_tuning, n_test = _split_counts(len(y), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    feature_std = np.std(X, axis=0, ddof=0)
    feature_std[feature_std < 1e-12] = 1.0
    X_scaled = (X - np.mean(X, axis=0)) / feature_std
    spectral_distance = pairwise_distances(X_scaled, metric="euclidean")
    spectral_max = float(np.max(spectral_distance))
    if spectral_max > 0:
        spectral_distance /= spectral_max

    y_distance = np.abs(y[:, None] - y[None, :])
    y_max = float(np.max(y_distance))
    if y_max > 0:
        y_distance /= y_max
    joint_distance = spectral_distance + y_distance
    np.fill_diagonal(joint_distance, -np.inf)

    first, second = np.unravel_index(int(np.argmax(joint_distance)), joint_distance.shape)
    selected = np.zeros(len(y), dtype=bool)
    selected[[first, second]] = True
    order = [int(first), int(second)]
    min_distance = np.minimum(joint_distance[:, first], joint_distance[:, second])
    min_distance[selected] = -np.inf
    while len(order) < len(y):
        next_index = int(np.argmax(min_distance))
        if selected[next_index]:
            next_index = int(np.flatnonzero(~selected)[0])
        order.append(next_index)
        selected[next_index] = True
        min_distance = np.minimum(min_distance, joint_distance[:, next_index])
        min_distance[selected] = -np.inf

    return _allocate_ordered_indices(
        np.asarray(order, dtype=int),
        n_calibration=n_calibration,
        n_tuning=n_tuning,
        n_test=n_test,
    )


def _group_field_candidates(raw, *, y_col: int) -> list[dict]:
    """Rank plausible batch/domain grouping columns and reject ID-like fields."""
    n_samples = len(raw)
    max_groups = max(5, min(100, int(round(n_samples * 0.25))))
    candidates: list[dict] = []
    for column_index, column in enumerate(raw.columns):
        if column_index == int(y_col):
            continue
        name = str(column)
        normalized = "".join(character.lower() for character in name if character.isalnum() or "\u4e00" <= character <= "\u9fff")
        if any(token in normalized for token in ("sampleid", "recordid", "样本编号", "样本id")) or normalized in {"id", "index", "序号"}:
            continue
        score = next((weight for weight, patterns in _GROUP_FIELD_PATTERNS if any(pattern in normalized for pattern in patterns)), None)
        if score is None:
            continue
        values = raw[column]
        missing_fraction = float(values.isna().mean())
        n_groups = int(values.nunique(dropna=True))
        eligible = missing_fraction <= 0.20 and 5 <= n_groups <= max_groups
        candidates.append(
            {
                "column": name,
                "score": int(score),
                "n_groups": n_groups,
                "missing_fraction": missing_fraction,
                "eligible": eligible,
            }
        )
    return sorted(candidates, key=lambda item: (-item["eligible"], -item["score"], item["n_groups"], item["column"]))


def _choose_group_subset(values: np.ndarray, available_groups: list[str], *, target_samples: int, rng, min_groups_left: int) -> set[str]:
    """Greedily choose whole groups nearest a requested sample count."""
    shuffled = list(available_groups)
    rng.shuffle(shuffled)
    counts = {group: int(np.sum(values == group)) for group in shuffled}
    selected: set[str] = set()
    selected_count = 0
    while len(shuffled) - len(selected) > min_groups_left:
        remaining = [group for group in shuffled if group not in selected]
        best = min(remaining, key=lambda group: abs(selected_count + counts[group] - target_samples))
        new_distance = abs(selected_count + counts[best] - target_samples)
        current_distance = abs(selected_count - target_samples)
        if selected and new_distance >= current_distance:
            break
        selected.add(best)
        selected_count += counts[best]
    if not selected:
        selected.add(min(shuffled, key=lambda group: abs(counts[group] - target_samples)))
    return selected


def _group_holdout_indices(
    values,
    *,
    tuning_ratio: float,
    test_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, list[str]]]:
    """Split by whole groups so no batch/domain appears in two partitions."""
    group_values = np.asarray(["<missing>" if value is None or str(value) == "nan" else str(value) for value in values], dtype=object)
    groups = sorted(set(group_values.tolist()))
    if len(groups) < 3:
        raise ValueError("Group splitting requires at least three distinct groups")
    _, n_tuning, n_test = _split_counts(len(group_values), tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    rng = np.random.RandomState(random_state)
    test_groups = _choose_group_subset(group_values, groups, target_samples=n_test, rng=rng, min_groups_left=2)
    remaining = [group for group in groups if group not in test_groups]
    tuning_groups = _choose_group_subset(group_values, remaining, target_samples=n_tuning, rng=rng, min_groups_left=1)
    calibration_groups = set(remaining) - tuning_groups
    calibration = np.flatnonzero(np.isin(group_values, list(calibration_groups)))
    tuning = np.flatnonzero(np.isin(group_values, list(tuning_groups)))
    holdout = np.flatnonzero(np.isin(group_values, list(test_groups)))
    if min(len(calibration), len(tuning), len(holdout)) == 0:
        raise ValueError("Group splitting produced an empty partition")
    return (
        np.sort(calibration),
        np.sort(tuning),
        np.sort(holdout),
        {
            "calibration": sorted(calibration_groups),
            "tuning": sorted(tuning_groups),
            "holdout_test": sorted(test_groups),
        },
    )


def _select_autonomous_split(
    raw,
    X: np.ndarray,
    y: np.ndarray,
    *,
    y_col: int,
    strategy: str,
    group_col: str | None,
    tuning_ratio: float,
    test_ratio: float,
    random_state: int,
    spxy_max_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Select and execute a deterministic split using fields and sample size."""
    requested = (strategy or "auto").strip().lower().replace("-", "_")
    aliases = {"stratified": "y_stratified", "stratified_y": "y_stratified", "ks": "spxy"}
    requested = aliases.get(requested, requested)
    allowed = {"auto", "group", "spxy", "y_stratified", "random"}
    if requested not in allowed:
        raise ValueError(f"Unknown split_strategy {strategy!r}; use one of: {', '.join(sorted(allowed))}")
    if int(spxy_max_samples) < 6:
        raise ValueError("spxy_max_samples must be at least 6")

    candidates = _group_field_candidates(raw, y_col=y_col)
    selected_group = group_col
    if selected_group is not None and selected_group not in raw.columns:
        raise ValueError(f"group_col {selected_group!r} is not a CSV header")

    effective = requested
    reason = ""
    if requested == "auto":
        if selected_group is None:
            selected_group = next((item["column"] for item in candidates if item["eligible"]), None)
        if selected_group is not None:
            effective = "group"
            reason = f"Detected grouping field {selected_group!r}; whole groups are isolated to prevent batch/domain leakage."
        elif len(y) <= int(spxy_max_samples):
            effective = "spxy"
            reason = f"No eligible grouping field; n_samples={len(y)} is within the SPXY limit {int(spxy_max_samples)}."
        else:
            effective = "y_stratified"
            reason = f"No eligible grouping field; n_samples={len(y)} exceeds the SPXY limit {int(spxy_max_samples)}, so quadratic distances are avoided."
    elif requested == "group":
        if selected_group is None:
            selected_group = next((item["column"] for item in candidates if item["eligible"]), None)
        if selected_group is None:
            raise ValueError("split_strategy='group' requires group_col or an eligible detected grouping field")
        reason = f"Group split explicitly requested with field {selected_group!r}."
    elif requested == "spxy":
        if len(y) > int(spxy_max_samples):
            raise ValueError(f"SPXY requested for {len(y)} samples, above spxy_max_samples={int(spxy_max_samples)}")
        reason = "SPXY explicitly requested; joint spectral and target distances determine a balanced maximin order."
    elif requested == "y_stratified":
        reason = "Target-value stratification explicitly requested."
    else:
        reason = "Deterministic random split explicitly requested."

    group_assignments = None
    if effective == "group":
        calibration, tuning, holdout, group_assignments = _group_holdout_indices(
            raw[selected_group].to_numpy(),
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    elif effective == "spxy":
        calibration, tuning, holdout = _spxy_holdout_indices(X, y, tuning_ratio=tuning_ratio, test_ratio=test_ratio)
    elif effective == "y_stratified":
        calibration, tuning, holdout = _y_stratified_holdout_indices(
            y,
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )
    else:
        calibration, tuning, holdout = _deterministic_holdout_indices(
            len(y),
            tuning_ratio=tuning_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

    n_total = len(y)
    decision = {
        "requested_strategy": requested,
        "strategy": effective,
        "reason": reason,
        "random_state": int(random_state),
        "spxy_max_samples": int(spxy_max_samples),
        "group_column": selected_group if effective == "group" else None,
        "group_field_candidates": candidates,
        "group_assignments": group_assignments,
        "actual_ratios": {
            "calibration": float(len(calibration) / n_total),
            "tuning": float(len(tuning) / n_total),
            "holdout_test": float(len(holdout) / n_total),
        },
    }
    return calibration, tuning, holdout, decision


@tool("nir_train_auto_split_model", parse_docstring=True)
def nir_train_auto_split_model_tool(
    runtime: Runtime,
    file_path: str,
    y_col: int,
    x_cols: str,
    wv_row: int | None = 0,
    tuning_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    split_strategy: str = "auto",
    group_col: str | None = None,
    spxy_max_samples: int = 500,
    pipeline_steps: str = '["snv", "autoscale"]',
    method: str = "auto",
    max_components: int = 20,
    compare_cars: bool | None = None,
    cars_params: str = '{"n_mc_samples": 20, "n_folds": 5, "random_state": 42}',
    max_cars_samples: int = 750,
    min_cars_improvement: float = 0.005,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    model_output: str = "/mnt/user-data/outputs/auto_split_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/auto_split_metrics.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train an autonomously selected model for a CSV without official partitions.

    The tool autonomously selects a reproducible calibration/tuning/holdout
    split. It prefers eligible batch/domain metadata fields, uses SPXY for
    smaller ungrouped datasets, and falls back to target-value stratification
    above the SPXY size limit. All
    preprocessing and optional CARS selection are fitted on calibration rows
    only. Tuning rows select the wavelength candidate and, by default, a
    bounded model family from PLS/Ridge/SVR/Extra Trees. Holdout rows are
    evaluated once after all choices are fixed. The holdout metrics are not
    presented as external validation.

    Args:
        file_path: Virtual path to the original labelled CSV file.
        y_col: 0-based numeric target-column index in the CSV.
        x_cols: Spectral-column selector passed to nir_core, for example ``"9:"``.
        wv_row: Wavelength/header row index for mixed text/numeric CSV layouts.
        tuning_ratio: Fraction reserved for model selection; default 0.15.
        test_ratio: Fraction reserved for one-time holdout testing; default 0.15.
        random_state: Reproducible split seed.
        split_strategy: ``auto`` (default), ``group``, ``spxy``,
            ``y_stratified``, or ``random``.
        group_col: Optional explicit batch/domain column for group splitting.
        spxy_max_samples: Largest sample count eligible for automatic SPXY;
            larger ungrouped datasets use target-value stratification.
        pipeline_steps: JSON preprocessing pipeline fitted on calibration rows.
        method: ``auto`` (default) for bounded model-family selection, or an
            explicit supported model such as ``pls`` or ``svr``.
        max_components: Maximum PLS latent variables considered on tuning rows.
        compare_cars: Optional override. None (default) lets the runtime decide;
            true forces a CARS comparison and false disables it.
        cars_params: JSON CARS parameters used when CARS is evaluated.
        max_cars_samples: Maximum calibration samples used to fit CARS. Larger
            sets use deterministic target-rank coverage, still calibration-only.
        min_cars_improvement: Minimum relative Tuning RMSE improvement required
            before the selected CARS wavelengths replace the full spectrum.
        max_svr_samples: Largest calibration set eligible for autonomous SVR.
        max_tree_samples: Largest calibration set eligible for autonomous
            Extra Trees comparison.
        min_model_improvement: Minimum relative Tuning RMSE improvement an
            alternative must achieve before replacing PLS.
        model_output: Virtual output path for a deployable model artifact.
        metrics_output: Virtual output path for holdout metrics and split indices.
        domain: Application domain used by the quality gate.

    Returns:
        JSON containing the deterministic split, tuning and holdout metrics,
        selected model details, and persisted artifact/report paths.
    """
    try:
        import json

        import joblib
        import pandas as pd
        from nir_core.io.loaders import load_csv
        from nir_core.model.evaluation import compute_metrics
        from nir_core.model.selection import cars_wavelength_selection
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.utils.metrics import evaluate_quality

        real_in = _resolve(runtime, file_path, read_only=True)
        raw = pd.read_csv(real_in)
        if not (0 <= int(y_col) < raw.shape[1]):
            return _err(f"y_col {y_col} is outside the CSV column range 0..{raw.shape[1] - 1}")
        target_name = str(raw.columns[int(y_col)])
        data = load_csv(real_in, y_col=int(y_col), x_cols=x_cols, wv_row=wv_row)
        if data.y is None or data.wv is None:
            return _err("Auto-split calibration requires both a target column and wavelength headers.")

        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel()
        wv = np.asarray(data.wv, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            return _err("Auto-split calibration requires finite X and y values.")

        calibration_indices, tuning_indices, test_indices, split_decision = _select_autonomous_split(
            raw,
            X,
            y,
            y_col=int(y_col),
            strategy=split_strategy,
            group_col=group_col,
            tuning_ratio=float(tuning_ratio),
            test_ratio=float(test_ratio),
            random_state=int(random_state),
            spxy_max_samples=int(spxy_max_samples),
        )
        X_cal, y_cal = X[calibration_indices], y[calibration_indices]
        X_tune, y_tune = X[tuning_indices], y[tuning_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        try:
            raw_steps = json.loads(pipeline_steps)
            if not isinstance(raw_steps, list):
                return _err("pipeline_steps must be a JSON list")
            steps = [_parse_pipeline_step(value) for value in raw_steps]
        except (TypeError, ValueError) as exc:
            return _err(f"Invalid pipeline_steps JSON: {exc}")
        is_valid, reason = validate_pipeline(steps)
        if not is_valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")

        selection_pipe = PreprocessingPipeline(steps).fit(X_cal, wv)
        X_cal_s = selection_pipe.transform(X_cal, wv)
        X_tune_s = selection_pipe.transform(X_tune, wv)

        full_result = _score_pls_wavelength_candidate(
            {"method": "none", "indices": list(range(X.shape[1])), "params": {}},
            X_cal_s,
            y_cal,
            X_tune_s,
            y_tune,
            max_components=int(max_components),
        )
        selection_decision = _decide_autonomous_wavelength_selection(
            compare_cars=compare_cars,
            n_calibration=X_cal_s.shape[0],
            n_wavelengths=X_cal_s.shape[1],
            baseline_rmse=full_result["RMSE_tuning"],
            y_tuning=y_tune,
            max_selection_samples=int(max_cars_samples),
        )
        candidate_results: list[dict] = [full_result]
        if selection_decision["evaluate_cars"]:
            parsed_cars = json.loads(cars_params)
            if not isinstance(parsed_cars, dict):
                return _err("cars_params must be a JSON object")
            selection_rows = _target_rank_selection_indices(y_cal, selection_decision["cars_fit_samples"])
            _, cars_indices = cars_wavelength_selection(
                X_cal_s[selection_rows],
                y_cal[selection_rows],
                n_mc_samples=int(parsed_cars.get("n_mc_samples", 20)),
                n_folds=int(parsed_cars.get("n_folds", 5)),
                random_state=int(parsed_cars.get("random_state", random_state)),
            )
            candidate_results.append(
                _score_pls_wavelength_candidate(
                    {
                        "method": "cars",
                        "indices": cars_indices,
                        "params": parsed_cars,
                        "selection_fit_samples": int(selection_rows.size),
                    },
                    X_cal_s,
                    y_cal,
                    X_tune_s,
                    y_tune,
                    max_components=int(max_components),
                )
            )

        chosen, adoption_decision = _choose_wavelength_candidate(
            candidate_results,
            min_relative_improvement=float(min_cars_improvement),
        )
        selection_decision["adoption"] = adoption_decision
        selected_indices = [int(index) for index in chosen["indices"]]
        X_cal_selected = X_cal_s[:, selected_indices]
        X_tune_selected = X_tune_s[:, selected_indices]
        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            method,
            X_cal_selected,
            y_cal,
            X_tune_selected,
            y_tune,
            max_components=int(max_components),
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )
        tuning_prediction = chosen_model["_tuning_prediction"]

        final_train_indices = np.sort(np.concatenate([calibration_indices, tuning_indices]))
        final_pipe = PreprocessingPipeline(steps).fit(X[final_train_indices], wv)
        X_final = final_pipe.transform(X[final_train_indices], wv)[:, selected_indices]
        X_holdout = final_pipe.transform(X_test, wv)[:, selected_indices]
        final_model, best_n, cv_results, final_predict_fn = _train_one_model(
            chosen_model["method"],
            X_final,
            y[final_train_indices],
            max_components=int(max_components),
            cv_folds=5,
            cv_strategy="auto",
        )
        train_prediction = final_predict_fn(final_model, X_final)
        holdout_prediction = final_predict_fn(final_model, X_holdout)

        selection_meta = {
            "method": chosen["method"],
            "params": chosen["params"],
            "n_original": int(X.shape[1]),
            "n_selected": int(len(selected_indices)),
            "selected_indices": None if chosen["method"] == "none" else selected_indices,
            "selected_wavelengths": None if chosen["method"] == "none" else [float(wv[index]) for index in selected_indices],
        }
        partition_metrics = {
            "calibration": {"n_samples": int(calibration_indices.size), "sample_indices": calibration_indices.tolist()},
            "tuning": {"n_samples": int(tuning_indices.size), "sample_indices": tuning_indices.tolist()},
            "holdout_test": {"n_samples": int(test_indices.size), "sample_indices": test_indices.tolist()},
        }
        group_assignments = split_decision.get("group_assignments")
        if group_assignments:
            for partition_name, group_values in group_assignments.items():
                partition_metrics[partition_name]["group_values"] = group_values

        metrics = {
            "protocol": "deterministic_auto_split_holdout",
            "validation_scope": "independent_holdout_not_external",
            "target": target_name,
            "method": chosen_model["method"],
            "n_components": None if best_n is None else int(best_n),
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(X.shape[1]),
            "n_wavelengths_model": int(len(selected_indices)),
            "preprocessing": final_pipe.description(),
            "preprocessing_steps": raw_steps,
            "wavelength_selection": selection_meta,
            "wavelength_selection_decision": selection_decision,
            "model_selection_decision": model_selection_decision,
            "split": {
                **split_decision,
                "requested_ratios": {
                    "calibration": float(1.0 - tuning_ratio - test_ratio),
                    "tuning": float(tuning_ratio),
                    "holdout_test": float(test_ratio),
                },
            },
            "partitions": partition_metrics,
            "candidate_results": candidate_results,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y[final_train_indices], train_prediction),
            "val": compute_metrics(y_tune, tuning_prediction),
            "test": compute_metrics(y_test, holdout_prediction),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        metrics["RMSECV"] = metrics["val"]["RMSE"]
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(final_train_indices.size))
        metrics["quality"] = quality

        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        joblib.dump(
            _build_model_artifact(
                final_model,
                method=chosen_model["method"],
                preprocessing_pipeline=final_pipe,
                preprocessing_desc=final_pipe.description(),
                wavelength_selection=selection_meta,
            ),
            real_model,
        )
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)

        report_path = os.path.join(os.path.dirname(real_model), "auto_split_report.md")
        report = [
            "# Auto-split NIR holdout report",
            "",
            f"- Target: `{target_name}`",
            f"- Split strategy: `{split_decision['strategy']}` (requested `{split_decision['requested_strategy']}`)",
            f"- Split reason: {split_decision['reason']}",
            f"- Random seed: {int(random_state)}",
            f"- Samples: {calibration_indices.size} calibration / {tuning_indices.size} tuning / {test_indices.size} holdout",
            "- Validation scope: independent holdout generated from this dataset; not external validation",
            f"- Pipeline: {final_pipe.description()}",
            f"- Wavelength-selection policy: {selection_decision['mode']} — {selection_decision['reason']}",
            f"- Selection: {chosen['method']} ({len(selected_indices)} of {X.shape[1]} wavelengths)",
            f"- Model-selection policy: {model_selection_decision['mode']} - {model_selection_decision['reason']}",
            f"- Selected model: {chosen_model['method']}",
            f"- Latent variables: {best_n if best_n is not None else 'not applicable'}",
            f"- Tuning RMSE: {metrics['val']['RMSE']:.5f}",
            f"- Holdout RMSE: {metrics['test']['RMSE']:.5f}",
            f"- Holdout R2: {metrics['test']['R2']:.5f}",
            f"- Holdout RPD: {metrics['test']['RPD']:.5f}",
            f"- Holdout bias: {metrics['test']['bias']:.5f}",
        ]
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")

        out_virtual = os.path.dirname(model_output)
        candidate_summary = [
            {
                "method": item["method"],
                "n_selected": item["n_selected"],
                "n_components": item["n_components"],
                "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            }
            for item in candidate_results
        ]
        return _ok(
            {
                "status": "ok",
                "protocol": "deterministic_auto_split_holdout",
                "validation_scope": "independent_holdout_not_external",
                "target": target_name,
                "split": metrics["split"],
                "partitions": {name: {"n_samples": value["n_samples"]} for name, value in metrics["partitions"].items()},
                "preprocessing": final_pipe.description(),
                "wavelength_selection": selection_meta,
                "wavelength_selection_decision": selection_decision,
                "model_selection_decision": model_selection_decision,
                "method": chosen_model["method"],
                "n_components": None if best_n is None else int(best_n),
                "RMSE_tuning": round(float(metrics["val"]["RMSE"]), 5),
                "holdout": {key: round(float(metrics["test"][key]), 5) for key in ("RMSE", "R2", "RPD", "bias")},
                "candidate_results": candidate_summary,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "report": out_virtual + "/auto_split_report.md",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_train_partitioned_model", parse_docstring=True)
def nir_train_partitioned_model_tool(
    runtime: Runtime,
    file_path: str,
    split_col: str,
    train_label: str,
    tuning_label: str,
    test_label: str,
    y_col: int,
    x_cols: str,
    wv_row: int | None = 0,
    pipeline_steps: str = '["snv", {"method": "derivative1", "params": {"window": 15, "order": 2}}, "autoscale"]',
    method: str = "auto",
    max_components: int = 20,
    compare_cars: bool | None = None,
    cars_params: str = '{"n_mc_samples": 50, "n_folds": 5, "random_state": 42}',
    max_cars_samples: int = 750,
    min_cars_improvement: float = 0.005,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    model_output: str = "/mnt/user-data/outputs/partitioned_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/partitioned_metrics.json",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train an autonomously selected model on named data partitions.

    This is intended for datasets whose CSV records carry an official split
    column, rather than a random split.  It enforces the following protocol:
    preprocessing and CARS selection are fitted on ``train_label`` only;
    ``tuning_label`` selects wavelengths and a bounded model family;
    ``test_label`` is used once for final external metrics. This prevents a
    held-out season, batch, or instrument set from leaking into any selection
    decision.

    Args:
        file_path: Virtual path to the original labelled CSV file.
        split_col: Header name containing named partitions, for example ``Set``.
        train_label: Value in split_col used for calibration, for example ``Cal``.
        tuning_label: Value used for model selection, for example ``Tuning``.
        test_label: Value reserved for external testing, for example ``Val Ext``.
        y_col: 0-based numeric target-column index in the CSV.
        x_cols: Spectral-column selector passed to nir_core, for example ``"9:"``.
        wv_row: Wavelength/header row index for mixed text/numeric CSV layouts.
        pipeline_steps: JSON preprocessing pipeline, fitted on Cal only during selection.
        method: ``auto`` (default) or an explicit supported model family.
        max_components: Maximum PLS latent variables considered on Tuning.
        compare_cars: Optional override. None (default) autonomously decides,
            true forces CARS comparison, and false disables it.
        cars_params: JSON CARS parameters used when CARS is evaluated.
        max_cars_samples: Maximum calibration samples used to fit CARS.
        min_cars_improvement: Minimum relative Tuning RMSE improvement required
            before adopting the CARS subset.
        max_svr_samples: Largest Cal set eligible for autonomous SVR.
        max_tree_samples: Largest Cal set eligible for autonomous Extra Trees.
        min_model_improvement: Minimum relative Tuning RMSE improvement required
            before an alternative model replaces PLS.
        model_output: Virtual output path for a deployable model artifact.
        metrics_output: Virtual output path for external-validation metrics.

    Returns:
        JSON containing tuning and external metrics, selected wavelengths, and
        paths to the persisted model, metrics, and Markdown report.
    """
    try:
        import json

        import joblib
        import pandas as pd
        from nir_core.io.loaders import load_csv
        from nir_core.model.evaluation import compute_metrics
        from nir_core.model.selection import cars_wavelength_selection
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.utils.metrics import evaluate_quality

        real_in = _resolve(runtime, file_path, read_only=True)
        raw = pd.read_csv(real_in)
        if split_col not in raw.columns:
            return _err(f"split_col {split_col!r} is not a CSV header; available columns include {list(raw.columns[:12])}")

        data = load_csv(real_in, y_col=y_col, x_cols=x_cols, wv_row=wv_row)
        labels = raw[split_col].astype(str).to_numpy()
        if labels.shape[0] != data.X.shape[0]:
            return _err(f"Split column has {labels.shape[0]} rows but loaded spectra have {data.X.shape[0]}. Use a one-row CSV header and a wv_row that is not a sample row.")
        if data.y is None or data.wv is None:
            return _err("Partitioned calibration requires both y_col and a wavelength row.")

        train_mask = labels == str(train_label)
        tune_mask = labels == str(tuning_label)
        test_mask = labels == str(test_label)
        if not (train_mask.any() and tune_mask.any() and test_mask.any()):
            counts = {value: int((labels == value).sum()) for value in (str(train_label), str(tuning_label), str(test_label))}
            return _err(f"Each partition must contain samples; observed counts: {counts}")
        if np.any((train_mask.astype(int) + tune_mask.astype(int) + test_mask.astype(int)) > 1):
            return _err("train_label, tuning_label, and test_label must be distinct.")

        try:
            raw_steps = json.loads(pipeline_steps)
            steps = [_parse_pipeline_step(value) for value in raw_steps]
        except (TypeError, ValueError) as exc:
            return _err(f"Invalid pipeline_steps JSON: {exc}")
        is_valid, reason = validate_pipeline(steps)
        if not is_valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")

        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel()
        wv = np.asarray(data.wv, dtype=float).ravel()
        X_cal, y_cal = X[train_mask], y[train_mask]
        X_tune, y_tune = X[tune_mask], y[tune_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # Fit all stateful transforms on Cal only for every tuning decision.
        selection_pipe = PreprocessingPipeline(steps).fit(X_cal, wv)
        X_cal_s = selection_pipe.transform(X_cal, wv)
        X_tune_s = selection_pipe.transform(X_tune, wv)

        full_result = _score_pls_wavelength_candidate(
            {"method": "none", "indices": list(range(X.shape[1])), "params": {}},
            X_cal_s,
            y_cal,
            X_tune_s,
            y_tune,
            max_components=int(max_components),
        )
        selection_decision = _decide_autonomous_wavelength_selection(
            compare_cars=compare_cars,
            n_calibration=X_cal_s.shape[0],
            n_wavelengths=X_cal_s.shape[1],
            baseline_rmse=full_result["RMSE_tuning"],
            y_tuning=y_tune,
            max_selection_samples=int(max_cars_samples),
        )
        candidate_results: list[dict] = [full_result]
        if selection_decision["evaluate_cars"]:
            try:
                parsed_cars = json.loads(cars_params)
                if not isinstance(parsed_cars, dict):
                    return _err("cars_params must be a JSON object")
                selection_rows = _target_rank_selection_indices(y_cal, selection_decision["cars_fit_samples"])
                _, cars_indices = cars_wavelength_selection(
                    X_cal_s[selection_rows],
                    y_cal[selection_rows],
                    n_mc_samples=int(parsed_cars.get("n_mc_samples", 50)),
                    n_folds=int(parsed_cars.get("n_folds", 5)),
                    random_state=int(parsed_cars.get("random_state", 42)),
                )
                candidate_results.append(
                    _score_pls_wavelength_candidate(
                        {
                            "method": "cars",
                            "indices": cars_indices,
                            "params": parsed_cars,
                            "selection_fit_samples": int(selection_rows.size),
                        },
                        X_cal_s,
                        y_cal,
                        X_tune_s,
                        y_tune,
                        max_components=int(max_components),
                    )
                )
            except (TypeError, ValueError) as exc:
                return _err(f"CARS selection failed: {exc}")

        chosen, adoption_decision = _choose_wavelength_candidate(
            candidate_results,
            min_relative_improvement=float(min_cars_improvement),
        )
        selection_decision["adoption"] = adoption_decision
        selected_indices = [int(index) for index in chosen["indices"]]
        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            method,
            X_cal_s[:, selected_indices],
            y_cal,
            X_tune_s[:, selected_indices],
            y_tune,
            max_components=int(max_components),
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )

        # Final fitting may incorporate Tuning data, but never external test data.
        final_train_mask = train_mask | tune_mask
        final_pipe = PreprocessingPipeline(steps).fit(X[final_train_mask], wv)
        X_final = final_pipe.transform(X[final_train_mask], wv)[:, selected_indices]
        X_external = final_pipe.transform(X_test, wv)[:, selected_indices]
        final_model, best_n, cv_results, final_predict_fn = _train_one_model(
            chosen_model["method"],
            X_final,
            y[final_train_mask],
            max_components=int(max_components),
            cv_folds=5,
            cv_strategy="auto",
        )
        y_pred_train = final_predict_fn(final_model, X_final)
        y_pred_external = final_predict_fn(final_model, X_external)

        selection_meta = {
            "method": chosen["method"],
            "params": chosen["params"],
            "n_original": int(X.shape[1]),
            "n_selected": int(len(selected_indices)),
            "selected_indices": None if chosen["method"] == "none" else selected_indices,
            "selected_wavelengths": None if chosen["method"] == "none" else [float(wv[index]) for index in selected_indices],
        }
        metrics = {
            "method": chosen_model["method"],
            "n_components": None if best_n is None else int(best_n),
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(X.shape[1]),
            "n_wavelengths_model": int(len(selected_indices)),
            "preprocessing": final_pipe.description(),
            "preprocessing_steps": raw_steps,
            "wavelength_selection": selection_meta,
            "wavelength_selection_decision": selection_decision,
            "model_selection_decision": model_selection_decision,
            "partitions": {
                "column": split_col,
                "train": {"label": train_label, "n_samples": int(train_mask.sum())},
                "tuning": {"label": tuning_label, "n_samples": int(tune_mask.sum())},
                "external_test": {"label": test_label, "n_samples": int(test_mask.sum())},
            },
            "candidate_results": candidate_results,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y[final_train_mask], y_pred_train),
            "val": {"RMSE": float(chosen_model["RMSE_tuning"])},
            "test": compute_metrics(y_test, y_pred_external),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = float(1.0 - (chosen_model["RMSE_tuning"] ** 2) / np.var(y_tune)) if np.var(y_tune) > 0 else float("nan")
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        metrics["RMSECV"] = float(chosen_model["RMSE_tuning"])
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain="food", n_samples=int(final_train_mask.sum()))
        metrics["quality"] = quality

        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        joblib.dump(
            _build_model_artifact(
                final_model,
                method=chosen_model["method"],
                preprocessing_pipeline=final_pipe,
                preprocessing_desc=final_pipe.description(),
                wavelength_selection=selection_meta,
            ),
            real_model,
        )
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)

        report_path = os.path.join(os.path.dirname(real_model), "partitioned_report.md")
        report = [
            "# Partitioned NIR external-validation report",
            "",
            f"- Split: `{split_col}` = `{train_label}` / `{tuning_label}` / `{test_label}`",
            f"- Samples: {int(train_mask.sum())} / {int(tune_mask.sum())} / {int(test_mask.sum())}",
            f"- Pipeline: {final_pipe.description()}",
            f"- Wavelength-selection policy: {selection_decision['mode']} — {selection_decision['reason']}",
            f"- Selection: {chosen['method']} ({len(selected_indices)} of {X.shape[1]} wavelengths)",
            f"- Model-selection policy: {model_selection_decision['mode']} - {model_selection_decision['reason']}",
            f"- Selected model: {chosen_model['method']}",
            f"- Latent variables: {best_n if best_n is not None else 'not applicable'}",
            f"- Tuning RMSE: {chosen_model['RMSE_tuning']:.5f}",
            f"- External RMSE: {metrics['test']['RMSE']:.5f}",
            f"- External R2: {metrics['test']['R2']:.5f}",
            f"- External RPD: {metrics['test']['RPD']:.5f}",
        ]
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")

        out_virtual = os.path.dirname(model_output)
        candidate_summary = [
            {
                "method": item["method"],
                "n_selected": item["n_selected"],
                "n_components": item["n_components"],
                "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            }
            for item in candidate_results
        ]
        return _ok(
            {
                "status": "ok",
                "protocol": "named_partition_external_validation",
                "preprocessing": final_pipe.description(),
                "wavelength_selection": selection_meta,
                "wavelength_selection_decision": selection_decision,
                "model_selection_decision": model_selection_decision,
                "method": chosen_model["method"],
                "n_components": None if best_n is None else int(best_n),
                "RMSE_tuning": round(float(chosen_model["RMSE_tuning"]), 5),
                "external": {key: round(float(metrics["test"][key]), 5) for key in ("RMSE", "R2", "RPD", "bias")},
                "candidate_results": candidate_summary,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "report": out_virtual + "/partitioned_report.md",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


def _parse_multi_component_names(data_dict: dict, component_names: str | None, n_targets: int) -> list[str]:
    """Resolve target names from an override, NPZ metadata, or stable defaults."""
    import json

    if component_names is not None:
        parsed = json.loads(component_names) if isinstance(component_names, str) else component_names
        if not isinstance(parsed, list):
            raise ValueError("component_names must be a JSON list")
        names = [str(name).strip() for name in parsed]
    elif data_dict.get("y_names") is not None and np.asarray(data_dict["y_names"]).size:
        names = [str(name).strip() for name in np.asarray(data_dict["y_names"]).ravel()]
    else:
        names = [f"y{i}" for i in range(n_targets)]
    if len(names) != n_targets:
        raise ValueError(f"component_names length ({len(names)}) != y target count ({n_targets})")
    if any(not name for name in names):
        raise ValueError("component_names must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("component_names must be unique")
    return names


def _parse_multi_pipeline(pipeline_steps: str | None) -> tuple[list, list]:
    """Parse and validate a multi-component preprocessing specification."""
    import json

    if pipeline_steps is None:
        return [], []
    try:
        raw_steps = json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pipeline_steps JSON: {pipeline_steps!r}") from exc
    if not isinstance(raw_steps, list):
        raise ValueError("pipeline_steps must be a JSON list")
    steps = [_parse_pipeline_step(item) for item in raw_steps]
    from nir_core.preprocess.pipeline import validate_pipeline

    is_valid, reason = validate_pipeline(steps)
    if not is_valid:
        raise ValueError(f"Invalid pipeline: {reason}")
    return raw_steps, steps


def _fit_multi_pipeline(steps: list, X_train: np.ndarray, wv: np.ndarray | None):
    """Fit one configured pipeline on the training spectra only."""
    if not steps:
        return None, "none"
    from nir_core.preprocess.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(steps=steps)
    pipeline.fit(X_train, wv)
    return pipeline, pipeline.description()


def _multi_preprocessing_artifact(pipeline, description: str, *, shared: bool) -> dict:
    return {
        "shared": shared,
        "description": description,
        "pipeline": pipeline,
        "apply_on_predict": pipeline is not None,
    }


def _write_multi_report(out_dir: str, metrics: dict) -> str:
    """Write a compact comparison report for all target models."""
    lines = [
        "# Multi-component NIR calibration report",
        "",
        f"Method: `{metrics['method']}`  ",
        f"Targets: {metrics['n_targets']}  ",
        f"Shared preprocessing: {metrics['shared_preprocessing']}",
        "",
        "| Component | R2 val | RPD | RMSEP | Grade | Passed |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in metrics["per_component"]:
        lines.append(f"| {item['name']} | {item['R2_val']:.4f} | {item['RPD']:.4f} | {item['RMSEP']:.4f} | {item['quality']['grade']} | {item['quality']['passed']} |")
    overall = metrics["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"Mean validation R2: {overall['mean_R2_val']:.4f}  ",
            f"Mean RPD: {overall['mean_RPD']:.4f}  ",
            f"Passed: {overall['n_passed']}/{overall['n_total']}",
            "",
            "Each component's diagnostic plots are stored in its numbered subdirectory.",
        ]
    )
    report_path = os.path.join(out_dir, "multi_report.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


@tool("nir_train_multi_model", parse_docstring=True)
def nir_train_multi_model_tool(
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
    shared_preprocessing: bool = True,
    component_names: str | None = None,
    model_output: str = "/mnt/user-data/outputs/multi_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/multi_metrics.json",
    output_dir: str = "/mnt/user-data/outputs/multi_outputs",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train independent models for multiple reference components in one run.

    All components share one train/validation/test split. Preprocessing is
    fitted on training spectra only and can be shared or fitted separately;
    wavelength selection and model training are always independent per
    component. The saved version-3 artifact can be passed directly to
    ``nir_predict`` for N-by-K predictions.

    Args:
        input_path: Virtual path to an NPZ containing X and two-dimensional y.
        method: Modeling method accepted by nir_train_model.
        pipeline_steps: Optional JSON list of leakage-safe preprocessing steps.
        test_ratio: Fraction reserved for the independent test set.
        val_ratio: Fraction reserved for validation.
        max_components: Upper bound for PLS/PCR component search.
        cv_folds: Number of cross-validation folds.
        cv_strategy: Cross-validation strategy: auto, loocv, or fixed.
        wavelength_selection: Per-component selection: none, cars, spa, or manual.
        wavelength_selection_params: Optional JSON object for wavelength selection.
        shared_preprocessing: Fit one preprocessing pipeline for all components when true.
        component_names: Optional JSON list overriding NPZ y_names.
        model_output: Virtual output path for the version-3 model artifact.
        metrics_output: Virtual output path for multi-component metrics JSON.
        output_dir: Virtual directory for the report and component plots.
        domain: Application domain used by each component quality gate.

    Returns:
        JSON containing per-component metrics, overall pass rate, model,
        metrics and report paths, and retrieval hints for weak components.
    """
    try:
        import base64
        import json

        import joblib
        from nir_core.diagnostics import compute_residual_diagnostics
        from nir_core.model.evaluation import compute_metrics, split_dataset
        from nir_core.models import SpectralData
        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_multi_predicted_vs_reference,
            plot_multi_residuals,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra
        from nir_core.utils.metrics import evaluate_quality

        real_in = _resolve(runtime, input_path, read_only=True)
        data_dict = dict(np.load(real_in, allow_pickle=True))
        X = np.asarray(data_dict["X"], dtype=float)
        y = np.asarray(data_dict["y"], dtype=float)
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None
        if y.ndim != 2 or y.shape[1] < 2:
            return _err(f"nir_train_multi_model requires y with shape (n_samples, K>=2), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")

        names = _parse_multi_component_names(data_dict, component_names, y.shape[1])
        raw_steps, steps = _parse_multi_pipeline(pipeline_steps)
        (X_tr_raw, y_tr), (X_val_raw, y_val), (X_te_raw, y_te) = split_dataset(X, y, test_ratio=test_ratio, val_ratio=val_ratio, random_state=42)

        shared_pipe = None
        shared_desc = "none"
        if shared_preprocessing:
            shared_pipe, shared_desc = _fit_multi_pipeline(steps, X_tr_raw, wv)
            X_tr_shared = shared_pipe.transform(X_tr_raw, wv) if shared_pipe is not None else X_tr_raw
            X_val_shared = shared_pipe.transform(X_val_raw, wv) if shared_pipe is not None else X_val_raw
            X_te_shared = shared_pipe.transform(X_te_raw, wv) if shared_pipe is not None else X_te_raw

        real_model = _resolve(runtime, model_output, read_only=False)
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        real_output_dir = _resolve(runtime, output_dir, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        os.makedirs(real_output_dir, exist_ok=True)

        models: list = []
        selections: list[dict] = []
        preprocessing_artifacts: list[dict] = []
        per_component: list[dict] = []
        knowledge_hints: list[dict] = []
        test_predictions: list[np.ndarray] = []

        try:
            from nir_core.model.pls import compute_vip, get_regression_coefficients, get_regression_intercept
        except ImportError:
            compute_vip = get_regression_coefficients = get_regression_intercept = None

        raw_plot = plot_raw_spectra(SpectralData(X=X, y=y, y_names=names, wv=wv), n_highlight=5)
        with open(os.path.join(real_output_dir, "raw_spectra.png"), "wb") as file:
            file.write(base64.b64decode(raw_plot))

        for index, name in enumerate(names):
            if shared_preprocessing:
                pipeline = shared_pipe
                preprocessing_desc = shared_desc
                X_tr_base, X_val_base, X_te_base = X_tr_shared, X_val_shared, X_te_shared
                preprocessing_artifacts.append(_multi_preprocessing_artifact(shared_pipe, shared_desc, shared=True))
            else:
                pipeline, preprocessing_desc = _fit_multi_pipeline(steps, X_tr_raw, wv)
                X_tr_base = pipeline.transform(X_tr_raw, wv) if pipeline is not None else X_tr_raw
                X_val_base = pipeline.transform(X_val_raw, wv) if pipeline is not None else X_val_raw
                X_te_base = pipeline.transform(X_te_raw, wv) if pipeline is not None else X_te_raw
                preprocessing_artifacts.append(_multi_preprocessing_artifact(pipeline, preprocessing_desc, shared=False))

            X_tr, X_val, X_te, model_wv, selection = _apply_wavelength_selection(
                X_tr_base,
                X_val_base,
                X_te_base,
                y_tr[:, index],
                wv=wv,
                wavelength_selection=wavelength_selection,
                wavelength_selection_params=wavelength_selection_params,
                cv_folds=cv_folds,
            )
            model, best_n, cv_results, predict_fn = _train_one_model(
                method,
                X_tr,
                y_tr[:, index],
                max_components=max_components,
                cv_folds=cv_folds,
                cv_strategy=cv_strategy,
            )
            pred_tr = np.asarray(predict_fn(model, X_tr), dtype=float).ravel()
            pred_val = np.asarray(predict_fn(model, X_val), dtype=float).ravel()
            pred_te = np.asarray(predict_fn(model, X_te), dtype=float).ravel()

            item = {
                "name": name,
                "method": method,
                "n_components": best_n,
                "train": compute_metrics(y_tr[:, index], pred_tr),
                "val": compute_metrics(y_val[:, index], pred_val),
                "test": compute_metrics(y_te[:, index], pred_te),
                "cv_results": cv_results,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": selection,
            }
            item["R2_val"] = item["val"]["R2"]
            item["RPD"] = item["test"]["RPD"]
            item["RMSEC"] = item["train"]["RMSE"]
            item["RMSECV"] = _extract_rmsecv(cv_results, best_n)
            item["RMSEP"] = item["test"]["RMSE"]
            item["diagnostics"] = compute_residual_diagnostics(y_val[:, index], pred_val)

            if method == "pls" and compute_vip is not None:
                try:
                    vip_scores = compute_vip(model, X_tr, y_tr[:, index])
                    top_local = [int(i) for i in np.argsort(vip_scores)[-10:][::-1]]
                    selected_indices = selection.get("selected_indices") or list(range(len(vip_scores)))
                    item["vip_summary"] = {
                        "max": float(np.max(vip_scores)),
                        "mean": float(np.mean(vip_scores)),
                        "n_above_1": int(np.sum(vip_scores > 1.0)),
                        "top_10_indices": [int(selected_indices[i]) for i in top_local],
                    }
                    coefficients = get_regression_coefficients(model)
                    item["coef_summary"] = {
                        "max_abs": float(np.max(np.abs(coefficients))),
                        "mean_abs": float(np.mean(np.abs(coefficients))),
                        "intercept": get_regression_intercept(model),
                    }
                except Exception as exc:
                    warnings.warn(f"Interpretability failed for {name}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

            quality = evaluate_quality(item, domain=domain, n_samples=int(X.shape[0]))
            item["quality"] = quality
            hint = _build_knowledge_hint(
                domain=domain,
                grade=quality.get("grade"),
                passed=quality.get("passed"),
                r2_val=item["R2_val"],
                diagnostics=item["diagnostics"],
            )
            if hint is not None:
                knowledge_hints.append({"name": name, **hint})

            component_dir = os.path.join(real_output_dir, f"component_{index + 1:02d}")
            os.makedirs(component_dir, exist_ok=True)
            plot_payloads = {
                "predicted_vs_reference.png": plot_predicted_vs_reference(y_te[:, index], pred_te, title=name),
                "residuals.png": plot_residuals(y_te[:, index], pred_te),
            }
            if isinstance(cv_results, dict) and cv_results.get("n_components"):
                plot_payloads["cv_curve.png"] = plot_cv_curve(cv_results["n_components"], cv_results["mean_rmse_cv"], best_n, cv_results.get("std_rmse_cv"))
            for filename, encoded in plot_payloads.items():
                with open(os.path.join(component_dir, filename), "wb") as file:
                    file.write(base64.b64decode(encoded))
            item["plots"] = {key.removesuffix(".png"): f"{output_dir}/component_{index + 1:02d}/{key}" for key in plot_payloads}

            models.append(model)
            selections.append(selection)
            per_component.append(item)
            test_predictions.append(pred_te)

        prediction_matrix = np.column_stack(test_predictions)
        combined_plots = {
            "multi_predicted_vs_reference.png": plot_multi_predicted_vs_reference(y_te, prediction_matrix, names),
            "multi_residuals.png": plot_multi_residuals(y_te, prediction_matrix, names),
        }
        for filename, encoded in combined_plots.items():
            with open(os.path.join(real_output_dir, filename), "wb") as file:
                file.write(base64.b64decode(encoded))

        preprocessing_artifact = preprocessing_artifacts
        overall = {
            "mean_R2_val": float(np.mean([item["R2_val"] for item in per_component])),
            "mean_RPD": float(np.mean([item["RPD"] for item in per_component])),
            "n_passed": int(sum(bool(item["quality"]["passed"]) for item in per_component)),
            "n_total": len(per_component),
        }
        overall["passed"] = overall["n_passed"] == overall["n_total"]
        metrics = {
            "method": method,
            "n_targets": len(names),
            "component_names": names,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "shared_preprocessing": shared_preprocessing,
            "preprocessing": shared_desc if shared_preprocessing else [item["preprocessing"] for item in per_component],
            "preprocessing_steps": raw_steps,
            "per_component": per_component,
            "overall": overall,
            "plots": {
                "predicted_vs_reference": f"{output_dir}/multi_predicted_vs_reference.png",
                "residuals": f"{output_dir}/multi_residuals.png",
                "raw_spectra": f"{output_dir}/raw_spectra.png",
            },
        }

        artifact = {
            "format": "nir_model_artifact",
            "version": 3,
            "multi_output": True,
            "n_targets": len(names),
            "component_names": names,
            "models": models,
            "method": method,
            "preprocessing": preprocessing_artifact,
            "wavelength_selection": selections,
        }
        joblib.dump(artifact, real_model)
        with open(real_metrics, "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2, default=_json_default)
        _write_multi_report(real_output_dir, metrics)

        concise_components = [
            {
                "name": item["name"],
                "R2_val": round(item["R2_val"], 4),
                "RPD": round(item["RPD"], 4),
                "RMSEP": round(item["RMSEP"], 4),
                "grade": item["quality"]["grade"],
                "passed": item["quality"]["passed"],
            }
            for item in per_component
        ]
        return _ok(
            {
                "status": "ok",
                "method": method,
                "n_targets": len(names),
                "component_names": names,
                "shared_preprocessing": shared_preprocessing,
                "per_component": concise_components,
                "overall": overall,
                "grade": "good" if overall["passed"] else "fair",
                "passed": overall["passed"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "report": f"{output_dir}/multi_report.md",
                "plots": metrics["plots"],
                "knowledge_hint": knowledge_hints or None,
            }
        )
    except MemoryError:
        return _err("内存不足: 多成分训练数据过大。请减少成分、波长或样本数量后重试。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_analyze
# ---------------------------------------------------------------------------


@tool("nir_analyze", parse_docstring=True)
def nir_analyze_tool(
    runtime: Runtime,
    data_path: str,
    subset: str | None = None,
    auto_preprocess: bool = True,
    method: str = "auto",
    wavelength_selection: str = "auto",
    wavelength_selection_params: str | None = None,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    output_dir: str = "/mnt/user-data/outputs/nir_analysis",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """One-shot end-to-end NIR analysis pipeline (deterministic, single run).

    Runs: load → three-way split → (optional) nested-CV preprocessing
    selection → model training → full evaluation → quality assessment →
    Markdown report. This tool does NOT run the reflection loop; it returns
    a single best-effort result. For quality-driven retries, use the
    step-by-step workflow (nir_preprocess + nir_train_model) driven by the
    nir-coordinator Skill.

    Args:
        data_path: Virtual path to the raw spectral file (.mat/.csv/.txt) or
            an already-standardised .npz.
        subset: Optional sub-dataset name inside a MATLAB struct. Leave unset
            for flat files. For multiple subsets prefer
            ``nir_analyze_collection`` so the agent receives one compact result.
        auto_preprocess: If True, run nested-CV preprocessing selection over
            the default candidate pipelines and use the winner. If False,
            use raw spectra directly.
        method: ``auto`` (default) autonomously compares a bounded subset of
            PLS/Ridge/SVR/Extra Trees, or explicitly use ``pls`` / ``pcr`` /
            ``svr`` / ``rf`` / ``et`` / ``gbm`` / ``ridge`` / ``lasso`` /
            ``elasticnet`` / ``knn`` / ``mlp`` / ``cnn``.
        wavelength_selection: Train-only wavelength selection. ``"auto"``
            (default) compares full-spectrum PLS with CARS only when data and
            tuning signals justify the cost. Explicit alternatives are
            ``"none"``, ``"cars"``, ``"spa"``, or ``"manual"``.
        wavelength_selection_params: Optional JSON object with method params.
            Auto mode accepts CARS params plus ``max_selection_samples`` and
            ``min_relative_improvement``.
        max_svr_samples: Largest training set eligible for autonomous SVR.
        max_tree_samples: Largest training set eligible for autonomous Extra
            Trees comparison.
        min_model_improvement: Minimum relative validation RMSE improvement
            required before an alternative model replaces PLS.
        output_dir: Virtual directory for outputs (report, model, metrics).
        domain: Application domain for quality-gate thresholds.

    Returns:
        JSON with the full metrics, quality assessment, paths to the
        generated report and model files, and knowledge_hint.

        ★ knowledge_hint: When non-null (unknown domain or R²_val < 0.7),
        the LLM SHOULD call ``nir_search_knowledge(query=hint['query'])`` to
        retrieve relevant paper sections before reporting to the user. If the
        result is unsatisfactory, switch to the step-by-step workflow
        (nir_train_model + nir_reflect) for a reflection loop.
    """
    try:
        import joblib
        from nir_core.io.loaders import auto_detect_and_load, load_mat
        from nir_core.io.sniffers import detect_format
        from nir_core.model.evaluation import (
            compute_metrics,
            nested_cv_preprocessing,
            split_dataset,
        )
        from nir_core.utils.metrics import evaluate_quality

        real_in = _resolve(runtime, data_path, read_only=True)
        real_outdir = _resolve_writable_dir(runtime, output_dir)

        # Load (npz or raw file).
        if data_path.endswith(".npz"):
            data_dict = dict(np.load(real_in, allow_pickle=True))
            from nir_core.models import SpectralData

            data = SpectralData(
                X=np.asarray(data_dict["X"], dtype=float),
                y=(np.asarray(data_dict["y"], dtype=float).ravel() if data_dict.get("y") is not None and data_dict["y"].size else None),
                wv=(np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None),
            )
        elif subset is not None and detect_format(real_in) == "mat":
            data = load_mat(real_in, subset=subset)
        else:
            data = auto_detect_and_load(real_in)

        if data.y is None:
            return _err("No reference values (y) found; cannot train a model.")

        X, y = np.asarray(data.X, dtype=float), np.asarray(data.y, dtype=float).ravel()
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=0.20,
            val_ratio=0.10,
            random_state=42,
        )

        # Preprocessing selection (leakage-safe fit/transform).
        best_pipe = None
        if auto_preprocess:
            best_pipe, _ = nested_cv_preprocessing(
                X_tr,
                y_tr,
                X_val,
                y_val,
                inner_folds=3,
                max_components=10,
                random_state=42,
                wv=data.wv,
            )
            best_pipe = best_pipe.__class__(best_pipe.steps).fit(X_tr, data.wv)
            X_tr = best_pipe.transform(X_tr, data.wv)
            X_val = best_pipe.transform(X_val, data.wv)
            X_te = best_pipe.transform(X_te, data.wv)

        preprocessing_desc = best_pipe.description() if best_pipe else "none"
        requested_model = (method or "auto").strip().lower()
        requested_selection = (wavelength_selection or "auto").strip().lower()
        wavelength_selection_candidates: list[dict] = []
        if requested_selection == "auto" and requested_model in {"auto", "pls"}:
            (
                X_tr,
                X_val,
                X_te,
                _model_wv,
                wavelength_selection_meta,
                wavelength_selection_decision,
                wavelength_selection_candidates,
            ) = _autonomous_pls_wavelength_selection(
                X_tr,
                y_tr,
                X_val,
                y_val,
                X_te,
                wv=data.wv,
                wavelength_selection_params=wavelength_selection_params,
                max_components=10,
                cv_folds=5,
            )
        else:
            explicit_selection = "none" if requested_selection == "auto" else requested_selection
            X_tr, X_val, X_te, _model_wv, wavelength_selection_meta = _apply_wavelength_selection(
                X_tr,
                X_val,
                X_te,
                y_tr,
                wv=data.wv,
                wavelength_selection=explicit_selection,
                wavelength_selection_params=wavelength_selection_params,
                cv_folds=5,
            )
            if requested_selection == "auto":
                wavelength_selection_decision = {
                    "mode": "auto",
                    "evaluate_cars": False,
                    "reason_code": "non_pls_method_uses_full_spectrum",
                    "reason": "Autonomous CARS comparison is currently limited to PLS; the requested model uses the full spectrum.",
                    "signals": [],
                    "selected_method": "none",
                    "adoption": {"reason_code": "full_spectrum_for_non_pls_method"},
                }
            else:
                wavelength_selection_decision = {
                    "mode": "disabled" if explicit_selection == "none" else "forced",
                    "evaluate_cars": explicit_selection == "cars",
                    "reason_code": "explicit_selection_method",
                    "reason": f"The caller explicitly selected wavelength_selection={explicit_selection!r}.",
                    "signals": ["explicit_request"],
                    "selected_method": explicit_selection,
                    "adoption": {"reason_code": "explicit_selection_honored"},
                }

        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            requested_model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            max_components=10,
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )
        selected_method = chosen_model["method"]
        model = chosen_model["_model"]
        best_n = chosen_model["n_components"]
        cv_results = chosen_model["cv_results"]
        predict_fn = chosen_model["_predict_fn"]
        y_pred_te = predict_fn(model, X_te)
        y_pred_val = chosen_model["_tuning_prediction"]
        y_pred_tr = predict_fn(model, X_tr)

        metrics = {
            "method": selected_method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(wavelength_selection_meta["n_original"]),
            "n_wavelengths_model": int(wavelength_selection_meta["n_selected"]),
            "auto_preprocess": auto_preprocess,
            "preprocessing": preprocessing_desc,
            "wavelength_selection": wavelength_selection_meta,
            "wavelength_selection_decision": wavelength_selection_decision,
            "wavelength_selection_candidates": wavelength_selection_candidates,
            "model_selection_decision": model_selection_decision,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y_tr, y_pred_tr),
            "val": compute_metrics(y_val, y_pred_val),
            "test": compute_metrics(y_te, y_pred_te),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(X.shape[0]))
        metrics["quality"] = quality

        try:
            from nir_core.diagnostics import compute_residual_diagnostics

            metrics["diagnostics"] = compute_residual_diagnostics(y_val, y_pred_val)
        except Exception as exc:
            warnings.warn(f"Residual diagnostics failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

        # Save model + metrics + plots + report.
        model_path = os.path.join(real_outdir, "model.pkl")
        metrics_path = os.path.join(real_outdir, "metrics.json")
        raw_spectra_path = os.path.join(real_outdir, "raw_spectra.png")
        pred_path = os.path.join(real_outdir, "predicted_vs_reference.png")
        resid_path = os.path.join(real_outdir, "residuals.png")
        cv_path = os.path.join(real_outdir, "cv_curve.png")
        report_path = os.path.join(real_outdir, "report.md")

        joblib.dump(
            _build_model_artifact(
                model,
                method=selected_method,
                preprocessing_pipeline=best_pipe,
                preprocessing_desc=preprocessing_desc,
                wavelength_selection=wavelength_selection_meta,
            ),
            model_path,
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)

        import base64

        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra

        raw_b64 = plot_raw_spectra(data, n_highlight=5)
        pred_b64 = plot_predicted_vs_reference(y_te, y_pred_te, title="Predicted vs Reference (test set)")
        resid_b64 = plot_residuals(y_te, y_pred_te)

        # CV curve only for methods that have cv_results with n_components.
        cv_b64 = ""
        if isinstance(cv_results, dict) and cv_results.get("n_components"):
            cv_b64 = plot_cv_curve(
                cv_results["n_components"],
                cv_results["mean_rmse_cv"],
                best_n,
                cv_results.get("std_rmse_cv"),
            )

        for path, b64 in [
            (raw_spectra_path, raw_b64),
            (pred_path, pred_b64),
            (resid_path, resid_b64),
            (cv_path, cv_b64),
        ]:
            if b64:
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))

        # Markdown report (with embedded images).
        report = _build_report(
            data,
            metrics,
            quality,
            best_pipe,
            raw_spectra_b64=raw_b64,
            predicted_vs_reference_b64=pred_b64,
            residuals_b64=resid_b64,
            cv_curve_b64=cv_b64,
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # ★ Knowledge-base hint (same trigger policy as nir_train_model).
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
                "method": selected_method,
                "n_components": best_n,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": wavelength_selection_meta,
                "wavelength_selection_decision": wavelength_selection_decision,
                "wavelength_selection_candidates": _wavelength_candidate_summary(wavelength_selection_candidates),
                "model_selection_decision": model_selection_decision,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "R2_val": round(metrics["R2_val"], 4),
                "RPD": round(metrics["RPD"], 4),
                "RMSEP": round(metrics["RMSEP"], 4),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "thresholds_used": quality["thresholds_used"],
                "output_dir": output_dir,
                "report": output_dir + "/report.md",
                "model": output_dir + "/model.pkl",
                "metrics": output_dir + "/metrics.json",
                "knowledge_hint": knowledge_hint,
                "plots": {
                    "raw_spectra": output_dir + "/raw_spectra.png",
                    "predicted_vs_reference": output_dir + "/predicted_vs_reference.png",
                    "residuals": output_dir + "/residuals.png",
                    "cv_curve": output_dir + "/cv_curve.png" if cv_b64 else None,
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_analyze_collection
# ---------------------------------------------------------------------------


def _collection_subset_names(real_path: str, subsets: str) -> list[str]:
    """Resolve explicit or auto-detected MAT subset names with stable ordering."""
    requested = subsets.strip()
    if requested.lower() == "auto":
        info = json.loads(inspect_file(real_path))
        values = info.get("available_subsets")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError("MAT collection analysis requires at least two available_subsets")
    else:
        parsed = json.loads(requested)
        if not isinstance(parsed, list):
            raise ValueError('subsets must be "auto" or a JSON array of subset names')
        values = parsed

    names: list[str] = []
    for value in values:
        name = str(value).strip()
        if not name or name in names:
            continue
        names.append(name)
    if len(names) < 2:
        raise ValueError("At least two unique MAT subset names are required")
    if len(names) > 20:
        raise ValueError("At most 20 MAT subsets can be analyzed in one collection run")
    return names


def _safe_subset_dir(name: str, index: int) -> str:
    """Return a path-safe, deterministic directory name for a MAT subset."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or f"subset_{index:02d}"


def _analyze_collection_item(**kwargs) -> str:
    """Invoke the single-dataset implementation without another agent turn."""
    analyze = nir_analyze_tool.func
    if analyze is None:  # pragma: no cover - defensive LangChain compatibility
        raise RuntimeError("nir_analyze implementation is unavailable")
    return analyze(**kwargs)


def _compact_collection_result(subset: str, payload: dict) -> dict:
    """Keep only decision-grade metrics and artifact paths in model context."""
    keys = (
        "method",
        "n_components",
        "preprocessing",
        "wavelength_selection",
        "wavelength_selection_decision",
        "wavelength_selection_candidates",
        "model_selection_decision",
        "model_candidates",
        "R2_val",
        "RPD",
        "RMSEP",
        "grade",
        "passed",
        "action",
        "report",
        "model",
        "metrics",
        "plots",
    )
    return {"subset": subset, **{key: payload.get(key) for key in keys if payload.get(key) is not None}}


def _collection_summary_markdown(results: list[dict], failures: list[dict]) -> str:
    """Build a small human-readable collection summary without plot payloads."""
    lines = [
        "# NIR MAT 多子数据集汇总",
        "",
        "> 完整报告、指标、模型和图表保存在各子目录；本文件仅保留汇总指标。",
        "",
        "| 子数据集 | 方法 | 成分数 | R²(验证) | RPD | RMSEP | 等级 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in results:
        subset = str(item["subset"]).replace("|", "\\|")
        lines.append(
            f"| {subset} | {item.get('method', '-')} | {item.get('n_components', '-')} | {item.get('R2_val', '-')} | {item.get('RPD', '-')} | {item.get('RMSEP', '-')} | {item.get('grade', '-')} | {'是' if item.get('passed') else '否'} |"
        )
    if failures:
        lines.extend(["", "## 未完成的子数据集"])
        for item in failures:
            lines.append(f"- `{item['subset']}`: {item['error']}")
    lines.append("")
    return "\n".join(lines)


@tool("nir_analyze_collection", parse_docstring=True)
def nir_analyze_collection_tool(
    runtime: Runtime,
    data_path: str,
    subsets: str = "auto",
    auto_preprocess: bool = True,
    method: str = "auto",
    wavelength_selection: str = "auto",
    wavelength_selection_params: str | None = None,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    output_dir: str = "/mnt/user-data/outputs/nir_collection",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Analyze every independent sub-dataset in a MATLAB collection in one tool call.

    The computation remains sequential for predictable memory use, while all
    intermediate orchestration stays inside the tool. The agent receives only
    compact per-subset metrics and artifact paths, never full report bodies.

    Args:
        data_path: Virtual path to a MATLAB ``.mat`` file containing nested
            sub-datasets.
        subsets: ``"auto"`` to use all inspected ``available_subsets``, or a
            JSON array of explicit subset names.
        auto_preprocess: Whether each subset runs nested-CV preprocessing
            selection.
        method: Modeling mode passed to ``nir_analyze``; defaults to autonomous
            model-family selection.
        wavelength_selection: Train-only selection mode passed to
            ``nir_analyze``; defaults to autonomous full-spectrum/CARS
            evaluation for PLS.
        wavelength_selection_params: Optional JSON object for selection params.
        max_svr_samples: Per-subset autonomous SVR sample limit.
        max_tree_samples: Per-subset autonomous Extra Trees sample limit.
        min_model_improvement: Relative tuning improvement required to replace
            PLS with another model family.
        output_dir: Virtual base directory. Each subset receives its own child
            directory, plus one compact ``collection_summary.md``.
        domain: Application domain used for quality thresholds.

    Returns:
        Compact JSON with per-subset metrics, a primary result, all artifact
        paths, and the collection summary path.
    """
    try:
        real_input = _resolve(runtime, data_path, read_only=True)
        subset_names = _collection_subset_names(real_input, subsets)
        real_output_dir = _resolve_writable_dir(runtime, output_dir)
        os.makedirs(real_output_dir, exist_ok=True)

        results: list[dict] = []
        failures: list[dict] = []
        for index, subset_name in enumerate(subset_names, start=1):
            child_dir = f"{output_dir.rstrip('/')}/{_safe_subset_dir(subset_name, index)}"
            raw = _analyze_collection_item(
                runtime=runtime,
                data_path=data_path,
                subset=subset_name,
                auto_preprocess=auto_preprocess,
                method=method,
                wavelength_selection=wavelength_selection,
                wavelength_selection_params=wavelength_selection_params,
                max_svr_samples=max_svr_samples,
                max_tree_samples=max_tree_samples,
                min_model_improvement=min_model_improvement,
                output_dir=child_dir,
                domain=domain,
                tool_call_id="",
            )
            payload = json.loads(raw)
            if payload.get("status") == "ok":
                results.append(_compact_collection_result(subset_name, payload))
            else:
                failures.append({"subset": subset_name, "error": str(payload.get("error", "analysis failed"))[:500]})

        if not results:
            return _err(f"All MAT sub-dataset analyses failed: {failures}")

        primary = max(results, key=lambda item: float(item.get("R2_val", float("-inf"))))
        summary_virtual = f"{output_dir.rstrip('/')}/collection_summary.md"
        summary_real = os.path.join(real_output_dir, "collection_summary.md")
        with open(summary_real, "w", encoding="utf-8") as handle:
            handle.write(_collection_summary_markdown(results, failures))

        artifacts: list[str] = [summary_virtual]
        for item in results:
            for key in ("report", "metrics", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    artifacts.append(value)
            plots = item.get("plots")
            if isinstance(plots, dict):
                artifacts.extend(value for value in plots.values() if isinstance(value, str))

        return _ok(
            {
                "status": "ok",
                "protocol": "mat_collection_sequential_compact",
                "subset_count": len(results),
                "requested_subset_count": len(subset_names),
                "failed_subsets": failures,
                "passed": not failures and all(bool(item.get("passed")) for item in results),
                "grade": primary.get("grade"),
                "primary_subset": primary["subset"],
                "model_path": primary.get("model"),
                "metrics_path": primary.get("metrics"),
                "report": summary_virtual,
                "results": results,
                "artifacts": list(dict.fromkeys(artifacts)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_compare
# ---------------------------------------------------------------------------


@tool("nir_compare", parse_docstring=True)
def nir_compare_tool(
    runtime: Runtime,
    data_path: str,
    pipelines: str,
    method: str = "pls",
    domain: str = "default",
    output_dir: str = "/mnt/user-data/outputs/nir_comparison",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Compare multiple preprocessing pipelines and pick the best model.

    Runs each pipeline sequentially (deterministic, no LLM in the loop):
    load then apply pipeline then train model then evaluate.  After all
    pipelines are evaluated, generates an HTML comparison gallery and a
    Markdown summary, and returns the best pipeline.

    Args:
        data_path: Virtual path to the raw spectral file (.mat/.csv/.txt) or
            an already-standardised .npz.
        pipelines: JSON string: list of pipeline definitions. Each pipeline
            is a list of steps; each step can be a method name string or a
            dict with a method name and optional hyper-parameters dict.
            Use "[]" for raw spectra (no preprocessing).
        method: Modelling method: ``pls`` / ``pcr`` / ``svr`` / ``rf`` /
            ``et`` / ``gbm`` / ``ridge`` / ``lasso`` / ``elasticnet`` /
            ``knn`` / ``mlp`` / ``cnn``.
        domain: Application domain for quality-gate thresholds.
        output_dir: Virtual directory for comparison outputs.

    Returns:
        JSON with the best pipeline, its metrics, and paths to the gallery
        HTML and summary Markdown.
    """
    try:
        import json as _json

        from nir_core.io.loaders import auto_detect_and_load
        from nir_core.model.evaluation import compute_metrics, split_dataset
        from nir_core.plotting.gallery import generate_comparison_gallery
        from nir_core.utils.metrics import evaluate_quality

        try:
            from nir_core.models import ModelResult
            from nir_core.preprocess.pipeline import PreprocessingPipeline
        except ImportError:
            return _err("nir_core V3 features (PreprocessingPipeline) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core.")

        # Parse pipelines.
        try:
            pipe_list = _json.loads(pipelines) if isinstance(pipelines, str) else pipelines
        except (ValueError, TypeError):
            return _err(f"Invalid pipelines JSON: {pipelines!r}")

        if not pipe_list:
            pipe_list = [[]]  # raw spectra

        # Load data.
        real_in = _resolve(runtime, data_path, read_only=True)
        data = auto_detect_and_load(real_in)
        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel() if data.y is not None else None
        if y is None:
            return _err("Data has no reference values (y); cannot train models.")

        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=0.20,
            val_ratio=0.10,
            random_state=42,
        )

        results = []
        summaries = []
        best_idx = 0
        best_rpd = -1.0

        for i, methods in enumerate(pipe_list):
            # Build and apply pipeline.
            steps = [_parse_pipeline_step(m) for m in methods]
            pipe = PreprocessingPipeline(steps=steps) if steps else None

            try:
                if pipe is not None:
                    pipe.fit(X_tr, data.wv)
                    X_tr_p = pipe.transform(X_tr, data.wv)
                    X_val_p = pipe.transform(X_val, data.wv)
                    X_te_p = pipe.transform(X_te, data.wv)
                    pp_desc = pipe.description()
                else:
                    X_tr_p, X_val_p, X_te_p = X_tr, X_val, X_te
                    pp_desc = "无（原始光谱）"
            except Exception as pp_exc:
                summaries.append(
                    {
                        "pipeline": methods,
                        "description": " → ".join(methods) if methods else "raw",
                        "error": f"{type(pp_exc).__name__}: {pp_exc}",
                    }
                )
                continue

            # Train.
            model, best_n, cv_results, predict_fn = _train_one_model(method, X_tr_p, y_tr, max_components=20, cv_folds=10, cv_strategy="auto")
            y_pred_tr = predict_fn(model, X_tr_p)
            y_pred_val = predict_fn(model, X_val_p)
            y_pred_te = predict_fn(model, X_te_p)

            m = {
                "method": method,
                "n_components": best_n,
                "domain": domain,
                "n_samples": int(X.shape[0]),
                "train": compute_metrics(y_tr, y_pred_tr),
                "val": compute_metrics(y_val, y_pred_val),
                "test": compute_metrics(y_te, y_pred_te),
                "preprocessing": pp_desc,
                "pipeline": methods,
            }
            m["R2_val"] = m["val"]["R2"]
            m["RPD"] = m["test"]["RPD"]
            m["RMSEP"] = m["test"]["RMSE"]
            m["RMSEC"] = m["train"]["RMSE"]
            m["RMSECV"] = _extract_rmsecv(cv_results, best_n)
            m["quality"] = evaluate_quality(m, domain=domain, n_samples=int(X.shape[0]))
            try:
                from nir_core.diagnostics import compute_residual_diagnostics

                m["diagnostics"] = compute_residual_diagnostics(y_val, y_pred_val)
            except Exception as exc:
                warnings.warn(f"Residual diagnostics failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

            # Store y_ref/y_pred for gallery plot.
            m_with_arrays = dict(m)
            m_with_arrays["y_ref"] = y_te
            m_with_arrays["y_pred"] = y_pred_te
            results.append(
                ModelResult(
                    method=method,
                    n_components=best_n,
                    metrics=m_with_arrays,
                    preprocessing_steps=steps,
                )
            )

            summary_idx = len(summaries)
            summary = {
                "pipeline": methods,
                "description": pp_desc,
                "n_components": best_n,
                "R2_val": round(m["R2_val"], 4),
                "RPD": round(m["RPD"], 4),
                "RMSEP": round(m["RMSEP"], 4),
                "RMSECV": round(m["RMSECV"], 4) if m["RMSECV"] else None,
                "grade": m["quality"]["grade"],
                "passed": m["quality"]["passed"],
                "diagnostics": m.get("diagnostics"),
            }
            summaries.append(summary)

            if m["RPD"] > best_rpd:
                best_rpd = m["RPD"]
                best_idx = summary_idx

        if not summaries:
            return _err("All pipelines failed; no models to compare.")

        # Generate gallery HTML.
        real_outdir = _resolve_writable_dir(runtime, output_dir)
        html = generate_comparison_gallery(results)
        gallery_path = os.path.join(real_outdir, "comparison_gallery.html")
        with open(gallery_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Markdown summary.
        best = summaries[best_idx]
        md_lines = [
            "# 预处理流水线对比报告",
            "",
            f"共比较 {len(summaries)} 种预处理组合，建模方法: {method}",
            "",
            "## 对比结果",
            "",
            "| # | 预处理 | 成分数 | R²(验证) | RPD | RMSEP | 等级 | 通过 |",
            "|---|--------|--------|----------|-----|-------|------|------|",
        ]
        for i, s in enumerate(summaries):
            mark = " ⭐" if i == best_idx else ""
            md_lines.append(f"| {i + 1} | {s['description']}{mark} | {s.get('n_components', '-')} | {s.get('R2_val', '-')} | {s.get('RPD', '-')} | {s.get('RMSEP', '-')} | {s.get('grade', '-')} | {'是' if s.get('passed') else '否'} |")
        md_lines.extend(
            [
                "",
                "## 最佳组合",
                f"- 预处理: {best['description']}",
                f"- 成分数: {best.get('n_components')}",
                f"- R²(验证): {best.get('R2_val')}",
                f"- RPD: {best.get('RPD')}",
                f"- RMSEP: {best.get('RMSEP')}",
                f"- 等级: {best.get('grade')}",
                "",
            ]
        )
        summary_md = "\n".join(md_lines)
        md_path = os.path.join(real_outdir, "comparison_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

        # Save full metrics JSON.
        all_metrics_path = os.path.join(real_outdir, "all_metrics.json")
        with open(all_metrics_path, "w", encoding="utf-8") as f:
            _json.dump(summaries, f, ensure_ascii=False, indent=2, default=_json_default)

        knowledge_hint = _build_knowledge_hint(
            domain=domain,
            grade=best.get("grade"),
            passed=best.get("passed"),
            r2_val=best.get("R2_val"),
            diagnostics=best.get("diagnostics"),
        )

        return _ok(
            {
                "status": "ok",
                "n_compared": len(summaries),
                "best_index": best_idx,
                "best": best,
                "all_results": summaries,
                "gallery": output_dir + "/comparison_gallery.html",
                "summary": output_dir + "/comparison_summary.md",
                "all_metrics": output_dir + "/all_metrics.json",
                "knowledge_hint": knowledge_hint,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_register_model
# ---------------------------------------------------------------------------


@tool("nir_register_model", parse_docstring=True)
def nir_register_model_tool(
    runtime: Runtime,
    model_id: str,
    model_path: str,
    metrics_path: str,
    registry_path: str = "/mnt/user-data/outputs/registry.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Register a trained model into the versioned model registry.

    ★ v3: This tool is designed to be called by the coordinator agent
    *after* the reflection loop finishes, in a serial (non-parallel) manner.
    Sub-agents (nir-modeler) never call this — they only produce .pkl and
    metrics.json files. The coordinator collects all results, picks the best,
    and calls this tool once to register it.

    Args:
        model_id: Logical model identifier (e.g. ``"corn_protein_pls"``).
        model_path: Virtual path to the serialized model (.pkl).
        metrics_path: Virtual path to the metrics JSON file.
        registry_path: Virtual path to the registry JSON file.
        domain: Application domain tag for the registry record.

    Returns:
        JSON with the assigned version tag and registry summary.
    """
    from .workflow import registration_is_approved

    workflow_state = (runtime.state or {}).get("nir_workflow")
    if not registration_is_approved(workflow_state):
        return _err("Model registration requires an approved NIR workflow. Ask the user for explicit approval, then call nir_workflow(action='approve') before registering.")

    try:
        import hashlib
        import json as _json

        try:
            from nir_core.utils.registry import ModelRegistry
        except ImportError:
            return _err("nir_core V3 features (ModelRegistry) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core.")

        # Resolve paths.
        real_model = _resolve(runtime, model_path, read_only=True)
        real_metrics = _resolve(runtime, metrics_path, read_only=True)
        real_registry = _resolve(runtime, registry_path, read_only=False)
        os.makedirs(os.path.dirname(real_registry), exist_ok=True)

        # Load metrics from file.
        with open(real_metrics, encoding="utf-8") as f:
            metrics = _json.load(f)

        # Compute a data hash from the model file (best-effort fingerprint).
        with open(real_model, "rb") as f:
            model_bytes = f.read()
        data_hash = hashlib.md5(model_bytes).hexdigest()

        # Extract preprocessing steps from metrics if available.
        pp_steps = []
        if isinstance(metrics.get("preprocessing_steps"), list):
            pp_steps = metrics["preprocessing_steps"]
        elif isinstance(metrics.get("preprocessing"), str):
            pp_steps = [{"method": metrics["preprocessing"]}]
        elif isinstance(metrics.get("preprocessing"), list):
            pp_steps = metrics["preprocessing"]

        # Register.
        registry = ModelRegistry(registry_path=real_registry)
        version = registry.register(
            model_id=model_id,
            method=metrics.get("method", "unknown"),
            metrics=metrics,
            preprocessing_steps=pp_steps,
            data_hash=data_hash,
            model_path=model_path,
        )

        # Return a summary (no matrix data).
        all_versions = registry.list_versions(model_id)
        return _ok(
            {
                "status": "registered",
                "model_id": model_id,
                "version": version,
                "n_versions": len(all_versions),
                "domain": domain,
                "n_targets": int(metrics.get("n_targets", 1)),
                "component_names": metrics.get("component_names"),
                "registry_path": registry_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
