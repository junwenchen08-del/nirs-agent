"""NIR modeling tools: train / analyze / compare / register.

This module contains the four model-building tools that share a common
pattern: load data → optional leakage-safe preprocessing → train PLS/PCR/SVR
→ full evaluation → quality gate → plots + report → knowledge_hint.

Keeping them together (rather than splitting per-tool) avoids duplicating
the shared train+evaluate helper code, while still separating them from
I/O, preprocessing-only, and reflection tools.
"""

from __future__ import annotations

import os
import warnings
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _json_default, _ok, _parse_pipeline_step, _resolve, _resolve_writable_dir
from ._knowledge_hint import _build_knowledge_hint
from ._report import _build_report

_WAVELENGTH_SELECTION_METHODS = ("none", "cars", "spa", "manual")


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
# nir_analyze
# ---------------------------------------------------------------------------


@tool("nir_analyze", parse_docstring=True)
def nir_analyze_tool(
    runtime: Runtime,
    data_path: str,
    auto_preprocess: bool = True,
    method: str = "pls",
    wavelength_selection: str = "none",
    wavelength_selection_params: str | None = None,
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
        auto_preprocess: If True, run nested-CV preprocessing selection over
            the default candidate pipelines and use the winner. If False,
            use raw spectra directly.
        method: Modelling method: ``pls`` / ``pcr`` / ``svr`` / ``rf`` /
            ``et`` / ``gbm`` / ``ridge`` / ``lasso`` / ``elasticnet`` /
            ``knn`` / ``mlp`` / ``cnn``.
        wavelength_selection: Optional train-only wavelength selection:
            ``"none"`` (default), ``"cars"``, ``"spa"``, or ``"manual"``.
        wavelength_selection_params: Optional JSON object with method params.
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
        import json

        import joblib
        from nir_core.io.loaders import auto_detect_and_load
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
        X_tr, X_val, X_te, _model_wv, wavelength_selection_meta = _apply_wavelength_selection(
            X_tr,
            X_val,
            X_te,
            y_tr,
            wv=data.wv,
            wavelength_selection=wavelength_selection,
            wavelength_selection_params=wavelength_selection_params,
            cv_folds=5,
        )

        # Train (method-aware; nested_cv_preprocessing always uses PLS for
        # the inner component search, but the final model honours `method`).
        model, best_n, cv_results, predict_fn = _train_one_model(method, X_tr, y_tr, max_components=10, cv_folds=5, cv_strategy="auto")
        y_pred_te = predict_fn(model, X_te)
        y_pred_val = predict_fn(model, X_val)
        y_pred_tr = predict_fn(model, X_tr)

        metrics = {
            "method": method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(wavelength_selection_meta["n_original"]),
            "n_wavelengths_model": int(wavelength_selection_meta["n_selected"]),
            "auto_preprocess": auto_preprocess,
            "preprocessing": preprocessing_desc,
            "wavelength_selection": wavelength_selection_meta,
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
                method=method,
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
                "method": method,
                "n_components": best_n,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": wavelength_selection_meta,
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
        if isinstance(metrics.get("preprocessing"), str):
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
                "registry_path": registry_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
