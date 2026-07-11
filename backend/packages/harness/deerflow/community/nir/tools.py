"""NIR spectroscopy community tools for DeerFlow.

Each ``@tool`` function exposes a deterministic nir_core capability to the
LLM agent. Tools receive virtual sandbox paths (``/mnt/user-data/...``) and
resolve them to real container paths via the DeerFlow sandbox path-resolver,
so the LLM only ever sees virtual paths while nir_core operates on real files.

Design rules (verified against framework source):
- ``@tool("name", parse_docstring=True)`` decorator (matches tavily reference).
- ``runtime: Runtime`` injected parameter for thread-local path resolution
  (matches view_image_tool / task_tool pattern).
- Virtual paths resolved with ``resolve_and_validate_user_data_path``;
  write paths validated with ``validate_local_tool_path(read_only=False)``.
- Spectral matrices never enter the LLM context — only JSON summaries,
  metrics, and paths are returned.
- Tools are deterministic; the reflection loop lives in the nir-coordinator
  Skill decision tree, not in these tools.
"""

from __future__ import annotations

import json
import os
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime


def _parse_pipeline_step(m: str | dict):
    """Parse a pipeline step from a method-name string or a dict with params.

    Accepts both ``"snv"`` (string) and ``{"method": "sg_smooth", "params":
    {"window": 15}}`` (dict) forms, so the LLM can pass hyper-parameters
    through ``pipeline_steps``.
    """
    from nir_core.models import PreprocessingStep

    if isinstance(m, dict):
        return PreprocessingStep(**m)
    return PreprocessingStep(method=str(m))


# ---------------------------------------------------------------------------
# Path-resolution helper
# ---------------------------------------------------------------------------


def _resolve(runtime: Runtime, virtual_path: str, *, read_only: bool = True) -> str:
    """Resolve a ``/mnt/user-data/...`` virtual path to a real container path.

    Args:
        runtime: DeerFlow runtime (injected by the tool framework).
        virtual_path: Virtual path as seen by the LLM (e.g.
            ``/mnt/user-data/uploads/data.mat``).
        read_only: If True, validate for read access; if False, for write.

    Returns:
        Resolved host/container path string.

    Raises:
        RuntimeError: If thread data is unavailable or the path is invalid.
    """
    from deerflow.sandbox.exceptions import SandboxRuntimeError
    from deerflow.sandbox.tools import (
        get_thread_data,
        resolve_and_validate_user_data_path,
        validate_local_tool_path,
    )

    thread_data = get_thread_data(runtime)
    if thread_data is None:
        raise SandboxRuntimeError(f"Thread data not available; cannot resolve virtual path {virtual_path!r}.")
    validate_local_tool_path(virtual_path, thread_data, read_only=read_only)
    return resolve_and_validate_user_data_path(virtual_path, thread_data)


def _resolve_writable_dir(runtime: Runtime, virtual_dir: str) -> str:
    """Resolve a virtual directory for writing, creating it if needed.

    Used for ``output_dir``-style parameters. The directory is resolved via
    the user-data workspace root and created on disk.
    """
    real_dir = _resolve(runtime, virtual_dir, read_only=False)
    os.makedirs(real_dir, exist_ok=True)
    return real_dir


def _err(msg: str) -> str:
    """Format an error as a JSON string for the LLM."""
    return json.dumps({"status": "error", "error": msg}, ensure_ascii=False)


def _ok(payload: dict) -> str:
    """Format a success payload as a JSON string."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool("nir_load_data", parse_docstring=True)
def nir_load_data_tool(
    runtime: Runtime,
    file_path: str,
    output_path: str | None = None,
    y_col: int | None = None,
    wv_row: int | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Load a near-infrared spectral data file (.mat / .csv / .txt).

    Auto-detects the file format and internal layout (samples-in-rows vs
    samples-in-columns) and standardises it to a SpectralData container.
    Optionally saves a normalised .npz for downstream preprocessing.

    When ``y_col`` / ``wv_row`` are omitted (the common case), the loader
    auto-detects the canonical NIR "row-label + column-label" layout (empty
    corner cell, wavelength header row, reference-value column). Pass them
    explicitly ONLY when auto-detection fails — e.g. unusual CSV layouts
    without an empty corner cell, or when the first column is NOT the
    reference values. See the ``nir_inspect`` hint for guidance.

    Args:
        file_path: Virtual path to the input file, e.g.
            ``/mnt/user-data/uploads/spectra.mat``.
        output_path: Optional virtual path for a standardised .npz output,
            e.g. ``/mnt/user-data/workspace/data.npz``.
        y_col: Optional 0-based index of the reference-value column
            (CSV/TXT only; ignored for .mat). When provided, auto-detection
            is bypassed and this column is split as y. ``None`` (default)
            triggers auto-detection.
        wv_row: Optional 0-based index of the wavelength row (CSV/TXT only;
            ignored for .mat). When provided, auto-detection is bypassed and
            this row is split as wv. ``None`` (default) triggers
            auto-detection.

    Returns:
        JSON summary of the loaded data (sample count, wavelength range,
        reference value range, source format). Spectral matrices are NOT
        returned — only metadata.
    """
    try:
        from nir_core.io.loaders import auto_detect_and_load, load_csv
        from nir_core.io.sniffers import detect_format
        from nir_core.io.writers import save_npz

        real_in = _resolve(runtime, file_path, read_only=True)

        # When explicit y_col / wv_row are provided for CSV/TXT files,
        # bypass auto-detection and call load_csv with the override.
        # This gives the agent an escape hatch when auto-detection fails,
        # so it never needs to fall back to writing Python scripts.
        has_override = y_col is not None or wv_row is not None
        if has_override and detect_format(real_in) != "mat":
            data = load_csv(real_in, y_col=y_col, wv_row=wv_row)
        else:
            data = auto_detect_and_load(real_in)

        if output_path:
            real_out = _resolve(runtime, output_path, read_only=False)
            os.makedirs(os.path.dirname(real_out), exist_ok=True)
            save_npz(data, real_out)
        summary = data.summary()
        summary["output_saved"] = bool(output_path)
        summary["layout_override_used"] = has_override
        # ⭐ Surface the auto-detected layout so the agent can confirm y and
        # wv have been correctly separated from the raw CSV block. Without
        # these explicit fields the agent tends to "double-check" by writing
        # its own Python script — which is exactly the anti-pattern this
        # tool exists to prevent.
        summary["y_separated"] = data.y is not None
        summary["wv_separated"] = data.wv is not None
        if data.y is not None:
            summary["y_first_values"] = [round(float(v), 4) for v in data.y[:5]]
            summary["y_last_values"] = [round(float(v), 4) for v in data.y[-3:]]
        if data.wv is not None:
            summary["wv_first_values"] = [round(float(v), 2) for v in data.wv[:3]]
            summary["wv_last_values"] = [round(float(v), 2) for v in data.wv[-3:]]
        if not summary["y_separated"]:
            summary["warning"] = (
                "No reference values (y) detected. The file may be spectra-only, "
                "or its layout does not match the auto-detected NIR pattern. "
                "Use nir_inspect to confirm the structure, then re-call "
                "nir_load_data with explicit y_col / wv_row if needed."
            )
        return _ok(summary)
    except MemoryError:
        return _err("内存不足: 文件过大，无法加载。请先用 nir_inspect 查看文件结构，或使用更小的数据子集。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_inspect", parse_docstring=True)
def nir_inspect_tool(
    runtime: Runtime,
    file_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Inspect a spectral file's structure without fully loading it.

    Returns format, shape, estimated sample/wavelength counts, value range
    and NaN presence as a JSON report. Useful for previewing data before
    committing to a full load.

    Args:
        file_path: Virtual path to the file to inspect.
    """
    try:
        from nir_core.io.sniffers import inspect_file

        real_in = _resolve(runtime, file_path, read_only=True)
        raw = inspect_file(real_in)
        # inspect_file returns a JSON string. Parse it so we can layer extra
        # advisory fields and re-serialize.
        import json as _json

        try:
            info = _json.loads(raw)
        except (TypeError, ValueError):
            return raw

        # Add layout advisory so the agent knows whether the NIR-style
        # "row-label + column-label" pattern is present without having to
        # open the file in Python.
        corner_nan = bool(info.get("corner_is_nan"))
        info["layout_pattern"] = "row_label_and_column_label" if corner_nan else "plain_matrix"
        info["hint"] = (
            "Empty/NaN corner cell detected — file uses the canonical NIR "
            "row-label + column-label layout. nir_load_data will auto-split "
            "y (first column) and wv (first row) from X. The 'structure' "
            "field above is just the raw shape heuristic and may misclassify "
            "this layout as 'samples_in_columns' because columns > rows; "
            "that is NOT a bug — the corner NaN signal is the authoritative cue."
            if corner_nan
            else "No empty corner cell detected. The file may use a plain matrix layout (no separate y column or wavelength header). Pass y_col / wv_row to nir_load_data explicitly if needed."
        )
        return _json.dumps(info, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_preprocess", parse_docstring=True)
def nir_preprocess_tool(
    runtime: Runtime,
    input_path: str,
    output_path: str,
    method: str | None = None,
    pipeline_steps: str | None = None,
    window: int = 11,
    order: int = 2,
    lambda_: float | None = None,
    p: float = 0.001,
    norm: str = "l2",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Apply preprocessing to a spectral .npz file (single method or multi-step pipeline).

    Two modes:
    - **Single method** (backward-compatible): pass ``method="snv"`` etc.
    - **Multi-step pipeline** (★ v3): pass ``pipeline_steps`` as a JSON array
      of method names, e.g. ``'["snv","sg_smooth","mean_center"]'``. All
      steps execute atomically in one call, reducing LLM round-trips.

    The input .npz must contain at least an ``X`` array. The result is saved
    to ``output_path``.

    Args:
        input_path: Virtual path to the input .npz file.
        output_path: Virtual path for the output .npz file.
        method: Single preprocessing method (backward-compatible). One of:
            snv, msc, sg_smooth, derivative1, derivative2, airpls, asls,
            detrend, mean_center, autoscale, normalize.
        pipeline_steps: JSON array of method names for multi-step atomic
            execution. Takes precedence over ``method`` if both given.
            For example ``'["snv","sg_smooth","mean_center"]'``.
        window: Savitzky-Golay window (odd, > order).
        order: Savitzky-Golay polynomial order.
        lambda_: Smoothness penalty for airpls / asls. If None, each method
            uses its own default (airpls=1e7, asls=1e5).
        p: Asymmetry parameter for asls.
        norm: Normalisation norm for ``normalize``: l1 / l2 / max.

    Returns:
        JSON status with the method(s) applied, input/output shapes, and
        output path.
    """
    try:
        import json as _json

        try:
            from nir_core.preprocess.pipeline import PRESTEP_METHODS, PreprocessingPipeline
            from nir_core.models import PreprocessingStep
        except ImportError:
            return _err(
                "nir_core V3 features (PreprocessingPipeline / PreprocessingStep) are not available in the current sandbox. Please rebuild the Docker image so the editable install of ../nir_core picks up the latest sources, then re-run."
            )

        # Resolve which mode: multi-step pipeline or single method.
        steps_list: list[str] = []
        if pipeline_steps is not None:
            try:
                steps_list = _json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
            except (ValueError, TypeError):
                return _err(f"Invalid pipeline_steps JSON: {pipeline_steps!r}")
            if not isinstance(steps_list, list) or not steps_list:
                return _err("pipeline_steps must be a non-empty JSON array of method names")
        elif method is not None:
            steps_list = [method]
        else:
            return _err("Either 'method' or 'pipeline_steps' must be provided")

        # Validate all methods exist.
        for m in steps_list:
            if m not in PRESTEP_METHODS:
                return _err(f"Unknown method {m!r}. Available: {sorted(PRESTEP_METHODS.keys())}")

        # Validate Savitzky-Golay parameters.
        sg_methods = {"sg_smooth", "derivative1", "derivative2"}
        if sg_methods & set(steps_list):
            if window % 2 == 0:
                return _err(f"window must be odd, got {window}")
            if window <= order:
                return _err(f"window ({window}) must be > order ({order})")

        real_in = _resolve(runtime, input_path, read_only=True)
        real_out = _resolve(runtime, output_path, read_only=False)
        os.makedirs(os.path.dirname(real_out), exist_ok=True)

        data_dict = dict(np.load(real_in, allow_pickle=True))
        X = np.asarray(data_dict["X"], dtype=float)

        # Build and apply pipeline atomically.
        steps = []
        for m in steps_list:
            params: dict = {}
            if m in sg_methods:
                params["window"] = window
                params["order"] = order
            elif m == "airpls":
                if lambda_ is not None:
                    params["lambda_"] = lambda_
            elif m == "asls":
                if lambda_ is not None:
                    params["lambda_"] = lambda_
                params["p"] = p
            elif m == "normalize":
                params["norm"] = norm
            steps.append(PreprocessingStep(method=m, params=params))

        pipe = PreprocessingPipeline(steps=steps)
        X_processed = pipe.apply(X, wv=(np.asarray(data_dict["wv"]).ravel() if data_dict.get("wv") is not None else None))
        data_dict["X"] = X_processed
        # Sync wavelength vector if preprocessing changed the wavelength count.
        if "wv" in data_dict and data_dict["wv"] is not None:
            wv_arr = np.asarray(data_dict["wv"]).ravel()
            if wv_arr.shape[0] > X_processed.shape[1]:
                data_dict["wv"] = wv_arr[: X_processed.shape[1]]
        np.savez(real_out, **data_dict)
        return _ok(
            {
                "status": "ok",
                "pipeline": steps_list,
                "description": pipe.description(),
                "input_shape": list(X.shape),
                "output_shape": list(X_processed.shape),
                "output_path": output_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


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
    model_output: str = "/mnt/user-data/outputs/model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/metrics.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train a chemometric calibration model (PLS / PCR / SVR).

    ★ v3 改进: 接受可选的 ``pipeline_steps`` JSON 参数实现防泄露预处理。
    当提供 ``pipeline_steps`` 时，工具内部执行：划分数据 → 仅在训练集
    拟合预处理参数 → 变换全部集合 → 建模。避免了分步模式中先全局预处理
    后划分导致的数据泄露问题。

    自动执行: 三集分离（可选内置防泄露预处理）→ 内部CV选成分数 →
    全指标评估（RMSEC/RMSECV/RMSEP, R², RPD, bias, slope）→ VIP
    可解释性输出 → 模型序列化 → 指标JSON。

    Args:
        input_path: Virtual path to the .npz file (must contain ``X`` and
            ``y``). Can be raw data or pre-preprocessed data.
        method: Modelling method: ``pls`` / ``pcr`` / ``svr``.
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
        model_output: Virtual path for the serialised model (.pkl).
        metrics_output: Virtual path for the metrics JSON file.
        domain: Application domain for quality-gate thresholds.

    Returns:
        JSON with method, n_components, R2_val, RPD, RMSEC, RMSECV, RMSEP,
        VIP summary, model_path, metrics_path, and quality assessment.
    """
    try:
        import json as _json
        import joblib

        from nir_core.io.loaders import auto_detect_and_load
        from nir_core.model.evaluation import (
            compute_metrics,
            split_dataset,
        )
        from nir_core.model.pls import predict_pls, train_pls
        from nir_core.utils.metrics import evaluate_quality

        # Defensive: newer nir_core exposes compute_vip / get_regression_coefficients;
        # older sandboxes may not. Import lazily and tolerate ImportError so the
        # tool still runs end-to-end on the older wheel.
        try:
            from nir_core.model.pls import compute_vip, get_regression_coefficients
        except ImportError:
            compute_vip = None
            get_regression_coefficients = None

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
        if pipeline_steps is not None:
            try:
                from nir_core.models import PreprocessingStep
                from nir_core.preprocess.pipeline import PreprocessingPipeline
            except ImportError:
                return _err("nir_core V3 features (PreprocessingPipeline) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core, or remove pipeline_steps and train without inline preprocessing.")

            try:
                steps_list = _json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
            except (ValueError, TypeError):
                return _err(f"Invalid pipeline_steps JSON: {pipeline_steps!r}")

            steps = [_parse_pipeline_step(m) for m in steps_list]

            # ★ V3.6: Validate pipeline before training (flexible guardrail).
            from nir_core.preprocess.pipeline import validate_pipeline

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

        if method == "pls":
            # Defensive: older nir_core versions do not accept cv_strategy.
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
            y_pred_tr = predict_pls(model, X_tr)
            y_pred_val = predict_pls(model, X_val)
            y_pred_te = predict_pls(model, X_te)
        elif method == "pcr":
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
            y_pred_tr = predict_pcr(model, X_tr)
            y_pred_val = predict_pcr(model, X_val)
            y_pred_te = predict_pcr(model, X_te)
        elif method == "svr":
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
            best_n = None
            y_pred_tr = predict_svr(model, X_tr)
            y_pred_val = predict_svr(model, X_val)
            y_pred_te = predict_svr(model, X_te)
        else:
            return _err(f"Unknown method {method!r}; use pls/pcr/svr")

        metrics = {
            "method": method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "preprocessing": preprocessing_desc,
            "preprocessing_steps": steps_list if pipeline_steps is not None else [],
            "train": compute_metrics(y_tr, y_pred_tr),
            "val": compute_metrics(y_val, y_pred_val),
            "test": compute_metrics(y_te, y_pred_te),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        _rmse_cv = cv_results.get("mean_rmse_cv") if isinstance(cv_results, dict) else None
        if isinstance(_rmse_cv, list) and _rmse_cv:
            _nc_list = cv_results.get("n_components", [])
            if best_n in _nc_list:
                metrics["RMSECV"] = float(_rmse_cv[_nc_list.index(best_n)])
            else:
                metrics["RMSECV"] = float(min(_rmse_cv))
        elif isinstance(_rmse_cv, (int, float)):
            metrics["RMSECV"] = float(_rmse_cv)
        else:
            metrics["RMSECV"] = None
        metrics["RMSEP"] = metrics["test"]["RMSE"]

        # ★ v3: Interpretability — VIP and regression coefficients (PLS only).
        vip_summary = None
        coef_summary = None
        if method == "pls":
            try:
                vip_scores = compute_vip(model, X_tr, y_tr)
                vip_summary = {
                    "max": float(np.max(vip_scores)),
                    "mean": float(np.mean(vip_scores)),
                    "n_above_1": int(np.sum(vip_scores > 1.0)),
                    "top_10_indices": [int(i) for i in np.argsort(vip_scores)[-10:][::-1]],
                }
                coef = get_regression_coefficients(model)
                coef_summary = {
                    "max_abs": float(np.max(np.abs(coef))),
                    "mean_abs": float(np.mean(np.abs(coef))),
                }
                metrics["vip_summary"] = vip_summary
                metrics["coef_summary"] = coef_summary
            except Exception:
                pass  # VIP computation failure is non-fatal.

        # Quality assessment.
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(X.shape[0]))
        metrics["quality"] = quality

        # ★ V3.6: Residual diagnostics for LLM-driven preprocessing decisions.
        try:
            from nir_core.diagnostics import compute_residual_diagnostics

            val_diag = compute_residual_diagnostics(y_val, y_pred_val)
            metrics["diagnostics"] = val_diag
        except Exception:
            pass  # Diagnostics failure is non-fatal.

        # Serialise model + metrics.
        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        joblib.dump(model, real_model)

        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)

        # Generate plots + report.
        import base64

        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra

        from nir_core.models import SpectralData

        spec_data = SpectralData(X=X, y=y, wv=wv)

        out_dir = os.path.dirname(real_model)
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
                from nir_core.plotting.model_diag import plot_vip, plot_regression_coefficients

                vip_scores = compute_vip(model, X_tr, y_tr)
                vip_b64 = plot_vip(vip_scores, wv=wv)
                coef = get_regression_coefficients(model)
                coef_b64 = plot_regression_coefficients(coef, wv=wv)
                for fname, b64 in [("vip_scores.png", vip_b64), ("regression_coefficients.png", coef_b64)]:
                    if b64:
                        with open(os.path.join(out_dir, fname), "wb") as f:
                            f.write(base64.b64decode(b64))
            except Exception:
                pass  # VIP plot failure is non-fatal.

        # Markdown report with embedded plots.
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

        out_virtual = os.path.dirname(model_output)

        return _ok(
            {
                "status": "ok",
                "method": method,
                "n_components": best_n,
                "preprocessing": preprocessing_desc,
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


@tool("nir_predict", parse_docstring=True)
def nir_predict_tool(
    runtime: Runtime,
    model_path: str,
    data_path: str,
    output_path: str | None = None,
    detect_drift: bool = True,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Predict reference values for new spectra using a trained model.

    Loads a joblib-serialised model and applies it to a preprocessed .npz.
    Optionally runs Mahalanobis drift detection against the training
    distribution (if the metrics JSON contains training stats).

    Args:
        model_path: Virtual path to the .pkl model file.
        data_path: Virtual path to the preprocessed new-data .npz.
        output_path: Optional virtual path for a CSV of predictions.
        detect_drift: If True, compute Mahalanobis drift of the new spectra
            against the input data's own distribution (as a proxy when no
            training reference is available).

    Returns:
        JSON with predictions summary (count, mean, min, max) and optional
        drift info. Individual predictions are saved to output_path if given.
    """
    try:
        import joblib

        real_model = _resolve(runtime, model_path, read_only=True)
        real_data = _resolve(runtime, data_path, read_only=True)
        model = joblib.load(real_model)
        data_dict = dict(np.load(real_data, allow_pickle=True))
        X = np.asarray(data_dict["X"], dtype=float)

        # Predict (sklearn-style model).
        y_pred = np.asarray(model.predict(X)).ravel()

        result = {
            "status": "ok",
            "n_samples": int(X.shape[0]),
            "prediction_mean": float(np.mean(y_pred)),
            "prediction_min": float(np.min(y_pred)),
            "prediction_max": float(np.max(y_pred)),
        }

        if detect_drift:
            from nir_core.utils.drift import compute_mahalanobis_drift

            # Use the input data itself as the reference distribution
            # (best available proxy when no separate training matrix exists).
            drift = compute_mahalanobis_drift(X, X, threshold=3.0)
            result["drift"] = {
                "note": "Self-reference drift (no training matrix supplied); supply training X for a true drift estimate.",
                "drift_score": float(drift["drift_score"]),
            }

        if output_path:
            real_out = _resolve(runtime, output_path, read_only=False)
            os.makedirs(os.path.dirname(real_out), exist_ok=True)
            import csv

            with open(real_out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sample_index", "prediction"])
                for i, v in enumerate(y_pred):
                    w.writerow([i, float(v)])
            result["output_path"] = output_path

        return _ok(result)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_analyze", parse_docstring=True)
def nir_analyze_tool(
    runtime: Runtime,
    data_path: str,
    auto_preprocess: bool = True,
    method: str = "pls",
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
        method: Modelling method: ``pls`` / ``pcr`` / ``svr``.
        output_dir: Virtual directory for outputs (report, model, metrics).
        domain: Application domain for quality-gate thresholds.

    Returns:
        JSON with the full metrics, quality assessment, and paths to the
        generated report and model files.
    """
    try:
        import joblib

        from nir_core.io.loaders import auto_detect_and_load
        from nir_core.io.writers import save_npz
        from nir_core.model.evaluation import (
            compute_metrics,
            nested_cv_preprocessing,
            split_dataset,
        )
        from nir_core.model.pls import predict_pls, train_pls
        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra
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
            p = best_pipe.__class__(best_pipe.steps).fit(X_tr, data.wv)
            X_tr = p.transform(X_tr, data.wv)
            X_val = p.transform(X_val, data.wv)
            X_te = p.transform(X_te, data.wv)

        # Train (method-aware; nested_cv_preprocessing always uses PLS for
        # the inner component search, but the final model honours `method`).
        if method == "pls":
            try:
                model, best_n, cv_results = train_pls(
                    X_tr,
                    y_tr,
                    n_components=None,
                    max_components=10,
                    cv_folds=5,
                    cv_strategy="auto",
                    random_state=42,
                )
            except TypeError:
                model, best_n, cv_results = train_pls(
                    X_tr,
                    y_tr,
                    n_components=None,
                    max_components=10,
                    cv_folds=5,
                    random_state=42,
                )
            y_pred_te = predict_pls(model, X_te)
            y_pred_val = predict_pls(model, X_val)
            y_pred_tr = predict_pls(model, X_tr)
        elif method == "pcr":
            from nir_core.model.pcr import predict_pcr, train_pcr

            try:
                model, best_n, cv_results = train_pcr(
                    X_tr,
                    y_tr,
                    n_components=None,
                    max_components=10,
                    cv_folds=5,
                    cv_strategy="auto",
                    random_state=42,
                )
            except TypeError:
                model, best_n, cv_results = train_pcr(
                    X_tr,
                    y_tr,
                    n_components=None,
                    max_components=10,
                    cv_folds=5,
                    random_state=42,
                )
            y_pred_te = predict_pcr(model, X_te)
            y_pred_val = predict_pcr(model, X_val)
            y_pred_tr = predict_pcr(model, X_tr)
        elif method == "svr":
            from nir_core.model.svr import predict_svr, train_svr

            try:
                model, cv_results = train_svr(
                    X_tr,
                    y_tr,
                    cv_folds=5,
                    cv_strategy="auto",
                    random_state=42,
                )
            except TypeError:
                model, cv_results = train_svr(
                    X_tr,
                    y_tr,
                    cv_folds=5,
                    random_state=42,
                )
            best_n = None
            y_pred_te = predict_svr(model, X_te)
            y_pred_val = predict_svr(model, X_val)
            y_pred_tr = predict_svr(model, X_tr)
        else:
            return _err(f"Unknown method {method!r}; use pls/pcr/svr")

        metrics = {
            "method": method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "auto_preprocess": auto_preprocess,
            "preprocessing": (best_pipe.description() if best_pipe else "none"),
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

        # Save model + metrics + plots + report.
        model_path = os.path.join(real_outdir, "model.pkl")
        metrics_path = os.path.join(real_outdir, "metrics.json")
        raw_spectra_path = os.path.join(real_outdir, "raw_spectra.png")
        pred_path = os.path.join(real_outdir, "predicted_vs_reference.png")
        resid_path = os.path.join(real_outdir, "residuals.png")
        cv_path = os.path.join(real_outdir, "cv_curve.png")
        report_path = os.path.join(real_outdir, "report.md")

        joblib.dump(model, model_path)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)

        import base64

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

        return _ok(
            {
                "status": "ok",
                "method": method,
                "n_components": best_n,
                "preprocessing": (best_pipe.description() if best_pipe else "none"),
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
# Report builder
# ---------------------------------------------------------------------------


def _build_report(
    data,
    metrics: dict,
    quality: dict,
    best_pipe,
    raw_spectra_b64: str = "",
    predicted_vs_reference_b64: str = "",
    residuals_b64: str = "",
    cv_curve_b64: str = "",
) -> str:
    """Build a Chinese Markdown analysis report with embedded plots."""
    pp = best_pipe.description() if best_pipe else "无（使用原始光谱）"
    lines = [
        "# 近红外光谱分析报告",
        "",
        "> 本报告由 `nir_analyze` 工具自动生成。所有图表已嵌入下方，无需额外调用 matplotlib 绘制。",
        "",
        "## 1. 数据概览",
        f"- 样本数: {metrics['n_samples']}",
        f"- 波长数: {data.X.shape[1]}",
        (f"- 波长范围: {float(data.wv.min()):.1f} - {float(data.wv.max()):.1f} nm" if data.wv is not None else "- 波长范围: 未提供"),
        (f"- 参考值范围: {float(data.y.min()):.3f} - {float(data.y.max()):.3f}" if data.y is not None else ""),
        "",
        "## 2. 预处理方法",
        f"- 选定流水线: {pp}",
        "",
        "## 3. 模型结果",
        f"- 方法: {metrics['method']}",
        f"- 成分数: {metrics['n_components']}",
        f"- R²(验证集): {metrics['R2_val']:.4f}",
        f"- RPD(测试集): {metrics['RPD']:.4f}",
        f"- RMSEP(测试集): {metrics['RMSEP']:.4f}",
        "",
        "## 4. 质量评估",
        f"- 等级: {quality['grade']}",
        f"- 是否通过: {'是' if quality['passed'] else '否'}",
        f"- 行动建议: {quality['action']}",
        f"- 使用阈值: R²≥{quality['thresholds_used']['min_r2']}, RPD≥{quality['thresholds_used']['min_rpd']}",
        f"- 领域: {quality['thresholds_used']['domain']}",
        "",
        "## 5. 可视化",
    ]

    def _embed(title: str, b64: str) -> list[str]:
        if not b64:
            return []
        return [
            f"### {title}",
            "",
            f"![{title}](data:image/png;base64,{b64})",
            "",
        ]

    lines.extend(_embed("原始光谱", raw_spectra_b64))
    lines.extend(_embed("预测 vs 参考值（测试集）", predicted_vs_reference_b64))
    lines.extend(_embed("残差诊断", residuals_b64))
    lines.extend(_embed("交叉验证选成分", cv_curve_b64))

    lines.extend(
        [
            "## 6. 部署建议",
        ]
    )
    if quality["passed"]:
        lines.append("- 模型质量达标，可用于预测。建议定期做漂移检测。")
    elif quality["action"] == "retry_preprocessing":
        lines.append("- 模型质量未达标，建议尝试不同预处理组合（由协调器反思闭环驱动）。")
    elif quality["action"] == "investigate_data":
        lines.append("- 模型质量较差，建议检查数据质量、异常样本和样本代表性。")
    else:
        lines.append("- 模型质量一般，谨慎使用并持续优化。")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: Deterministic reflection & parallel comparison tools
# ---------------------------------------------------------------------------


@tool("nir_reflect", parse_docstring=True)
def nir_reflect_tool(
    runtime: Runtime,
    metrics_path: str,
    history: str = "[]",
    domain: str = "default",
    attempt: int = 1,
    max_retries: int = 3,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Deterministic reflection decision for the NIR preprocessing retry loop.

    ★ V3.6: Returns diagnostics (residual trend/variance/outliers) from the
    metrics file so the LLM can reason about *what* to try next.  The
    deterministic ``fallback_suggestion`` is provided as a safety net — the
    LLM should prefer constructing its own ``pipeline_steps`` based on the
    diagnostics, falling back to the suggestion only when stuck.

    Wraps should_retry + get_next_pipeline + suggest_lv_adjustment into a
    single deterministic call.  Call this after every nir_train_model to
    decide what to do next.

    Args:
        metrics_path: Virtual path to the metrics.json produced by
            nir_train_model or nir_analyze.
        history: JSON string of prior attempt records. Each record should
            have a pipeline field (list of method-name dicts) and a metrics
            field (a metrics dict with R2_val / RPD). Pass "[]" for the
            first attempt.
        domain: Application domain for quality-gate thresholds.
        attempt: Current 1-based attempt number.
        max_retries: Maximum allowed retries (default 3).

    Returns:
        JSON with should_retry, reason, diagnostics, fallback_suggestion,
        fallback_suggestion_steps, lv_adjustment, current_quality,
        best_so_far, attempt, and next_attempt.
    """
    try:
        import json as _json

        from nir_core.utils.validation import (
            get_next_pipeline,
            should_retry,
            suggest_lv_adjustment,
        )

        # Load current metrics.
        real_metrics = _resolve(runtime, metrics_path, read_only=True)
        with open(real_metrics, "r", encoding="utf-8") as f:
            metrics = _json.load(f)

        # Parse history.
        try:
            history_list = _json.loads(history) if isinstance(history, str) else history
        except (ValueError, TypeError):
            history_list = []

        n_samples = metrics.get("n_samples")

        # Quality + LV adjustment.
        lv_adj = suggest_lv_adjustment(
            metrics,
            domain=domain,
            n_samples=n_samples,
        )
        quality = lv_adj["quality"]

        # Should retry?
        retry = should_retry(
            quality,
            attempt=attempt,
            max_retries=max_retries,
            history=history_list,
        )

        # Next pipeline.
        next_pipeline = None
        next_steps = None
        if retry:
            steps = get_next_pipeline(attempt=attempt, history=history_list)
            if steps is not None:
                next_pipeline = [s.method for s in steps]
                next_steps = [{"method": s.method, "params": s.params} for s in steps]
            else:
                # Candidates exhausted — override retry to False.
                retry = False

        # Find best-so-far from history + current.
        all_records = list(history_list) + [
            {
                "pipeline": metrics.get("preprocessing_steps", []),
                "metrics": metrics,
            }
        ]
        best = None
        for rec in all_records:
            m = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            r2 = m.get("R2_val") or m.get("R2")
            rpd_v = m.get("RPD")
            if rpd_v is None:
                rpd_v = 0.0
            if best is None or (rpd_v or 0) > (best.get("RPD") or 0):
                best = {
                    "R2_val": r2,
                    "RPD": rpd_v,
                    "grade": m.get("quality", {}).get("grade", "unknown"),
                    "pipeline": rec.get("pipeline", []),
                }

        reason_parts = []
        if not retry:
            if quality["passed"]:
                reason_parts.append("质量门禁已通过，无需重试。")
            elif attempt >= max_retries:
                reason_parts.append(f"已达最大重试次数({max_retries})，停止重试。")
            elif get_next_pipeline(attempt=attempt, history=history_list) is None:
                reason_parts.append("所有候选预处理流水线已全部尝试完毕，停止重试。")
            else:
                reason_parts.append("连续两次无显著改善，停止重试。")
        else:
            reason_parts.append(f"当前等级={quality['grade']}，未通过门禁。")
            if lv_adj["issues"]:
                reason_parts.append(f"检测到问题: {', '.join(lv_adj['issues'])}。")
            diagnostics = metrics.get("diagnostics", {})
            if diagnostics:
                reason_parts.append(f"残差诊断: trend={diagnostics.get('residual_trend', 'unknown')}, variance={diagnostics.get('residual_variance', 'unknown')}, outliers={diagnostics.get('outlier_count', 0)}。")
                reason_parts.append("请根据诊断信息自主构造下一步 pipeline_steps（含超参数），经 validate_pipeline 校验后调用 nir_train_model。")
            reason_parts.append(lv_adj["recommendation"])

        return _ok(
            {
                "should_retry": retry,
                "reason": " ".join(reason_parts),
                "diagnostics": metrics.get("diagnostics", {}),
                "fallback_suggestion": next_pipeline,
                "fallback_suggestion_steps": next_steps,
                "lv_adjustment": {
                    "current_n": lv_adj["current_n"],
                    "suggested_n": lv_adj["suggested_n"],
                    "change": lv_adj["change"],
                    "delta": lv_adj["delta"],
                    "issues": lv_adj["issues"],
                    "recommendation": lv_adj["recommendation"],
                },
                "current_quality": {
                    "grade": quality["grade"],
                    "passed": quality["passed"],
                    "action": quality["action"],
                    "thresholds_used": quality["thresholds_used"],
                },
                "best_so_far": best,
                "attempt": attempt,
                "next_attempt": (attempt + 1) if retry else None,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


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
        method: Modelling method: pls / pcr / svr.
        domain: Application domain for quality-gate thresholds.
        output_dir: Virtual directory for comparison outputs.

    Returns:
        JSON with the best pipeline, its metrics, and paths to the gallery
        HTML and summary Markdown.
    """
    try:
        import json as _json

        from nir_core.io.loaders import auto_detect_and_load
        from nir_core.model.evaluation import (
            compute_metrics,
            split_dataset,
        )
        from nir_core.model.pls import predict_pls, train_pls
        from nir_core.plotting.gallery import generate_comparison_gallery
        from nir_core.utils.metrics import evaluate_quality

        try:
            from nir_core.models import ModelResult, PreprocessingStep
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
                    pipe.fit(X_tr)
                    X_tr_p = pipe.transform(X_tr)
                    X_val_p = pipe.transform(X_val)
                    X_te_p = pipe.transform(X_te)
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
            if method == "pcr":
                from nir_core.model.pcr import predict_pcr, train_pcr

                model, best_n, cv_results = train_pcr(
                    X_tr_p,
                    y_tr,
                    n_components=None,
                    max_components=20,
                    cv_folds=10,
                    random_state=42,
                )
                y_pred_tr = predict_pcr(model, X_tr_p)
                y_pred_val = predict_pcr(model, X_val_p)
                y_pred_te = predict_pcr(model, X_te_p)
            elif method == "svr":
                from nir_core.model.svr import predict_svr, train_svr

                model, cv_results = train_svr(
                    X_tr_p,
                    y_tr,
                    cv_folds=10,
                    random_state=42,
                )
                best_n = None
                y_pred_tr = predict_svr(model, X_tr_p)
                y_pred_val = predict_svr(model, X_val_p)
                y_pred_te = predict_svr(model, X_te_p)
            else:
                model, best_n, cv_results = train_pls(
                    X_tr_p,
                    y_tr,
                    n_components=None,
                    max_components=20,
                    cv_folds=10,
                    random_state=42,
                )
                y_pred_te = predict_pls(model, X_te_p)
                y_pred_val = predict_pls(model, X_val_p)
                y_pred_tr = predict_pls(model, X_tr_p)

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
            _rmse_cv = cv_results.get("mean_rmse_cv") if isinstance(cv_results, dict) else None
            if isinstance(_rmse_cv, list) and _rmse_cv:
                _nc = cv_results.get("n_components", [])
                if best_n in _nc:
                    m["RMSECV"] = float(_rmse_cv[_nc.index(best_n)])
                else:
                    m["RMSECV"] = float(min(_rmse_cv))
            else:
                m["RMSECV"] = None
            m["quality"] = evaluate_quality(m, domain=domain, n_samples=int(X.shape[0]))

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

            summaries.append(
                {
                    "pipeline": methods,
                    "description": pp_desc,
                    "n_components": best_n,
                    "R2_val": round(m["R2_val"], 4),
                    "RPD": round(m["RPD"], 4),
                    "RMSEP": round(m["RMSEP"], 4),
                    "RMSECV": round(m["RMSECV"], 4) if m["RMSECV"] else None,
                    "grade": m["quality"]["grade"],
                    "passed": m["quality"]["passed"],
                }
            )

            if m["RPD"] > best_rpd:
                best_rpd = m["RPD"]
                best_idx = i

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
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Phase 3 (v3): Model registration tool (anti-parallel-write-conflict)
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
        with open(real_metrics, "r", encoding="utf-8") as f:
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
