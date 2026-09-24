"""NIR preprocessing tool: apply single method or multi-step pipeline."""

from __future__ import annotations

import os
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _load_npz_safely, _ok, _parse_pipeline_step, _resolve


@tool("nir_list_preprocessing_methods", parse_docstring=True)
def nir_list_preprocessing_methods_tool(
    runtime: Runtime,  # noqa: ARG001
    category: str | None = None,
    auto_level: str | None = None,
    include_non_stable: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """List the preprocessing methods available in the current runtime.

    This is a read-only compact catalog. Use
    ``nir_describe_preprocessing_method`` for the full parameter schema of one
    method instead of guessing parameter names or values.

    Args:
        category: Optional category filter such as baseline, scatter, smooth,
            derivative, despike, or scaling.
        auto_level: Optional availability filter: default, conditional,
            explicit_only, or disabled.
        include_non_stable: Include experimental, unavailable, or deprecated
            catalog records. Keep false for executable recommendations.

    Returns:
        JSON with catalog version/hash and compact method records.
    """
    try:
        from nir_core.preprocess.registry import (
            catalog_hash,
            catalog_snapshot,
            list_methods,
        )

        categories = None
        if category is not None:
            normalized_category = str(category).strip().lower()
            if not normalized_category:
                return _err(
                    "category must not be empty",
                    code="nir_preprocessing_catalog_filter_invalid",
                )
            categories = {normalized_category}

        auto_levels = None
        if auto_level is not None:
            normalized_level = str(auto_level).strip().lower()
            allowed_levels = {"default", "conditional", "explicit_only", "disabled"}
            if normalized_level not in allowed_levels:
                return _err(
                    f"Unknown auto_level {auto_level!r}; expected one of {sorted(allowed_levels)}",
                    code="nir_preprocessing_catalog_filter_invalid",
                )
            auto_levels = {normalized_level}

        statuses = None if include_non_stable else {"stable"}
        methods = list_methods(
            categories=categories,
            auto_levels=auto_levels,
            statuses=statuses,
        )
        snapshot = catalog_snapshot()
        return _ok(
            {
                "status": "ok",
                "catalog_version": snapshot["catalog_version"],
                "catalog_sha256": catalog_hash(),
                "count": len(methods),
                "filters": {
                    "category": category,
                    "auto_level": auto_level,
                    "include_non_stable": bool(include_non_stable),
                },
                "methods": methods,
                "next_action": ("Call nir_describe_preprocessing_method for a method before constructing parameterized pipeline_steps."),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_preprocessing_catalog_unavailable",
        )


@tool("nir_describe_preprocessing_method", parse_docstring=True)
def nir_describe_preprocessing_method_tool(
    runtime: Runtime,  # noqa: ARG001
    method: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Describe one preprocessing method and its validated parameters.

    Args:
        method: Stable public method id returned by
            ``nir_list_preprocessing_methods``.

    Returns:
        JSON with parameter schema, search space, data requirements,
        automatic-selection level, and versioned implementation providers.
    """
    try:
        from nir_core.preprocess.registry import (
            catalog_hash,
            catalog_snapshot,
            describe_method,
        )

        normalized = str(method).strip().lower()
        if not normalized:
            return _err(
                "method must not be empty",
                code="nir_preprocessing_method_unknown",
            )
        try:
            description = describe_method(normalized)
        except KeyError as exc:
            return _err(
                str(exc),
                code="nir_preprocessing_method_unknown",
            )
        snapshot = catalog_snapshot()
        return _ok(
            {
                "status": "ok",
                "catalog_version": snapshot["catalog_version"],
                "catalog_sha256": catalog_hash(),
                "method": description,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_preprocessing_catalog_unavailable",
        )


@tool("nir_recommend_preprocessing", parse_docstring=True)
def nir_recommend_preprocessing_tool(
    runtime: Runtime,
    input_path: str,
    budget: str = "standard",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Profile spectra and return deterministic, bounded preprocessing candidates.

    This read-only helper is suitable for explaining likely preprocessing
    choices. Production training independently recomputes the same diagnostics
    on its calibration partition so this exploratory call cannot leak a final
    holdout into candidate generation.

    Args:
        input_path: Virtual path to a numeric NPZ containing ``X`` and
            optionally ``wv``.
        budget: Candidate budget: small, standard, or extended.

    Returns:
        JSON with bounded diagnostics, candidate steps, provider identities,
        explicit-only recommendations, and exclusions. No spectra are returned.
    """
    try:
        from nir_core.preprocess.recommendation import recommend_preprocessing

        real_input = _resolve(runtime, input_path, read_only=True)
        data = _load_npz_safely(real_input)
        X = np.asarray(data["X"], dtype=float)
        wv = np.asarray(data["wv"], dtype=float).ravel() if data.get("wv") is not None else None
        recommendation = recommend_preprocessing(X, wv, budget=budget)
        return _ok(
            {
                "status": "ok",
                "scope": "exploratory_input_only",
                "recommendation": recommendation.as_dict(),
                "training_note": ("Training recomputes diagnostics on the calibration partition; do not treat this exploratory result as validation evidence."),
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_preprocessing_recommendation_invalid",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_preprocessing_recommendation_unavailable",
        )


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
      of method names or step dicts, e.g.
      ``'["snv", {"method":"sg_smooth","params":{"window":15}}]'``.
      All steps execute atomically in one call, reducing LLM round-trips.

    The input .npz must contain at least an ``X`` array. The result is saved
    to ``output_path``.

    Args:
        input_path: Virtual path to the input .npz file.
        output_path: Virtual path for the output .npz file.
        method: Single preprocessing method (backward-compatible). Query
            ``nir_list_preprocessing_methods`` for the authoritative runtime list.
        pipeline_steps: JSON array of method names or step dicts for multi-step atomic
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
            from nir_core.models import PreprocessingStep
            from nir_core.preprocess.pipeline import (
                PRESTEP_METHODS,
                PreprocessingPipeline,
                validate_pipeline,
            )
            from nir_core.preprocess.registry import CATALOG_VERSION, catalog_hash
        except ImportError:
            return _err(
                "nir_core V3 features (PreprocessingPipeline / PreprocessingStep) are not available in the current sandbox. Please rebuild the Docker image so the editable install of ../nir_core picks up the latest sources, then re-run."
            )

        # Resolve which mode: multi-step pipeline or single method.
        steps_list: list = []
        if pipeline_steps is not None:
            try:
                steps_list = _json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
            except (ValueError, TypeError):
                return _err(f"Invalid pipeline_steps JSON: {pipeline_steps!r}")
            if not isinstance(steps_list, list) or not steps_list:
                return _err("pipeline_steps must be a non-empty JSON array of method names or step dicts")
        elif method is not None:
            steps_list = [method]
        else:
            return _err("Either 'method' or 'pipeline_steps' must be provided")

        try:
            parsed_steps = [_parse_pipeline_step(m) for m in steps_list]
        except Exception as exc:  # noqa: BLE001
            return _err(f"Invalid pipeline step: {type(exc).__name__}: {exc}")

        method_names = [s.method for s in parsed_steps]

        # Validate all methods exist.
        for m in method_names:
            if m not in PRESTEP_METHODS:
                return _err(f"Unknown method {m!r}. Available: {sorted(PRESTEP_METHODS.keys())}")

        # Validate Savitzky-Golay parameters.
        sg_methods = {"sg_smooth", "derivative1", "derivative2"}
        if sg_methods & set(method_names):
            if window % 2 == 0:
                return _err(f"window must be odd, got {window}")
            if window <= order:
                return _err(f"window ({window}) must be > order ({order})")

        real_in = _resolve(runtime, input_path, read_only=True)
        real_out = _resolve(runtime, output_path, read_only=False)
        os.makedirs(os.path.dirname(real_out), exist_ok=True)

        data_dict = _load_npz_safely(real_in)
        X = np.asarray(data_dict["X"], dtype=float)

        # Build and apply pipeline atomically.
        steps = []
        for step in parsed_steps:
            m = step.method
            params: dict = dict(step.params or {})
            if m in sg_methods:
                params.setdefault("window", window)
                params.setdefault("order", order)
            elif m == "airpls":
                if lambda_ is not None and "lambda_" not in params:
                    params["lambda_"] = lambda_
            elif m == "asls":
                if lambda_ is not None and "lambda_" not in params:
                    params["lambda_"] = lambda_
                params.setdefault("p", p)
            elif m == "normalize":
                params.setdefault("norm", norm)
            steps.append(PreprocessingStep(method=m, params=params))

        is_valid, reason = validate_pipeline(steps)
        if not is_valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")

        pipe = PreprocessingPipeline(steps=steps)
        X_processed = pipe.apply(X, wv=(np.asarray(data_dict["wv"]).ravel() if data_dict.get("wv") is not None else None))
        data_dict["X"] = X_processed
        # Sync wavelength vector if preprocessing changed the wavelength count.
        if "wv" in data_dict and data_dict["wv"] is not None:
            wv_arr = np.asarray(data_dict["wv"]).ravel()
            if wv_arr.shape[0] > X_processed.shape[1]:
                data_dict["wv"] = wv_arr[: X_processed.shape[1]]
        np.savez(real_out, **data_dict)
        result = {
            "status": "ok",
            "pipeline": [{"method": s.method, "params": s.params} for s in steps],
            "description": pipe.description(),
            "catalog_version": CATALOG_VERSION,
            "catalog_sha256": catalog_hash(),
            "providers": pipe.provider_manifest(),
            "input_shape": list(X.shape),
            "output_shape": list(X_processed.shape),
            "output_path": output_path,
        }
        if pipe.has_stateful_steps:
            result["modeling_use"] = "forbidden"
            result["leakage_risk"] = (
                "Stateful steps (MSC/EMSC/AirPLS/ArPLS/mean_center/autoscale) were applied to the "
                "full dataset in one pass, so their statistics include the whole input. "
                "This output is for exploration, visualization, or export only; it must "
                "NOT be used as the data_path of a modeling tool. For modeling, pass the "
                "raw data plus pipeline_steps to the training tool so preprocessing is "
                "fitted inside each training fold."
            )
        return _ok(result)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_align_wavelengths", parse_docstring=True)
def nir_align_wavelengths_tool(
    runtime: Runtime,
    input_path: str,
    output_path: str,
    target_wavelengths: list[float] | None = None,
    reference_path: str | None = None,
    kind: str = "linear",
    allow_extrapolation: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Align spectra to an explicit or reference-file wavelength axis.

    This axis-aware operation is deliberately separate from ordinary
    ``pipeline_steps`` because it changes both the spectral matrix and its
    wavelength vector. Exactly one of ``target_wavelengths`` or
    ``reference_path`` must be provided. Extrapolation is disabled by default.

    Args:
        input_path: Virtual path to an NPZ containing ``X`` and ``wv``.
        output_path: Virtual path for the aligned NPZ.
        target_wavelengths: Explicit target wavelength values.
        reference_path: Optional NPZ whose ``wv`` becomes the target axis.
        kind: Interpolation kind: linear, nearest, or cubic.
        allow_extrapolation: Whether values outside the input range may be
            extrapolated. Keep false for production use.

    Returns:
        JSON status with input/output shapes, target-axis summary, and path.
    """
    try:
        has_explicit = target_wavelengths is not None
        has_reference = bool(reference_path)
        if has_explicit == has_reference:
            return _err("Provide exactly one of target_wavelengths or reference_path")

        from nir_core.preprocess.alignment import SpectralAxisAligner

        real_in = _resolve(runtime, input_path, read_only=True)
        real_out = _resolve(runtime, output_path, read_only=False)
        os.makedirs(os.path.dirname(real_out), exist_ok=True)
        data_dict = _load_npz_safely(real_in)
        if data_dict.get("wv") is None:
            return _err("Input NPZ must contain a wavelength vector 'wv'")
        X = np.asarray(data_dict["X"], dtype=float)
        source_wv = np.asarray(data_dict["wv"], dtype=float).ravel()

        if has_reference:
            real_reference = _resolve(runtime, str(reference_path), read_only=True)
            reference_data = _load_npz_safely(real_reference)
            if reference_data.get("wv") is None:
                return _err("Reference NPZ must contain a wavelength vector 'wv'")
            target_wv = np.asarray(reference_data["wv"], dtype=float).ravel()
        else:
            target_wv = np.asarray(target_wavelengths, dtype=float).ravel()

        aligner = SpectralAxisAligner(
            target_wv,
            kind=kind,
            allow_extrapolation=allow_extrapolation,
        ).fit(source_wv)
        X_aligned = aligner.transform(X, source_wv)
        data_dict["X"] = X_aligned
        data_dict["wv"] = aligner.target_wavelengths_
        np.savez(real_out, **data_dict)
        return _ok(
            {
                "status": "ok",
                "method": "wavelength_alignment",
                "kind": kind,
                "allow_extrapolation": bool(allow_extrapolation),
                "input_shape": list(X.shape),
                "output_shape": list(X_aligned.shape),
                "source_wavelength_range": [
                    float(source_wv.min()),
                    float(source_wv.max()),
                ],
                "target_wavelength_range": [
                    float(target_wv.min()),
                    float(target_wv.max()),
                ],
                "target_wavelength_count": int(target_wv.size),
                "source_axis_sha256": aligner.source_axis_hash_,
                "target_axis_sha256": aligner.target_axis_hash_,
                "output_path": output_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
