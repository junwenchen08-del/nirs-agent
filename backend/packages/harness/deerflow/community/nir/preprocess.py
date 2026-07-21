"""NIR preprocessing tool: apply single method or multi-step pipeline."""

from __future__ import annotations

import os
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _load_npz_safely, _ok, _parse_pipeline_step, _resolve


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
        method: Single preprocessing method (backward-compatible). One of:
            snv, msc, sg_smooth, derivative1, derivative2, airpls, asls,
            detrend, mean_center, autoscale, normalize.
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
            from nir_core.preprocess.pipeline import PRESTEP_METHODS, PreprocessingPipeline
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
                "pipeline": [{"method": s.method, "params": s.params} for s in steps],
                "description": pipe.description(),
                "input_shape": list(X.shape),
                "output_shape": list(X_processed.shape),
                "output_path": output_path,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
