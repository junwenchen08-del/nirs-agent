"""Shared helpers for NIR community tools.

Centralises path-resolution, JSON formatting, and pipeline-step parsing so
the individual tool modules (io_tools / preprocess / modeling / reflect /
knowledge) can stay focused on their domain logic.

Kept deliberately dependency-light: only stdlib + numpy + the DeerFlow
sandbox/runtime imports that every tool already needs.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline-step parsing
# ---------------------------------------------------------------------------


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
# Path-resolution helpers
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


# ---------------------------------------------------------------------------
# JSON formatting helpers
# ---------------------------------------------------------------------------


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
