"""NIR tool adapters for core resource budgets and run cancellation."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from nir_core.io.resources import ResourceBudget, ResourceLimitError

from deerflow.runtime.cancellation import is_run_cancelled

from ._common import _err


def _runtime_run_id(runtime: Any | None) -> str | None:
    context = getattr(runtime, "context", None)
    if not isinstance(context, dict):
        return None
    value = context.get("run_id")
    return str(value) if value else None


def budget_for_runtime(runtime: Any | None) -> ResourceBudget:
    """Create one budget linked to the current gateway run when available."""
    run_id = _runtime_run_id(runtime)
    cancel_check = partial(is_run_cancelled, run_id) if run_id else None
    return ResourceBudget(cancel_check=cancel_check)


def check_spectral_data(
    budget: ResourceBudget,
    data: Any,
    *,
    stage: str,
    peak_multiplier: float = 3.0,
) -> int:
    """Apply matrix, target-count, and memory limits to loaded spectral data."""
    X = np.asarray(data.X)
    y = getattr(data, "y", None)
    target_count = 0
    if y is not None:
        y_array = np.asarray(y)
        target_count = 1 if y_array.ndim <= 1 else int(y_array.shape[1])
    return budget.check_array(
        X,
        target_count=target_count,
        peak_multiplier=peak_multiplier,
        stage=stage,
    )


def resource_error(exc: ResourceLimitError) -> str:
    """Render a resource failure in the common structured tool format."""
    return _err(
        str(exc),
        code=exc.code,
        details={"stage": exc.stage, **exc.details},
    )
