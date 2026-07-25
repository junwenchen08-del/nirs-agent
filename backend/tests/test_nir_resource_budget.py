from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from nir_core.io.resources import ResourceLimitError

from deerflow.community.nir._common import _err
from deerflow.community.nir._resources import budget_for_runtime, check_spectral_data
from deerflow.runtime.cancellation import (
    is_run_cancelled,
    register_run_cancellation,
    unregister_run_cancellation,
)


def test_run_cancellation_registry_tracks_async_event() -> None:
    event = asyncio.Event()
    register_run_cancellation("run-1", event)
    try:
        assert is_run_cancelled("run-1") is False
        event.set()
        assert is_run_cancelled("run-1") is True
    finally:
        unregister_run_cancellation("run-1")
    assert is_run_cancelled("run-1") is False


def test_runtime_budget_observes_registered_cancellation() -> None:
    event = asyncio.Event()
    register_run_cancellation("run-2", event)
    try:
        budget = budget_for_runtime(SimpleNamespace(context={"run_id": "run-2"}))
        event.set()
        with pytest.raises(ResourceLimitError, match="cancelled"):
            budget.checkpoint("before_fit")
    finally:
        unregister_run_cancellation("run-2")


def test_spectral_data_check_counts_multi_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NIR_MAX_TARGETS", "1")
    data = SimpleNamespace(X=np.ones((3, 4)), y=np.ones((3, 2)))

    with pytest.raises(ResourceLimitError) as exc_info:
        check_spectral_data(budget_for_runtime(None), data, stage="loaded_data")

    assert exc_info.value.code == "resource_matrix_dimension_exceeded"
    assert exc_info.value.details["dimension"] == "targets"


def test_structured_tool_error_preserves_backward_compatible_error_field() -> None:
    payload = json.loads(
        _err(
            "too large",
            code="resource_file_too_large",
            details={"actual_bytes": 33, "limit_bytes": 32},
        )
    )

    assert payload == {
        "status": "error",
        "error": "too large",
        "code": "resource_file_too_large",
        "details": {"actual_bytes": 33, "limit_bytes": 32},
    }
