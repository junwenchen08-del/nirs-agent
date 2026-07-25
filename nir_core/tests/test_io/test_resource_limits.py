"""Resource-budget regression tests for large NIR inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nir_core.io.resources import ResourceBudget, ResourceLimitError, ResourceLimits
from nir_core.io.loaders import load_csv


def _limits(**overrides: int | float) -> ResourceLimits:
    values: dict[str, int | float] = {
        "max_file_bytes": 1024,
        "max_matrix_elements": 100,
        "max_samples": 20,
        "max_wavelengths": 20,
        "max_targets": 4,
        "max_estimated_peak_bytes": 4096,
        "max_runtime_seconds": 30.0,
    }
    values.update(overrides)
    return ResourceLimits(**values)


def test_budget_rejects_oversized_file_before_parsing(tmp_path: Path) -> None:
    source = tmp_path / "large.csv"
    source.write_bytes(b"x" * 33)

    budget = ResourceBudget(_limits(max_file_bytes=32))

    with pytest.raises(ResourceLimitError) as exc_info:
        budget.check_file(source, stage="load")

    assert exc_info.value.code == "resource_file_too_large"
    assert exc_info.value.details["actual_bytes"] == 33
    assert exc_info.value.details["limit_bytes"] == 32


def test_budget_rejects_matrix_dimensions_and_element_count() -> None:
    dimension_budget = ResourceBudget(_limits(max_samples=4))
    with pytest.raises(ResourceLimitError) as exc_info:
        dimension_budget.check_matrix_shape((5, 3), stage="loaded_data")
    assert exc_info.value.code == "resource_matrix_dimension_exceeded"
    assert exc_info.value.details["dimension"] == "samples"

    element_budget = ResourceBudget(
        _limits(max_samples=20, max_wavelengths=20, max_matrix_elements=12)
    )
    with pytest.raises(ResourceLimitError) as exc_info:
        element_budget.check_matrix_shape((4, 4), stage="loaded_data")
    assert exc_info.value.code == "resource_matrix_too_large"
    assert exc_info.value.details["actual_elements"] == 16


def test_budget_rejects_estimated_peak_memory() -> None:
    budget = ResourceBudget(_limits(max_estimated_peak_bytes=255))

    with pytest.raises(ResourceLimitError) as exc_info:
        budget.check_matrix_shape(
            (4, 4), dtype=np.float64, peak_multiplier=2.0, stage="model_fit"
        )

    assert exc_info.value.code == "resource_memory_estimate_exceeded"
    assert exc_info.value.details["estimated_peak_bytes"] == 256


def test_budget_honours_deadline_and_soft_cancellation() -> None:
    now = [10.0]
    deadline_budget = ResourceBudget(
        _limits(max_runtime_seconds=2.0), clock=lambda: now[0]
    )
    now[0] = 12.1

    with pytest.raises(ResourceLimitError) as exc_info:
        deadline_budget.checkpoint("candidate_selection")
    assert exc_info.value.code == "resource_deadline_exceeded"

    cancelled_budget = ResourceBudget(
        _limits(), cancel_check=lambda: True, clock=lambda: 20.0
    )
    with pytest.raises(ResourceLimitError) as exc_info:
        cancelled_budget.checkpoint("preprocess")
    assert exc_info.value.code == "resource_cancelled"


def test_budget_evidence_records_high_water_marks() -> None:
    budget = ResourceBudget(_limits(), clock=lambda: 5.0)
    budget.check_matrix_shape(
        (4, 5), dtype=np.float32, peak_multiplier=3.0, stage="loaded_data"
    )

    evidence = budget.evidence()

    assert evidence["max_observed_elements"] == 20
    assert evidence["max_estimated_peak_bytes"] == 240
    assert evidence["last_stage"] == "loaded_data"


def test_public_csv_loader_enforces_environment_matrix_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "bounded.csv"
    source.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    monkeypatch.setenv("NIR_MAX_MATRIX_ELEMENTS", "5")

    with pytest.raises(ResourceLimitError) as exc_info:
        load_csv(str(source), auto_layout=False)

    assert exc_info.value.code == "resource_matrix_too_large"
