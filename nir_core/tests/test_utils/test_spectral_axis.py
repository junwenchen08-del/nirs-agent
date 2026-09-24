"""Tests for compact, JSON-safe spectral-axis summaries."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.utils.spectral_axis import summarize_spectral_axis


@pytest.mark.parametrize(
    ("values", "first", "last", "direction"),
    [
        (None, None, None, "missing"),
        ([], None, None, "missing"),
        ([1000.0], 1000.0, 1000.0, "single_point"),
        ([1000.0, 1100.0, 1200.0], 1000.0, 1200.0, "ascending"),
        ([1200.0, 1100.0, 1000.0], 1200.0, 1000.0, "descending"),
        ([1000.0, 1100.0, 1050.0], 1000.0, 1050.0, "non_monotonic"),
        ([1000.0, 1000.0, 1100.0], 1000.0, 1100.0, "duplicate_values"),
        ([np.nan, 1100.0], None, 1100.0, "invalid"),
        ([1000.0, np.inf], 1000.0, None, "invalid"),
    ],
)
def test_summarize_spectral_axis_reports_order_and_quality(
    values: object,
    first: float | None,
    last: float | None,
    direction: str,
) -> None:
    assert summarize_spectral_axis(values) == {
        "axis_first": first,
        "axis_last": last,
        "axis_direction": direction,
    }
