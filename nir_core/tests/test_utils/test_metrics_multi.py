"""Tests for per-component multi-target metrics."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.utils.metrics import compute_metrics, compute_metrics_multi


def test_compute_metrics_multi_matches_scalar_metrics() -> None:
    y_true = np.column_stack(
        (np.arange(10, dtype=float), np.arange(10, dtype=float) * 2)
    )
    y_pred = y_true + np.array([0.2, -0.4])

    metrics = compute_metrics_multi(y_true, y_pred, names=["protein", "moisture"])

    assert [item["name"] for item in metrics] == ["protein", "moisture"]
    assert metrics[0]["RMSE"] == compute_metrics(y_true[:, 0], y_pred[:, 0])["RMSE"]
    assert metrics[1]["RMSE"] == compute_metrics(y_true[:, 1], y_pred[:, 1])["RMSE"]


def test_compute_metrics_multi_validates_shape_and_names() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_metrics_multi(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError, match="names length"):
        compute_metrics_multi(np.zeros((5, 2)), np.zeros((5, 2)), names=["one"])
