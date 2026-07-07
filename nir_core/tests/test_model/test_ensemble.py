"""Tests for ensemble prediction aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.ensemble import ensemble_predict


def test_mean_method():
    preds = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0, 5.0])]
    out = ensemble_predict(preds, method="mean")
    assert out.shape == (3,)
    assert np.allclose(out, [2.0, 3.0, 4.0])


def test_weighted_method():
    preds = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    # weights [3, 1] -> normalised [0.75, 0.25]
    out = ensemble_predict(preds, weights=[3.0, 1.0], method="weighted")
    expected = 0.75 * np.array([1.0, 2.0]) + 0.25 * np.array([3.0, 4.0])
    assert np.allclose(out, expected)


def test_weighted_normalises_weights():
    preds = [np.array([0.0]), np.array([10.0])]
    # Non-normalised weights [2, 2] should behave like [0.5, 0.5].
    out = ensemble_predict(preds, weights=[2.0, 2.0], method="weighted")
    assert np.allclose(out, [5.0])


def test_median_method():
    # Three models; the median per sample should pick the middle value.
    preds = [
        np.array([1.0, 10.0, 5.0]),
        np.array([2.0, 0.0, 5.0]),
        np.array([3.0, 5.0, 5.0]),
    ]
    out = ensemble_predict(preds, method="median")
    # Per column: median([1,2,3])=2, median([10,0,5])=5, median([5,5,5])=5
    assert np.allclose(out, [2.0, 5.0, 5.0])


def test_single_prediction_mean():
    """A single prediction should pass through unchanged for 'mean'."""
    preds = [np.array([1.0, 2.0, 3.0])]
    out = ensemble_predict(preds, method="mean")
    assert np.allclose(out, [1.0, 2.0, 3.0])


def test_empty_predictions_raises():
    with pytest.raises(ValueError):
        ensemble_predict([], method="mean")


def test_mismatched_lengths_raises():
    preds = [np.array([1.0, 2.0]), np.array([1.0])]
    with pytest.raises(ValueError):
        ensemble_predict(preds, method="mean")


def test_weighted_without_weights_raises():
    preds = [np.array([1.0]), np.array([2.0])]
    with pytest.raises(ValueError):
        ensemble_predict(preds, method="weighted")


def test_weighted_wrong_length_raises():
    preds = [np.array([1.0]), np.array([2.0])]
    with pytest.raises(ValueError):
        ensemble_predict(preds, weights=[1.0, 2.0, 3.0], method="weighted")


def test_unknown_method_raises():
    preds = [np.array([1.0, 2.0])]
    with pytest.raises(ValueError):
        ensemble_predict(preds, method="bogus")


def test_weighted_zero_sum_raises():
    preds = [np.array([1.0]), np.array([2.0])]
    with pytest.raises(ValueError):
        ensemble_predict(preds, weights=[0.0, 0.0], method="weighted")
