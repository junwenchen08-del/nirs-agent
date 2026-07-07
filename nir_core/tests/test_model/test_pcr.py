"""Tests for PCR training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.pcr import PCRModel, predict_pcr, train_pcr
from nir_core.utils.metrics import r2_score


def test_train_pcr_auto_select(synthetic_data):
    """PCR auto-selects a component count and achieves high R^2."""
    X, y = synthetic_data.X, synthetic_data.y
    model, best_n, cv_results = train_pcr(
        X, y, n_components=None, max_components=10, cv_folds=5, random_state=42
    )
    assert isinstance(model, PCRModel)
    assert best_n >= 1
    assert cv_results["best_n_components"] == best_n
    assert len(cv_results["n_components"]) >= best_n
    pred = predict_pcr(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.90, f"PCR train R2={r2:.4f} <= 0.90"


def test_train_pcr_fixed_components(synthetic_data):
    """Explicit n_components bypasses the CV search."""
    X, y = synthetic_data.X, synthetic_data.y
    model, best_n, cv_results = train_pcr(X, y, n_components=5, cv_folds=5)
    assert best_n == 5
    assert cv_results["n_components"] == [5]
    pred = predict_pcr(model, X)
    assert pred.shape == (X.shape[0],)


def test_predict_pcr_shape(synthetic_data):
    """predict_pcr returns a 1-D array."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _, _ = train_pcr(X, y, n_components=4, cv_folds=5)
    pred = predict_pcr(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_train_pcr_reproducible(synthetic_data):
    """Same seed -> identical predictions."""
    X, y = synthetic_data.X, synthetic_data.y
    m1, n1, _ = train_pcr(X, y, n_components=None, max_components=8,
                          cv_folds=5, random_state=42)
    m2, n2, _ = train_pcr(X, y, n_components=None, max_components=8,
                          cv_folds=5, random_state=42)
    assert n1 == n2
    p1 = predict_pcr(m1, X)
    p2 = predict_pcr(m2, X)
    assert np.allclose(p1, p2)


def test_train_pcr_invalid_n_components_raises(synthetic_data):
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        train_pcr(X, y, n_components=0)


def test_train_pcr_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_pcr(X, np.zeros(X.shape[0] - 1), n_components=2)
