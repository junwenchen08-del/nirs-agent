"""Tests for PLS training and prediction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from nir_core.model.pls import (
    get_regression_coefficients,
    get_regression_intercept,
    predict_pls,
    train_pls,
)
from nir_core.utils.metrics import r2_score


def test_train_pls_auto_selects_components_near_truth(synthetic_data):
    """Auto-selected component count should be close to the true 3 latent vars."""
    X, y = synthetic_data.X, synthetic_data.y
    model, best_n, cv_results = train_pls(
        X, y, n_components=None, max_components=10, cv_folds=5, random_state=42
    )
    assert isinstance(model, PLSRegression)
    # Ground truth has 3 latent components; allow 2..5 (noise may shift it).
    assert 2 <= best_n <= 5, f"best_n_components={best_n} not near 3"
    # CV results shape.
    assert cv_results["best_n_components"] == best_n
    assert len(cv_results["n_components"]) >= best_n
    assert len(cv_results["mean_rmse_cv"]) == len(cv_results["n_components"])
    # R^2 on training set should be very high.
    pred = predict_pls(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.95, f"train R2={r2:.4f} <= 0.95"


def test_train_pls_fixed_components(synthetic_data):
    """Explicit n_components should bypass CV and train directly."""
    X, y = synthetic_data.X, synthetic_data.y
    model, best_n, cv_results = train_pls(X, y, n_components=3, cv_folds=5)
    assert best_n == 3
    # Single entry in the CV results (no search performed).
    assert cv_results["n_components"] == [3]
    pred = predict_pls(model, X)
    assert pred.shape == (X.shape[0],)
    r2 = r2_score(y, pred)
    assert r2 > 0.95


def test_predict_pls_shape_and_type(synthetic_data):
    """predict_pls returns a 1-D array of the right length."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _, _ = train_pls(X, y, n_components=3, cv_folds=5)
    pred = predict_pls(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_coefficients_and_intercept_reconstruct_predictions(synthetic_data):
    """The exported linear equation operates on raw, uncentered spectra."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _, _ = train_pls(X, y, n_components=3)

    coefficients = get_regression_coefficients(model)
    intercept = get_regression_intercept(model)

    np.testing.assert_allclose(
        X @ coefficients + intercept,
        predict_pls(model, X),
        atol=1e-10,
    )


def test_train_pls_reproducible(synthetic_data):
    """Same random_state should produce identical results."""
    X, y = synthetic_data.X, synthetic_data.y
    m1, n1, _ = train_pls(X, y, n_components=None, max_components=8,
                          cv_folds=5, random_state=42)
    m2, n2, _ = train_pls(X, y, n_components=None, max_components=8,
                          cv_folds=5, random_state=42)
    assert n1 == n2
    p1 = predict_pls(m1, X)
    p2 = predict_pls(m2, X)
    assert np.allclose(p1, p2)


def test_train_pls_invalid_n_components_raises(synthetic_data):
    """n_components < 1 must raise ValueError."""
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        train_pls(X, y, n_components=0)


def test_train_pls_shape_mismatch_raises(synthetic_data):
    """Mismatched X/y lengths must raise."""
    X = synthetic_data.X
    y_short = np.zeros(X.shape[0] - 1)
    with pytest.raises(ValueError):
        train_pls(X, y_short, n_components=2)
