"""Tests for SVR training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.svr import predict_svr, train_svr


def test_train_svr_returns_model_and_grid_results(synthetic_data):
    """train_svr returns a fitted model and a populated grid_results dict."""
    X, y = synthetic_data.X, synthetic_data.y
    model, grid_results = train_svr(X, y, cv_folds=3, random_state=42)
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "best_score" in grid_results
    assert "best_rmse" in grid_results
    assert "cv_folds" in grid_results
    assert "param_grid" in grid_results
    # best_params must contain C and gamma.
    bp = grid_results["best_params"]
    assert "svr__C" in bp
    assert "svr__gamma" in bp
    # best_rmse should be a positive finite number.
    assert np.isfinite(grid_results["best_rmse"])
    assert grid_results["best_rmse"] > 0


def test_predict_svr_shape(synthetic_data):
    """predict_svr returns a 1-D array of the right length."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_svr(X, y, cv_folds=3, random_state=42)
    pred = predict_svr(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_train_svr_achieves_reasonable_fit(synthetic_data):
    """SVR should fit the synthetic data with R^2 > 0.5 (non-trivial)."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_svr(X, y, cv_folds=3, random_state=42)
    pred = predict_svr(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"SVR train R2={r2:.4f}"


def test_train_svr_reproducible(synthetic_data):
    """Same seed -> identical best_params."""
    X, y = synthetic_data.X, synthetic_data.y
    _, g1 = train_svr(X, y, cv_folds=3, random_state=42)
    _, g2 = train_svr(X, y, cv_folds=3, random_state=42)
    assert g1["best_params"] == g2["best_params"]


def test_train_svr_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_svr(X, np.zeros(X.shape[0] - 1), cv_folds=3)
