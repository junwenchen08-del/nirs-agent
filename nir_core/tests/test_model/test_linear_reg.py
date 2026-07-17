"""Tests for Ridge / Lasso / ElasticNet training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.linear_reg import (
    predict_elasticnet,
    predict_lasso,
    predict_ridge,
    train_elasticnet,
    train_lasso,
    train_ridge,
)


# ---- Ridge ----


def test_train_ridge_returns_model_and_grid_results(synthetic_data):
    """train_ridge returns a fitted model and a populated grid_results dict."""
    X, y = synthetic_data.X, synthetic_data.y
    model, grid_results = train_ridge(X, y, cv_folds=3, random_state=42)
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "best_score" in grid_results
    assert "best_rmse" in grid_results
    assert "cv_strategy" in grid_results
    assert "param_grid" in grid_results
    bp = grid_results["best_params"]
    assert "estimator__alpha" in bp
    assert np.isfinite(grid_results["best_rmse"])
    assert grid_results["best_rmse"] > 0


def test_predict_ridge_shape(synthetic_data):
    """predict_ridge returns a 1-D array of the right length."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_ridge(X, y, cv_folds=3, random_state=42)
    pred = predict_ridge(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_train_ridge_achieves_reasonable_fit(synthetic_data):
    """Ridge should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_ridge(X, y, cv_folds=3, random_state=42)
    pred = predict_ridge(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"Ridge train R2={r2:.4f}"


def test_train_ridge_reproducible(synthetic_data):
    """Same seed -> identical best_params."""
    X, y = synthetic_data.X, synthetic_data.y
    _, g1 = train_ridge(X, y, cv_folds=3, random_state=42)
    _, g2 = train_ridge(X, y, cv_folds=3, random_state=42)
    assert g1["best_params"] == g2["best_params"]


def test_train_ridge_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_ridge(X, np.zeros(X.shape[0] - 1), cv_folds=3)


# ---- Lasso ----


def test_train_lasso_returns_model_and_grid_results(synthetic_data):
    """train_lasso returns a fitted model and a populated grid_results dict."""
    X, y = synthetic_data.X, synthetic_data.y
    model, grid_results = train_lasso(X, y, cv_folds=3, random_state=42)
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "estimator__alpha" in grid_results["best_params"]
    assert np.isfinite(grid_results["best_rmse"])


def test_predict_lasso_shape(synthetic_data):
    """predict_lasso returns a 1-D array of the right length."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_lasso(X, y, cv_folds=3, random_state=42)
    pred = predict_lasso(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_train_lasso_achieves_reasonable_fit(synthetic_data):
    """Lasso should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_lasso(X, y, cv_folds=3, random_state=42)
    pred = predict_lasso(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"Lasso train R2={r2:.4f}"


# ---- ElasticNet ----


def test_train_elasticnet_returns_model_and_grid_results(synthetic_data):
    """train_elasticnet returns a fitted model and a populated grid_results dict."""
    X, y = synthetic_data.X, synthetic_data.y
    model, grid_results = train_elasticnet(X, y, cv_folds=3, random_state=42)
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    bp = grid_results["best_params"]
    assert "estimator__alpha" in bp
    assert "estimator__l1_ratio" in bp
    assert np.isfinite(grid_results["best_rmse"])


def test_predict_elasticnet_shape(synthetic_data):
    """predict_elasticnet returns a 1-D array of the right length."""
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_elasticnet(X, y, cv_folds=3, random_state=42)
    pred = predict_elasticnet(model, X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == X.shape[0]


def test_train_elasticnet_achieves_reasonable_fit(synthetic_data):
    """ElasticNet should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = train_elasticnet(X, y, cv_folds=3, random_state=42)
    pred = predict_elasticnet(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.3, f"ElasticNet train R2={r2:.4f}"
