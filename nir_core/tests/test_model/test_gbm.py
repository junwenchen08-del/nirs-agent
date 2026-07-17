"""Tests for Gradient Boosting training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.gbm import predict_gbm, train_gbm


@pytest.fixture(scope="module")
def trained_gbm(synthetic_data):
    """Train once for assertions that inspect the same deterministic fit."""
    return train_gbm(synthetic_data.X, synthetic_data.y, cv_folds=3, random_state=42)


@pytest.mark.slow
def test_train_gbm_returns_model_and_grid_results(trained_gbm):
    """train_gbm returns a fitted model and a populated grid_results dict."""
    model, grid_results = trained_gbm
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "best_score" in grid_results
    assert "best_rmse" in grid_results
    assert "cv_strategy" in grid_results
    assert "param_grid" in grid_results
    bp = grid_results["best_params"]
    assert "n_estimators" in bp
    assert "learning_rate" in bp
    assert "max_depth" in bp
    assert np.isfinite(grid_results["best_rmse"])
    assert grid_results["best_rmse"] > 0
    assert np.prod([len(values) for values in grid_results["param_grid"].values()]) == 8


@pytest.mark.slow
def test_predict_gbm_shape(synthetic_data, trained_gbm):
    """predict_gbm returns a 1-D array of the right length."""
    model, _ = trained_gbm
    pred = predict_gbm(model, synthetic_data.X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == synthetic_data.X.shape[0]


@pytest.mark.slow
def test_train_gbm_achieves_reasonable_fit(synthetic_data, trained_gbm):
    """GBM should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = trained_gbm
    pred = predict_gbm(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"GBM train R2={r2:.4f}"


@pytest.mark.slow
def test_train_gbm_reproducible(synthetic_data, trained_gbm):
    """Same seed -> identical best_params."""
    X, y = synthetic_data.X, synthetic_data.y
    _, g1 = trained_gbm
    _, g2 = train_gbm(X, y, cv_folds=3, random_state=42)
    assert g1["best_params"] == g2["best_params"]


def test_train_gbm_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_gbm(X, np.zeros(X.shape[0] - 1), cv_folds=3)
