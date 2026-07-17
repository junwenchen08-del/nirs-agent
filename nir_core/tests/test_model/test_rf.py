"""Tests for Random Forest and Extra Trees training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.rf import predict_et, predict_rf, train_et, train_rf


@pytest.fixture(scope="module")
def trained_rf(synthetic_data):
    return train_rf(synthetic_data.X, synthetic_data.y, cv_folds=3, random_state=42)


@pytest.fixture(scope="module")
def trained_et(synthetic_data):
    return train_et(synthetic_data.X, synthetic_data.y, cv_folds=3, random_state=42)


@pytest.mark.slow
def test_train_rf_returns_model_and_grid_results(trained_rf):
    """train_rf returns a fitted model and a populated grid_results dict."""
    model, grid_results = trained_rf
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "best_score" in grid_results
    assert "best_rmse" in grid_results
    assert "cv_strategy" in grid_results
    assert "param_grid" in grid_results
    bp = grid_results["best_params"]
    assert "n_estimators" in bp
    assert "max_depth" in bp
    assert np.isfinite(grid_results["best_rmse"])
    assert grid_results["best_rmse"] > 0
    assert np.prod([len(values) for values in grid_results["param_grid"].values()]) == 4


@pytest.mark.slow
def test_predict_rf_shape(synthetic_data, trained_rf):
    """predict_rf returns a 1-D array of the right length."""
    model, _ = trained_rf
    pred = predict_rf(model, synthetic_data.X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == synthetic_data.X.shape[0]


@pytest.mark.slow
def test_train_rf_achieves_reasonable_fit(synthetic_data, trained_rf):
    """RF should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = trained_rf
    pred = predict_rf(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"RF train R2={r2:.4f}"


@pytest.mark.slow
def test_train_rf_reproducible(synthetic_data, trained_rf):
    """Same seed -> identical best_params."""
    X, y = synthetic_data.X, synthetic_data.y
    _, g1 = trained_rf
    _, g2 = train_rf(X, y, cv_folds=3, random_state=42)
    assert g1["best_params"] == g2["best_params"]


def test_train_rf_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_rf(X, np.zeros(X.shape[0] - 1), cv_folds=3)


# ---- Extra Trees ----


@pytest.mark.slow
def test_train_et_returns_model_and_grid_results(trained_et):
    """train_et returns a fitted model and a populated grid_results dict."""
    model, grid_results = trained_et
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert np.isfinite(grid_results["best_rmse"])
    assert np.prod([len(values) for values in grid_results["param_grid"].values()]) == 4


@pytest.mark.slow
def test_predict_et_shape(synthetic_data, trained_et):
    """predict_et returns a 1-D array of the right length."""
    model, _ = trained_et
    pred = predict_et(model, synthetic_data.X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == synthetic_data.X.shape[0]


@pytest.mark.slow
def test_train_et_achieves_reasonable_fit(synthetic_data, trained_et):
    """ET should fit the synthetic data with R^2 > 0.5."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = trained_et
    pred = predict_et(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.5, f"ET train R2={r2:.4f}"
