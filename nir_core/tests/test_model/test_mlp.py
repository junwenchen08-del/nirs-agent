"""Tests for MLP (neural network) training and prediction."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.mlp import predict_mlp, train_mlp


@pytest.fixture(scope="module")
def trained_mlp(synthetic_data):
    return train_mlp(synthetic_data.X, synthetic_data.y, cv_folds=3, random_state=42)


@pytest.mark.slow
def test_train_mlp_returns_model_and_grid_results(trained_mlp):
    """train_mlp returns a fitted model and a populated grid_results dict."""
    model, grid_results = trained_mlp
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_params" in grid_results
    assert "best_score" in grid_results
    assert "best_rmse" in grid_results
    assert "cv_strategy" in grid_results
    assert "param_grid" in grid_results
    bp = grid_results["best_params"]
    assert "hidden_layer_sizes" in bp
    assert "alpha" in bp
    assert np.isfinite(grid_results["best_rmse"])
    assert grid_results["best_rmse"] > 0


@pytest.mark.slow
def test_predict_mlp_shape(synthetic_data, trained_mlp):
    """predict_mlp returns a 1-D array of the right length."""
    model, _ = trained_mlp
    pred = predict_mlp(model, synthetic_data.X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == synthetic_data.X.shape[0]


@pytest.mark.slow
def test_train_mlp_achieves_reasonable_fit(synthetic_data, trained_mlp):
    """MLP should fit the synthetic data with R^2 > 0.3 (non-trivial)."""
    from nir_core.utils.metrics import r2_score
    X, y = synthetic_data.X, synthetic_data.y
    model, _ = trained_mlp
    pred = predict_mlp(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.3, f"MLP train R2={r2:.4f}"


@pytest.mark.slow
def test_train_mlp_reproducible(synthetic_data, trained_mlp):
    """Same seed -> identical best_params."""
    X, y = synthetic_data.X, synthetic_data.y
    _, g1 = trained_mlp
    _, g2 = train_mlp(X, y, cv_folds=3, random_state=42)
    assert g1["best_params"] == g2["best_params"]


def test_train_mlp_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    with pytest.raises(ValueError):
        train_mlp(X, np.zeros(X.shape[0] - 1), cv_folds=3)
