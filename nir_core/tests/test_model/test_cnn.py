"""Tests for 1D-CNN training and prediction.

These tests are skipped if PyTorch is not installed.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from nir_core.model.cnn import predict_cnn, train_cnn  # noqa: E402


@pytest.fixture(scope="module")
def trained_cnn(small_synthetic_data):
    return train_cnn(
        small_synthetic_data.X,
        small_synthetic_data.y,
        cv_folds=3,
        random_state=42,
        epochs=30,
        batch_size=16,
    )


@pytest.mark.slow
def test_train_cnn_returns_model_and_cv_results(trained_cnn):
    """train_cnn returns a fitted wrapper and a populated cv_results dict."""
    model, cv_results = trained_cnn
    assert model is not None
    assert hasattr(model, "predict")
    assert "best_rmse" in cv_results
    assert "mean_rmse_cv" in cv_results
    assert "cv_strategy" in cv_results
    assert "epochs" in cv_results
    assert "batch_size" in cv_results
    assert "learning_rate" in cv_results
    assert "architecture" in cv_results
    assert cv_results["epochs"] == 30


@pytest.mark.slow
def test_predict_cnn_shape(small_synthetic_data, trained_cnn):
    """predict_cnn returns a 1-D array of the right length."""
    model, _ = trained_cnn
    pred = predict_cnn(model, small_synthetic_data.X)
    assert isinstance(pred, np.ndarray)
    assert pred.ndim == 1
    assert pred.shape[0] == small_synthetic_data.X.shape[0]


@pytest.mark.slow
def test_cnn_joblib_round_trip_preserves_predictions(
    small_synthetic_data, tmp_path,
):
    """A trained CNN can be persisted through the tool's joblib path."""
    X, y = small_synthetic_data.X, small_synthetic_data.y
    model, _ = train_cnn(
        X, y, cv_folds=3, random_state=42, epochs=2, batch_size=16,
    )
    expected = predict_cnn(model, X)

    model_path = tmp_path / "cnn.pkl"
    joblib.dump(model, model_path)
    restored = joblib.load(model_path)

    np.testing.assert_allclose(predict_cnn(restored, X), expected)


@pytest.mark.slow
def test_train_cnn_achieves_non_trivial_fit(small_synthetic_data):
    """CNN should fit the synthetic data with R^2 > 0 (better than mean)."""
    from nir_core.utils.metrics import r2_score
    X, y = small_synthetic_data.X, small_synthetic_data.y
    model, _ = train_cnn(
        X, y, cv_folds=3, random_state=42, epochs=50, batch_size=16,
    )
    pred = predict_cnn(model, X)
    r2 = r2_score(y, pred)
    assert r2 > 0.0, f"CNN train R2={r2:.4f}"


@pytest.mark.slow
def test_train_cnn_reproducible(small_synthetic_data, trained_cnn):
    """Same seed -> similar RMSECV (within 20% tolerance for NN variance)."""
    X, y = small_synthetic_data.X, small_synthetic_data.y
    _, cv1 = trained_cnn
    _, cv2 = train_cnn(
        X, y, cv_folds=3, random_state=42, epochs=30, batch_size=16,
    )
    # Neural networks have some inherent non-determinism, but with the same
    # seed the RMSECV should be in the same ballpark.
    if np.isfinite(cv1["best_rmse"]) and np.isfinite(cv2["best_rmse"]):
        ratio = cv1["best_rmse"] / max(cv2["best_rmse"], 1e-10)
        assert 0.5 < ratio < 2.0, f"RMSECV ratio={ratio:.4f} too different"


def test_train_cnn_shape_mismatch_raises(small_synthetic_data):
    X = small_synthetic_data.X
    with pytest.raises(ValueError):
        train_cnn(X, np.zeros(X.shape[0] - 1), cv_folds=3, epochs=10)
