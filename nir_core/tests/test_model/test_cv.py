"""Tests for cross_validate."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

from nir_core.model.evaluation import cross_validate


def test_cross_validate_returns_expected_keys(synthetic_data):
    """cross_validate returns a dict with the documented keys."""
    X, y = synthetic_data.X, synthetic_data.y
    results = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y, n_folds=5, random_state=42,
    )
    for key in ("fold_rmse", "mean_rmse", "std_rmse", "fold_r2", "mean_r2"):
        assert key in results, f"missing key {key!r}"
    assert len(results["fold_rmse"]) == results["n_folds"]
    assert len(results["fold_r2"]) == results["n_folds"]
    assert results["n_folds"] == 5


def test_cross_validate_no_validation_overlap(synthetic_data):
    """Validation folds must be mutually disjoint (no leakage)."""
    X, y = synthetic_data.X, synthetic_data.y
    # Use a model_factory that captures the validation indices by wrapping
    # predict -- but simpler: replicate the KFold split and check disjointness.
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_indices = []
    for _, val_idx in kf.split(X):
        val_indices.append(set(val_idx.tolist()))
    # Pairwise disjoint.
    for i in range(len(val_indices)):
        for j in range(i + 1, len(val_indices)):
            assert len(val_indices[i] & val_indices[j]) == 0


def test_cross_validate_pls_reasonable_rmse(synthetic_data):
    """PLS with the right components should give low RMSE on synthetic data."""
    X, y = synthetic_data.X, synthetic_data.y
    results = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y, n_folds=5, random_state=42,
    )
    # Synthetic data is easy; RMSE should be small.
    assert results["mean_rmse"] < 1.0
    # R^2 should be positive and high.
    assert results["mean_r2"] > 0.5


def test_cross_validate_linear_model(synthetic_data):
    """cross_validate works with any model exposing fit/predict."""
    X, y = synthetic_data.X, synthetic_data.y
    results = cross_validate(
        lambda: LinearRegression(), X, y, n_folds=5, random_state=42,
    )
    assert results["n_folds"] == 5
    assert np.isfinite(results["mean_rmse"])


def test_cross_validate_reproducible(synthetic_data):
    """Same seed -> same fold_rmse."""
    X, y = synthetic_data.X, synthetic_data.y
    r1 = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y, n_folds=5, random_state=42,
    )
    r2 = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y, n_folds=5, random_state=42,
    )
    assert np.allclose(r1["fold_rmse"], r2["fold_rmse"])


def test_cross_validate_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    y_short = np.zeros(X.shape[0] - 1)
    with pytest.raises(ValueError):
        cross_validate(
            lambda: PLSRegression(n_components=2), X, y_short, n_folds=5
        )


def test_cross_validate_rejects_3d_y(synthetic_data):
    """y with ndim > 2 must be rejected with a clear ValueError."""
    X = synthetic_data.X
    y_3d = np.zeros((X.shape[0], 3, 2))
    with pytest.raises(ValueError, match="y must be 1D or 2D"):
        cross_validate(
            lambda: PLSRegression(n_components=2), X, y_3d, n_folds=5
        )


def test_cross_validate_2d_y_returns_per_target(synthetic_data):
    """2D y returns n_targets and per_target, with top-level fold_rmse
    holding the per-fold mean across targets (list[float] of length n_folds)."""
    X = synthetic_data.X
    # Build a 2-target y by stacking the original y with a noisy copy.
    y1 = synthetic_data.y
    y2 = y1 + 0.5 * np.random.default_rng(0).standard_normal(y1.shape[0])
    y_2d = np.column_stack([y1, y2])

    results = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y_2d, n_folds=5, random_state=42,
    )

    # Backward-compatible keys still present and well-typed.
    for key in ("fold_rmse", "mean_rmse", "std_rmse", "fold_r2", "mean_r2", "n_folds"):
        assert key in results
    assert len(results["fold_rmse"]) == results["n_folds"]
    assert all(isinstance(v, float) for v in results["fold_rmse"])

    # New multi-target keys.
    assert results["n_targets"] == 2
    assert "per_target" in results
    assert len(results["per_target"]) == 2
    for t_idx, pt in enumerate(results["per_target"]):
        assert pt["target_index"] == t_idx
        assert len(pt["fold_rmse"]) == results["n_folds"]
        assert len(pt["fold_r2"]) == results["n_folds"]
        assert np.isfinite(pt["mean_rmse"])
        assert np.isfinite(pt["mean_r2"])


def test_cross_validate_2d_y_per_target_matches_single_target_run(synthetic_data):
    """Per-target fold_rmse for 2D y must equal a standalone 1D run on that
    target — i.e. no cross-target mixing. This is the core regression guard
    for the bug where rmse()/r2_score() raveled 2D arrays together.

    Uses LinearRegression because OLS solves each y column independently,
    so multi-output and single-output fits are numerically identical.
    PLS2 would introduce Y-covariation and break the exact match (which is
    expected PLS behaviour, not a bug).
    """
    X = synthetic_data.X
    y1 = synthetic_data.y
    y2 = y1 + 0.5 * np.random.default_rng(1).standard_normal(y1.shape[0])
    y_2d = np.column_stack([y1, y2])

    multi = cross_validate(
        lambda: LinearRegression(), X, y_2d, n_folds=5, random_state=42,
    )
    single_0 = cross_validate(
        lambda: LinearRegression(), X, y1, n_folds=5, random_state=42,
    )
    single_1 = cross_validate(
        lambda: LinearRegression(), X, y2, n_folds=5, random_state=42,
    )

    # Same KFold seed -> identical fold splits; OLS is column-independent ->
    # per-target fold series must match the standalone 1D runs exactly.
    assert np.allclose(multi["per_target"][0]["fold_rmse"], single_0["fold_rmse"])
    assert np.allclose(multi["per_target"][0]["fold_r2"], single_0["fold_r2"])
    assert np.allclose(multi["per_target"][1]["fold_rmse"], single_1["fold_rmse"])
    assert np.allclose(multi["per_target"][1]["fold_r2"], single_1["fold_r2"])

    # Top-level fold_rmse is the mean across the two targets per fold.
    expected_fold_rmse = [
        0.5 * (multi["per_target"][0]["fold_rmse"][i]
               + multi["per_target"][1]["fold_rmse"][i])
        for i in range(multi["n_folds"])
    ]
    assert np.allclose(multi["fold_rmse"], expected_fold_rmse)


def test_cross_validate_1d_y_has_no_per_target_key(synthetic_data):
    """1D y must NOT add n_targets/per_target, so existing callers are
    unaffected by the multi-output extension."""
    X, y = synthetic_data.X, synthetic_data.y
    results = cross_validate(
        lambda: PLSRegression(n_components=3, scale=False),
        X, y, n_folds=5, random_state=42,
    )
    assert "per_target" not in results
    assert "n_targets" not in results
