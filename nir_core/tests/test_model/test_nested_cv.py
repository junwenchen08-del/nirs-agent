"""Tests for nested_cv_preprocessing and auto_select_components."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.evaluation import (
    auto_select_components,
    nested_cv_preprocessing,
)
from nir_core.model.pls import train_pls
from nir_core.preprocess.pipeline import PreprocessingPipeline
from nir_core.models import PreprocessingStep


@pytest.fixture(scope="module")
def nested_cv_result(synthetic_data):
    """Build one representative nested-CV result for contract assertions."""
    X, y = synthetic_data.X, synthetic_data.y
    from nir_core.model.evaluation import split_dataset
    (X_tr, y_tr), (X_val, y_val), _ = split_dataset(
        X, y, test_ratio=0.20, val_ratio=0.15, random_state=42
    )
    best_pipe, results = nested_cv_preprocessing(
        X_tr, y_tr, X_val, y_val,
        inner_folds=3, model_method="pls", max_components=6, random_state=42,
    )
    return X_tr, y_tr, X_val, y_val, best_pipe, results


@pytest.mark.slow
def test_nested_cv_returns_pipeline_and_results(nested_cv_result):
    """nested_cv_preprocessing returns a PreprocessingPipeline + results dict."""
    _, _, _, _, best_pipe, results = nested_cv_result
    assert isinstance(best_pipe, PreprocessingPipeline)
    assert "candidates" in results
    assert "best" in results
    # Each candidate has the expected keys.
    for desc, info in results["candidates"].items():
        assert "score" in info
        assert "cv_r2" in info
        assert "val_r2" in info
        assert "best_n_comp" in info
    # Best candidate's description matches.
    assert results["best"] == best_pipe.description(locale="zh")


@pytest.mark.slow
def test_nested_cv_validation_not_in_inner_search(nested_cv_result):
    """The validation set must not appear in the inner CV component search.

    We verify this indirectly: the cv_r2 reported for each candidate is
    computed using only X_train, so swapping the validation set for a
    different one should leave cv_r2 essentially unchanged (within
    numerical noise from PLS itself).
    """
    X_tr, y_tr, X_val, y_val, _, r1 = nested_cv_result
    # Run with a shuffled validation set (different rows).
    rng = np.random.default_rng(99)
    perm = rng.permutation(len(y_val))
    X_val2 = X_val[perm]
    y_val2 = y_val[perm]
    _, r2 = nested_cv_preprocessing(
        X_tr, y_tr, X_val2, y_val2,
        inner_folds=3, max_components=6, random_state=42,
    )
    # cv_r2 for each candidate should be identical (depends on X_train only).
    for desc in r1["candidates"]:
        cv_r2_1 = r1["candidates"][desc]["cv_r2"]
        cv_r2_2 = r2["candidates"][desc]["cv_r2"]
        assert abs(cv_r2_1 - cv_r2_2) < 1e-9, (
            f"cv_r2 differs ({cv_r2_1} vs {cv_r2_2}) -- validation leaked?"
        )
    # val_r2 should differ (different val set order -> same values though;
    # PLS prediction is permutation-invariant in y but R^2 is row-order
    # invariant, so val_r2 should actually match here too). The key check
    # is that cv_r2 is identical.


@pytest.mark.slow
def test_nested_cv_custom_pipelines(synthetic_data):
    """Passing custom candidate pipelines works."""
    X, y = synthetic_data.X, synthetic_data.y
    from nir_core.model.evaluation import split_dataset
    (X_tr, y_tr), (X_val, y_val), _ = split_dataset(
        X, y, test_ratio=0.20, val_ratio=0.15, random_state=42
    )
    custom = [
        PreprocessingPipeline([PreprocessingStep(method="snv")]),
        PreprocessingPipeline([PreprocessingStep(method="mean_center")]),
    ]
    best_pipe, results = nested_cv_preprocessing(
        X_tr, y_tr, X_val, y_val,
        candidate_pipelines=custom,
        inner_folds=3, max_components=6, random_state=42,
    )
    assert best_pipe in custom
    assert len(results["candidates"]) == 2


def test_nested_cv_unsupported_method_raises(synthetic_data):
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        nested_cv_preprocessing(X, y, X, y, model_method="random_forest")


# ---------------------------------------------------------------------------
# auto_select_components tests.
# ---------------------------------------------------------------------------


def test_auto_select_min_rmsecv():
    """min_rmsECV picks the minimum RMSECV."""
    cv_results = {
        "n_components": [1, 2, 3, 4, 5],
        "mean_rmse_cv": [1.0, 0.5, 0.3, 0.4, 0.6],
    }
    n = auto_select_components(cv_results, method="min_rmsECV")
    assert n == 3  # minimum at index 2


def test_auto_select_min_rmsecv_ties_fewest_components():
    """Ties should resolve to the fewest components."""
    cv_results = {
        "n_components": [1, 2, 3],
        "mean_rmse_cv": [0.5, 0.5, 0.5],
    }
    n = auto_select_components(cv_results, method="min_rmsECV")
    assert n == 1


def test_auto_select_first_minimum():
    """first_minimum returns the first local minimum."""
    cv_results = {
        "n_components": [1, 2, 3, 4, 5],
        "mean_rmse_cv": [1.0, 0.8, 0.9, 0.7, 0.6],
    }
    # 0.8 at index 1 is the first local minimum (next value 0.9 rises).
    n = auto_select_components(cv_results, method="first_minimum")
    assert n == 2


def test_auto_select_first_minimum_monotonic():
    """Monotonically decreasing curve -> returns the last (min)."""
    cv_results = {
        "n_components": [1, 2, 3],
        "mean_rmse_cv": [1.0, 0.8, 0.6],
    }
    n = auto_select_components(cv_results, method="first_minimum")
    assert n == 3


def test_auto_select_haaland_thomas_returns_fewer_than_min():
    """haaland_thomas should be more conservative than min_rmsECV."""
    cv_results = {
        "n_components": [1, 2, 3, 4, 5],
        # Component 1 is within 5% of the global min (0.5).
        "mean_rmse_cv": [0.51, 0.50, 0.55, 0.70, 0.90],
    }
    n_ht = auto_select_components(cv_results, method="haaland_thomas")
    n_min = auto_select_components(cv_results, method="min_rmsECV")
    assert n_ht <= n_min
    # Component 1 (0.51 / 0.50 = 1.02 <= 1.05) should be chosen.
    assert n_ht == 1


def test_auto_select_haaland_thomas_falls_back_to_min():
    """When no early component is within the threshold, falls back to the min."""
    cv_results = {
        "n_components": [1, 2, 3],
        "mean_rmse_cv": [2.0, 1.0, 0.5],  # ratio 4.0, 2.0 -> both > 1.05
    }
    n = auto_select_components(cv_results, method="haaland_thomas")
    assert n == 3  # falls back to the global min


def test_auto_select_unknown_method_raises():
    cv_results = {"n_components": [1, 2], "mean_rmse_cv": [1.0, 0.5]}
    with pytest.raises(ValueError):
        auto_select_components(cv_results, method="bogus")


def test_auto_select_missing_keys_raises():
    with pytest.raises(ValueError):
        auto_select_components({"n_components": [1]}, method="min_rmsECV")


def test_auto_select_on_real_pls_results(synthetic_data):
    """auto_select_components works with a real train_pls cv_results dict."""
    X, y = synthetic_data.X, synthetic_data.y
    _, best_n, cv_results = train_pls(
        X, y, n_components=None, max_components=6, cv_folds=5, random_state=42
    )
    for method in ("min_rmsECV", "first_minimum", "haaland_thomas"):
        n = auto_select_components(cv_results, method=method)
        assert 1 <= n <= 6
