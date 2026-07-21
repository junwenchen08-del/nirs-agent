"""Tests for nir_core.utils.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nir_core.utils.metrics import (
    bias,
    compute_metrics,
    evaluate_quality,
    mae,
    r2_score,
    rmse,
    rpd,
    slope,
)


def test_rmse_perfect_linear(linear_data):
    x, y = linear_data
    assert rmse(y, y) == pytest.approx(0.0, abs=1e-12)


def test_r2_perfect_linear(linear_data):
    x, y = linear_data
    assert r2_score(y, y) == pytest.approx(1.0, abs=1e-12)


def test_bias_perfect(linear_data):
    x, y = linear_data
    assert bias(y, y) == pytest.approx(0.0, abs=1e-12)


def test_slope_perfect_linear(linear_data):
    """y_pred == y == 2*x + 1 -> slope of y on x is 2.0."""
    x, y = linear_data
    assert slope(x, y) == pytest.approx(2.0, abs=1e-10)


def test_mae_perfect(linear_data):
    x, y = linear_data
    assert mae(y, y) == pytest.approx(0.0, abs=1e-12)


def test_metrics_with_injected_error():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true + 0.1  # constant offset
    assert rmse(y_true, y_pred) == pytest.approx(0.1, abs=1e-12)
    assert bias(y_true, y_pred) == pytest.approx(0.1, abs=1e-12)
    assert mae(y_true, y_pred) == pytest.approx(0.1, abs=1e-12)
    # R^2 for a constant offset is less than 1 because the raw residuals
    # (y_true - y_pred) are all 0.1 and SS_res > 0. It should still be
    # close to 1 for a small offset relative to the y variance.
    assert r2_score(y_true, y_pred) < 1.0
    assert r2_score(y_true, y_pred) > 0.9
    # slope of (y_true vs y_pred) should be 1.0 (perfectly correlated,
    # since y_pred = y_true + const => regression slope = 1).
    assert slope(y_true, y_pred) == pytest.approx(1.0, abs=1e-10)


def test_r2_negative_for_bad_pred():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Predicting the mean of y_true yields R^2 == 0.
    y_pred_mean = np.full_like(y_true, fill_value=y_true.mean())
    assert r2_score(y_true, y_pred_mean) == pytest.approx(0.0, abs=1e-12)
    # A prediction worse than the mean (e.g. all 10s) gives R^2 < 0.
    y_pred_worse = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    assert r2_score(y_true, y_pred_worse) < 0.0


def test_rpd_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rpd(y, y) == float("inf")


def test_rpd_zero_std():
    y = np.array([3.0, 3.0, 3.0])
    # std 0 -> RPD 0
    assert rpd(y, y) == 0.0


def test_compute_metrics_has_all_keys():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    m = compute_metrics(y_true, y_pred)
    assert set(m.keys()) == {
        "RMSE",
        "R2",
        "RPD",
        "bias",
        "slope",
        "MAE",
        "n",
        "reference_mean",
        "reference_std",
    }


def test_compute_matches_manual_numpy():
    rng = np.random.default_rng(0)
    y_true = rng.normal(loc=10.0, scale=2.0, size=50)
    y_pred = y_true + rng.normal(loc=0.0, scale=0.3, size=50)

    m = compute_metrics(y_true, y_pred)
    # Manual numpy recompute.
    expected_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    expected_mae = float(np.mean(np.abs(y_true - y_pred)))
    expected_bias = float(np.mean(y_pred - y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    expected_r2 = 1.0 - ss_res / ss_tot
    std_y = float(np.std(y_true, ddof=1))
    expected_rpd = std_y / expected_rmse
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    expected_slope = float(
        np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / denom
    )

    assert_allclose(m["RMSE"], expected_rmse, rtol=1e-12)
    assert_allclose(m["MAE"], expected_mae, rtol=1e-12)
    assert_allclose(m["bias"], expected_bias, rtol=1e-12)
    assert_allclose(m["R2"], expected_r2, rtol=1e-12)
    assert_allclose(m["RPD"], expected_rpd, rtol=1e-12)
    assert_allclose(m["slope"], expected_slope, rtol=1e-12)


def test_evaluate_quality_excellent():
    metrics = {
        "R2_val": 0.99,
        "RPD": 6.0,
        "RMSEP": 0.1,
        "RMSECV": 0.05,
        "test": {"bias": 0.0},
    }
    res = evaluate_quality(metrics, domain="default")
    assert res["grade"] == "excellent"
    assert res["passed"] is True
    assert res["action"] == "proceed"
    assert "thresholds_used" in res


def test_evaluate_quality_good():
    # default thresholds: r2 >= 0.80, rpd >= 3.0; just meet them.
    metrics = {"R2_val": 0.85, "RPD": 3.5, "RMSEP": 0.2, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="default")
    assert res["grade"] == "good"
    assert res["passed"] is True
    assert res["action"] == "proceed"


def test_evaluate_quality_fair():
    metrics = {"R2_val": 0.70, "RPD": 2.0, "RMSEP": 0.4, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="default")
    # default min_r2=0.80, min_rpd=3.0; r2>=0.60 and rpd>=1.5 -> fair
    assert res["grade"] == "fair"
    assert res["passed"] is False
    assert res["action"] == "retry_preprocessing"


def test_evaluate_quality_poor():
    metrics = {"R2_val": 0.30, "RPD": 1.0, "RMSEP": 0.9, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="default")
    assert res["grade"] == "poor"
    assert res["passed"] is False
    assert res["action"] == "investigate_data"


def test_evaluate_quality_domain_soil_relaxed():
    # soil: min_r2=0.70, min_rpd=2.0
    metrics = {"R2_val": 0.75, "RPD": 2.2, "RMSEP": 0.3, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="soil")
    assert res["grade"] == "good"
    assert res["passed"] is True
    # Same metrics under food_moisture (strict) should be worse.
    res_strict = evaluate_quality(metrics, domain="food_moisture")
    assert res_strict["grade"] in ("fair", "poor")
    assert res_strict["passed"] is False


def test_evaluate_quality_food_moisture_strict():
    metrics = {"R2_val": 0.88, "RPD": 3.8, "RMSEP": 0.2, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="food_moisture")
    # food_moisture: min_r2=0.90, min_rpd=4.0 -> fails good but meets fair
    assert res["passed"] is False


def test_evaluate_quality_small_sample_relaxed():
    metrics = {"R2_val": 0.72, "RPD": 1.8, "RMSEP": 0.3, "test": {"bias": 0.0}}
    # n_samples < 100 -> relaxed to min_r2=0.70, min_rpd=2.5
    res = evaluate_quality(metrics, domain="default", n_samples=60)
    # r2=0.72 >= 0.70; rpd=1.8 < 2.5 -> fair (rpd fair tier)
    assert res["grade"] in ("fair", "good")
    # Without small-sample relaxation (n_samples=None), default 0.80/3.0 applies.
    res_full = evaluate_quality(metrics, domain="default")
    assert res_full["grade"] in ("fair", "poor")


def test_evaluate_quality_overfitting_detected():
    metrics = {
        "R2_val": 0.95,
        "RPD": 5.0,
        "RMSEP": 0.5,
        "RMSECV": 0.1,  # RMSEP > 2 * RMSECV
        "test": {"bias": 0.0},
    }
    res = evaluate_quality(metrics, domain="default")
    assert res["details"]["overfitting_risk"] is True
    assert res["action"] == "retry_preprocessing"


def test_evaluate_quality_no_overfitting_without_rmsecv():
    metrics = {
        "R2_val": 0.95,
        "RPD": 5.0,
        "RMSEP": 0.5,
        "test": {"bias": 0.0},
    }
    res = evaluate_quality(metrics, domain="default")
    assert res["details"]["overfitting_risk"] is False
    assert res["action"] == "proceed"


def test_evaluate_quality_bias_issue():
    # mean y ~= 5, bias 1.0 > 0.1 * 5 = 0.5 -> bias issue
    metrics = {
        "R2_val": 0.95,
        "RPD": 5.0,
        "RMSEP": 0.3,
        "test": {"bias": 1.0},
        "y": np.array([4.0, 5.0, 6.0]),
    }
    res = evaluate_quality(metrics, domain="default")
    assert res["details"]["bias_issue"] is True


def test_evaluate_quality_no_bias_issue_for_zero_bias():
    metrics = {
        "R2_val": 0.95,
        "RPD": 5.0,
        "RMSEP": 0.3,
        "test": {"bias": 0.0},
        "y": np.array([4.0, 5.0, 6.0]),
    }
    res = evaluate_quality(metrics, domain="default")
    assert res["details"]["bias_issue"] is False


def test_evaluate_quality_r2_key_fallback():
    # Uses "R2" instead of "R2_val" - should still work.
    metrics = {"R2": 0.99, "RPD": 6.0, "RMSEP": 0.1, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="default")
    assert res["grade"] == "excellent"


def test_evaluate_quality_thresholds_transparent():
    metrics = {"R2_val": 0.99, "RPD": 6.0, "RMSEP": 0.1, "test": {"bias": 0.0}}
    res = evaluate_quality(metrics, domain="soil", n_samples=50)
    th = res["thresholds_used"]
    assert th["domain"] == "soil"
    assert th["n_samples"] == 50
    assert "min_r2" in th and "min_rpd" in th
