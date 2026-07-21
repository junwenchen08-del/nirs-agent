"""Tests for evaluate_quality domain/grade coverage (kept separate from
the core metric arithmetic tests in test_metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.utils.metrics import evaluate_quality


def _metrics(
    r2: float,
    rpd: float,
    rmsep: float = 0.2,
    rmsecv: float | None = None,
    bias_val: float = 0.0,
) -> dict:
    m: dict = {
        "R2_val": r2,
        "RPD": rpd,
        "RMSEP": rmsep,
        "test": {"bias": bias_val},
    }
    if rmsecv is not None:
        m["RMSECV"] = rmsecv
    return m


@pytest.mark.parametrize(
    "domain,r2,rpd,expected_grade",
    [
        # default thresholds: min_r2=0.80, min_rpd=3.0
        ("default", 0.95, 4.5, "excellent"),
        ("default", 0.82, 3.1, "good"),
        ("default", 0.65, 1.8, "fair"),
        ("default", 0.40, 1.0, "poor"),
        # soil: min_r2=0.70, min_rpd=2.0
        ("soil", 0.85, 3.5, "excellent"),
        ("soil", 0.72, 2.1, "good"),
        # soil fair band: r2>=0.50 and rpd>=0.5 -> fair
        ("soil", 0.55, 1.0, "fair"),
        # food_moisture: min_r2=0.90, min_rpd=4.0; excellent needs
        # r2>=1.00 and rpd>=5.0 which is unreachable here, so good.
        ("food_moisture", 0.95, 5.5, "good"),
        ("food_moisture", 0.91, 4.2, "good"),
    ],
)
def test_grade_matrix(domain, r2, rpd, expected_grade):
    res = evaluate_quality(_metrics(r2, rpd), domain=domain)
    assert res["grade"] == expected_grade


def test_action_proceed_for_passed():
    res = evaluate_quality(_metrics(0.95, 4.5), domain="default")
    assert res["action"] == "proceed"


def test_action_retry_for_fair():
    res = evaluate_quality(_metrics(0.65, 1.8), domain="default")
    assert res["action"] == "retry_preprocessing"


def test_action_investigate_for_poor():
    res = evaluate_quality(_metrics(0.40, 1.0), domain="default")
    assert res["action"] == "investigate_data"


def test_action_retry_on_overfitting_even_if_grade_good():
    res = evaluate_quality(_metrics(0.85, 3.2, rmsep=0.5, rmsecv=0.1), domain="default")
    assert res["details"]["overfitting_risk"] is True
    assert res["action"] == "retry_preprocessing"


def test_small_sample_does_not_relax_production_thresholds():
    res = evaluate_quality(_metrics(0.72, 2.6), domain="default", n_samples=50)
    assert res["grade"] == "fair"
    assert res["passed"] is False
    assert res["thresholds_used"]["small_sample_relaxation"] is False


def test_small_sample_relaxation_requires_explicit_exploratory_opt_in():
    res = evaluate_quality(
        _metrics(0.72, 2.6),
        domain="default",
        n_samples=50,
        allow_small_sample_relaxation=True,
    )
    assert res["grade"] == "good"
    assert res["thresholds_used"]["small_sample_relaxation"] is True


def test_significant_bias_blocks_otherwise_good_model():
    metrics = _metrics(0.95, 4.5, rmsep=0.2, bias_val=0.5)
    metrics["test"].update({"RMSE": 0.2, "n": 100, "reference_std": 1.0})
    res = evaluate_quality(metrics, domain="default")
    assert res["grade"] == "excellent"
    assert res["details"]["bias_issue"] is True
    assert res["passed"] is False
    assert res["action"] == "investigate_data"


def test_recommendation_is_chinese():
    res = evaluate_quality(_metrics(0.95, 4.5), domain="default")
    # Recommendation string should contain CJK characters.
    assert any("\u4e00" <= ch <= "\u9fff" for ch in res["recommendation"])
