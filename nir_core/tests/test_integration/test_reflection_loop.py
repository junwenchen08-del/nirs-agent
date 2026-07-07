"""Integration tests for the Phase 3 reflection loop.

These tests exercise the deterministic reflection machinery
(``should_retry``, ``get_next_pipeline``, ``suggest_lv_adjustment``)
in realistic end-to-end scenarios, not just unit-level edge cases.

Coverage:
    1. A high-quality model → no retry needed.
    2. A low-quality model → retry triggered, next pipeline returned.
    3. Plateau detection → retry stops after two non-improving attempts.
    4. Overfitting → LV reduction suggested.
    5. Full multi-attempt reflection cycle with history tracking.
"""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.config import get_nir_config, set_nir_config
from nir_core.models import PreprocessingStep
from nir_core.utils.metrics import evaluate_quality
from nir_core.utils.validation import (
    get_next_pipeline,
    should_retry,
    suggest_lv_adjustment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure a fresh default config for each test."""
    set_nir_config(get_nir_config().__class__())
    yield
    set_nir_config(get_nir_config().__class__())


def _metrics(r2_val: float, rpd: float, n_comp: int = 5,
             rmsep: float = 0.5, rmsecv: float = 0.4,
             bias: float = 0.01, n_samples: int = 200,
             domain: str = "default") -> dict:
    """Build a realistic metrics dict for testing."""
    return {
        "method": "pls",
        "n_components": n_comp,
        "domain": domain,
        "n_samples": n_samples,
        "R2_val": r2_val,
        "RPD": rpd,
        "RMSEP": rmsep,
        "RMSECV": rmsecv,
        "RMSEC": rmsep * 0.5,
        "test": {"RMSE": rmsep, "R2": r2_val, "RPD": rpd, "bias": bias},
        "val": {"RMSE": rmsecv, "R2": r2_val, "RPD": rpd},
        "train": {"RMSE": rmsep * 0.5, "R2": r2_val + 0.05, "RPD": rpd + 0.5},
    }


# ---------------------------------------------------------------------------
# 1. High-quality model → no retry
# ---------------------------------------------------------------------------


class TestHighQualityNoRetry:
    def test_passed_model_does_not_retry(self):
        metrics = _metrics(r2_val=0.95, rpd=5.0)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        assert quality["passed"] is True
        assert should_retry(quality, attempt=1, max_retries=3) is False

    def test_reflect_returns_no_next_pipeline(self):
        metrics = _metrics(r2_val=0.95, rpd=5.0)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        result = should_retry(quality, attempt=1, max_retries=3, history=[])
        assert result is False


# ---------------------------------------------------------------------------
# 2. Low-quality model → retry triggered
# ---------------------------------------------------------------------------


class TestLowQualityTriggersRetry:
    def test_poor_model_triggers_retry(self):
        metrics = _metrics(r2_val=0.50, rpd=1.2)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        assert quality["passed"] is False
        assert should_retry(quality, attempt=1, max_retries=3) is True

    def test_next_pipeline_returned_on_first_attempt(self):
        history = []
        nxt = get_next_pipeline(attempt=1, history=history)
        assert nxt is not None
        assert isinstance(nxt, list)
        assert all(isinstance(s, PreprocessingStep) for s in nxt)

    def test_next_pipeline_excludes_tried(self):
        cfg = get_nir_config()
        first = get_next_pipeline(attempt=1, history=[])
        assert first is not None
        history = [{"pipeline": [{"method": s.method} for s in first], "metrics": {}}]
        second = get_next_pipeline(attempt=2, history=history)
        if second is not None:
            first_methods = {s.method for s in first}
            second_methods = {s.method for s in second}
            assert first_methods != second_methods, (
                "get_next_pipeline should exclude already-tried pipelines"
            )

    def test_next_pipeline_returns_none_when_all_tried(self):
        cfg = get_nir_config()
        history = []
        for _ in cfg.candidate_pipelines:
            nxt = get_next_pipeline(attempt=1, history=history)
            if nxt is None:
                break
            history.append({"pipeline": [{"method": s.method} for s in nxt], "metrics": {}})
        assert get_next_pipeline(attempt=1, history=history) is None


# ---------------------------------------------------------------------------
# 3. Plateau detection
# ---------------------------------------------------------------------------


class TestPlateauDetection:
    def test_plateau_stops_retry(self):
        """Two consecutive attempts with R² change < 0.02 → stop."""
        history = [
            {"pipeline": [{"method": "snv"}], "metrics": {"R2_val": 0.72, "RPD": 2.1}},
            {"pipeline": [{"method": "msc"}], "metrics": {"R2_val": 0.73, "RPD": 2.15}},
        ]
        metrics = _metrics(r2_val=0.73, rpd=2.15)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        # Should stop because the last two differ by < 0.02.
        assert should_retry(quality, attempt=2, max_retries=5, history=history) is False

    def test_improving_model_continues_retry(self):
        """R² change >= 0.02 → continue retrying."""
        history = [
            {"pipeline": [{"method": "snv"}], "metrics": {"R2_val": 0.60, "RPD": 1.8}},
            {"pipeline": [{"method": "msc"}], "metrics": {"R2_val": 0.75, "RPD": 2.2}},
        ]
        metrics = _metrics(r2_val=0.75, rpd=2.2)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        assert should_retry(quality, attempt=2, max_retries=5, history=history) is True

    def test_max_retries_stops_loop(self):
        metrics = _metrics(r2_val=0.50, rpd=1.2)
        quality = evaluate_quality(metrics, domain="default", n_samples=200)
        assert should_retry(quality, attempt=3, max_retries=3) is False


# ---------------------------------------------------------------------------
# 4. Overfitting → LV reduction
# ---------------------------------------------------------------------------


class TestSmartLVAdjustment:
    def test_overfitting_suggests_reduction(self):
        """RMSEP > 2×RMSECV → reduce n_components."""
        metrics = _metrics(r2_val=0.88, rpd=3.2, n_comp=8, rmsep=1.0, rmsecv=0.3)
        result = suggest_lv_adjustment(metrics, domain="default", n_samples=200)
        assert "overfitting" in result["issues"]
        assert result["change"] == "reduce"
        assert result["suggested_n"] < metrics["n_components"]
        assert result["delta"] < 0

    def test_healthy_model_no_change(self):
        metrics = _metrics(r2_val=0.95, rpd=5.0, n_comp=5, rmsep=0.3, rmsecv=0.25)
        result = suggest_lv_adjustment(metrics, domain="default", n_samples=200)
        assert result["change"] == "none"
        assert result["suggested_n"] == metrics["n_components"]

    def test_underfitting_suggests_increase(self):
        metrics = _metrics(r2_val=0.55, rpd=1.3, n_comp=3, rmsep=0.8, rmsecv=0.7)
        result = suggest_lv_adjustment(metrics, domain="default", n_samples=200)
        assert "underfitting" in result["issues"]
        assert result["change"] == "increase"
        assert result["suggested_n"] > metrics["n_components"]

    def test_bias_issue_flagged(self):
        metrics = _metrics(
            r2_val=0.85, rpd=3.0, n_comp=5, rmsep=0.4, rmsecv=0.35, bias=0.5,
        )
        result = suggest_lv_adjustment(metrics, domain="default", n_samples=200)
        assert "bias_issue" in result["issues"]

    def test_retry_preprocessing_flagged_on_overfitting(self):
        metrics = _metrics(r2_val=0.85, rpd=3.0, n_comp=8, rmsep=1.0, rmsecv=0.3)
        result = suggest_lv_adjustment(metrics, domain="default", n_samples=200)
        assert result["retry_preprocessing"] is True


# ---------------------------------------------------------------------------
# 5. Full multi-attempt reflection cycle
# ---------------------------------------------------------------------------


class TestFullReflectionCycle:
    def test_three_attempt_cycle_then_stop(self):
        """Simulate a full cycle: 3 poor attempts → stop at max_retries."""
        history = []
        for attempt in range(1, 4):
            metrics = _metrics(r2_val=0.50 + attempt * 0.01, rpd=1.2 + attempt * 0.05)
            quality = evaluate_quality(metrics, domain="default", n_samples=200)
            retry = should_retry(quality, attempt=attempt, max_retries=3, history=history)
            if attempt < 3:
                assert retry is True, f"Should retry at attempt {attempt}"
            else:
                assert retry is False, "Should stop at max_retries"
            history.append({
                "pipeline": [{"method": f"method_{attempt}"}],
                "metrics": metrics,
            })

    def test_cycle_stops_on_pass(self):
        """Model passes on attempt 2 → no further retry."""
        history = []
        # Attempt 1: poor
        m1 = _metrics(r2_val=0.50, rpd=1.2)
        q1 = evaluate_quality(m1, domain="default", n_samples=200)
        assert should_retry(q1, attempt=1, max_retries=3, history=history) is True
        history.append({"pipeline": [{"method": "raw"}], "metrics": m1})

        # Attempt 2: excellent
        m2 = _metrics(r2_val=0.95, rpd=5.0)
        q2 = evaluate_quality(m2, domain="default", n_samples=200)
        assert q2["passed"] is True
        assert should_retry(q2, attempt=2, max_retries=3, history=history) is False

    def test_next_pipeline_progresses_through_candidates(self):
        """Each retry should yield a different pipeline until exhausted."""
        cfg = get_nir_config()
        history = []
        pipelines_tried = []
        for _ in range(len(cfg.candidate_pipelines)):
            nxt = get_next_pipeline(attempt=1, history=history)
            if nxt is None:
                break
            methods = tuple(s.method for s in nxt)
            pipelines_tried.append(methods)
            history.append({
                "pipeline": [{"method": s.method} for s in nxt],
                "metrics": {"R2_val": 0.5, "RPD": 1.5},
            })
        # All tried pipelines should be unique.
        assert len(pipelines_tried) == len(set(pipelines_tried))
        # After exhausting, get_next_pipeline returns None.
        assert get_next_pipeline(attempt=1, history=history) is None
