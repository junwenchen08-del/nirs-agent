"""Tests for should_retry and get_next_pipeline in nir_core.utils.validation."""

from __future__ import annotations

import pytest

from nir_core.models import PreprocessingStep
from nir_core.utils.validation import get_next_pipeline, should_retry


def test_should_retry_false_when_passed():
    res = {"passed": True}
    assert should_retry(res, attempt=1, max_retries=3) is False


def test_should_retry_true_when_not_passed_and_under_budget():
    res = {"passed": False}
    assert should_retry(res, attempt=1, max_retries=3) is True


def test_should_retry_false_when_budget_exhausted():
    res = {"passed": False}
    assert should_retry(res, attempt=3, max_retries=3) is False
    assert should_retry(res, attempt=4, max_retries=3) is False


def test_should_retry_false_on_plateau():
    res = {"passed": False}
    history = [
        {"metrics": {"R2_val": 0.80}},
        {"metrics": {"R2_val": 0.81}},  # diff 0.01 < 0.02
    ]
    assert should_retry(res, attempt=2, max_retries=5, history=history) is False


def test_should_retry_true_when_improving():
    res = {"passed": False}
    history = [
        {"metrics": {"R2_val": 0.70}},
        {"metrics": {"R2_val": 0.85}},  # diff 0.15 >= 0.02
    ]
    assert should_retry(res, attempt=2, max_retries=5, history=history) is True


def test_should_retry_uses_r2_key_fallback():
    res = {"passed": False}
    history = [
        {"metrics": {"R2": 0.80}},
        {"metrics": {"R2": 0.81}},
    ]
    assert should_retry(res, attempt=2, max_retries=5, history=history) is False


def test_get_next_pipeline_returns_first_untried():
    steps = get_next_pipeline(attempt=1, history=None)
    assert steps is not None
    assert all(isinstance(s, PreprocessingStep) for s in steps)
    # Default first candidate is ["snv","sg_smooth","mean_center"].
    assert [s.method for s in steps] == ["snv", "sg_smooth", "mean_center"]


def test_get_next_pipeline_skips_tried():
    history = [
        {"pipeline": [{"method": "snv"}, {"method": "sg_smooth"},
                      {"method": "mean_center"}]},
    ]
    steps = get_next_pipeline(attempt=2, history=history)
    assert steps is not None
    # Second default candidate.
    assert [s.method for s in steps] == ["msc", "derivative1", "autoscale"]


def test_get_next_pipeline_returns_none_when_all_tried():
    history = [
        {"pipeline": [{"method": "snv"}, {"method": "sg_smooth"},
                      {"method": "mean_center"}]},
        {"pipeline": [{"method": "msc"}, {"method": "derivative1"},
                      {"method": "autoscale"}]},
        {"pipeline": [{"method": "airpls"}, {"method": "sg_smooth"},
                      {"method": "snv"}, {"method": "mean_center"}]},
    ]
    steps = get_next_pipeline(attempt=4, history=history)
    assert steps is None


def test_get_next_pipeline_accepts_preprocessing_step_objects():
    history = [
        {"pipeline": [PreprocessingStep(method="snv"),
                      PreprocessingStep(method="sg_smooth"),
                      PreprocessingStep(method="mean_center")]},
    ]
    steps = get_next_pipeline(attempt=2, history=history)
    assert steps is not None
    assert [s.method for s in steps] == ["msc", "derivative1", "autoscale"]
