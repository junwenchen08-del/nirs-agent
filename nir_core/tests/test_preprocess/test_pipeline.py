"""Tests for :mod:`nir_core.preprocess.pipeline`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import (
    DEFAULT_CANDIDATE_PIPELINES,
    PRESTEP_METHODS,
    PreprocessingPipeline,
    validate_pipeline,
)
from nir_core.preprocess.scaling import mean_center
from nir_core.preprocess.scatter import snv
from nir_core.preprocess.smoothing import sg_smooth


def test_pipeline_apply_executes_steps_in_order(rng):
    """Pipeline.apply should apply steps in declared order with the right output."""
    X = rng.normal(1.0, 0.3, size=(10, 100))
    steps = [
        PreprocessingStep(method="snv", params={}),
        PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
        PreprocessingStep(method="mean_center", params={}),
    ]
    pipe = PreprocessingPipeline(steps=steps)
    out = pipe.apply(X)

    # Manual sequential application.
    expected = snv(X)
    expected = sg_smooth(expected, window=11, order=2)
    expected = mean_center(expected)
    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_pipeline_apply_with_wv_forwarded_to_detrend():
    """When wv is provided and a detrend step has no wv param, wv is forwarded."""
    n = 100
    x = np.linspace(0, 10, n)
    # Pure quadratic trend (no noise) so detrend residual should be ~0.
    X = np.tile(0.5 * x + 0.02 * x**2, (3, 1))
    steps = [
        PreprocessingStep(method="detrend", params={}),
    ]
    pipe = PreprocessingPipeline(steps=steps)
    out = pipe.apply(X, wv=x)
    # Quadratic trend should be removed.
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_pipeline_apply_detrend_step_wv_in_params_takes_precedence(rng):
    """If the step params already contain wv, the caller's wv is NOT used."""
    n = 80
    x_caller = np.linspace(0, 10, n)
    x_step = np.linspace(0, 1, n)  # different scale
    X = rng.normal(0, 0.01, size=(2, n)) + x_caller
    steps = [PreprocessingStep(method="detrend", params={"wv": x_step})]
    pipe = PreprocessingPipeline(steps=steps)
    out_with_caller = pipe.apply(X, wv=x_caller)
    out_without = pipe.apply(X, wv=None)
    # Both should use x_step (params take precedence).
    np.testing.assert_allclose(out_with_caller, out_without, rtol=1e-12)


def test_pipeline_description_chinese():
    steps = [
        PreprocessingStep(method="snv", params={}),
        PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
        PreprocessingStep(method="mean_center", params={}),
    ]
    pipe = PreprocessingPipeline(steps=steps)
    desc = pipe.description(locale="zh")
    assert "SNV" in desc
    assert "SG平滑" in desc
    assert "window=11" in desc
    assert "order=2" in desc
    assert "均值中心化" in desc
    assert "→" in desc


def test_pipeline_description_unknown_method_falls_back_to_name():
    steps = [PreprocessingStep(method="custom_method", params={})]
    pipe = PreprocessingPipeline(steps=steps)
    desc = pipe.description()
    assert "custom_method" in desc


def test_default_candidate_pipelines_are_executable(rng):
    """Each DEFAULT_CANDIDATE_PIPELINES entry should run on synthetic data."""
    X = rng.normal(1.0, 0.2, size=(8, 200))
    wv = np.linspace(1100, 2500, 200)
    for i, pipe in enumerate(DEFAULT_CANDIDATE_PIPELINES):
        out = pipe.apply(X, wv=wv)
        assert out.shape == X.shape, f"Pipeline {i} changed shape"
        assert not np.any(np.isnan(out)), f"Pipeline {i} produced NaN"
        assert not np.any(np.isinf(out)), f"Pipeline {i} produced Inf"


def test_default_candidate_pipelines_count():
    assert len(DEFAULT_CANDIDATE_PIPELINES) == 3


def test_default_candidate_pipelines_descriptions_nonempty():
    for pipe in DEFAULT_CANDIDATE_PIPELINES:
        desc = pipe.description()
        assert isinstance(desc, str)
        assert len(desc) > 0


def test_pipeline_unknown_method_raises_keyerror(rng):
    X = rng.normal(size=(3, 50))
    steps = [PreprocessingStep(method="does_not_exist", params={})]
    pipe = PreprocessingPipeline(steps=steps)
    with pytest.raises(KeyError):
        pipe.apply(X)


def test_pipeline_does_not_mutate_input(rng):
    X = rng.normal(1.0, 0.2, size=(5, 100))
    X_copy = X.copy()
    steps = [
        PreprocessingStep(method="snv", params={}),
        PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
    ]
    pipe = PreprocessingPipeline(steps=steps)
    _ = pipe.apply(X)
    np.testing.assert_array_equal(X, X_copy)


def test_prestep_methods_mapping_complete():
    """PRESTEP_METHODS should contain all expected method keys."""
    expected = {
        "snv",
        "robust_snv",
        "msc",
        "emsc",
        "despike",
        "sg_smooth",
        "derivative1",
        "derivative2",
        "norris_derivative1",
        "norris_derivative2",
        "airpls",
        "asls",
        "detrend",
        "mean_center",
        "autoscale",
        "normalize",
    }
    assert expected.issubset(set(PRESTEP_METHODS.keys()))


def test_pipeline_derivative_steps_callable(rng):
    """derivative1 and derivative2 dispatchers should produce expected shapes."""
    X = rng.normal(size=(4, 100))
    steps1 = [
        PreprocessingStep(method="derivative1", params={"window": 11, "order": 2})
    ]
    steps2 = [
        PreprocessingStep(method="derivative2", params={"window": 11, "order": 3})
    ]
    out1 = PreprocessingPipeline(steps1).apply(X)
    out2 = PreprocessingPipeline(steps2).apply(X)
    assert out1.shape == X.shape
    assert out2.shape == X.shape


# ---------------------------------------------------------------------------
# validate_pipeline tests
# ---------------------------------------------------------------------------
class TestValidatePipeline:
    """Tests for validate_pipeline (V3.6 flexible guardrail)."""

    def test_valid_pipeline_passes(self):
        """A well-ordered pipeline with valid params passes validation."""
        steps = [
            PreprocessingStep(method="airpls", params={"lambda_": 1e6}),
            PreprocessingStep(method="snv", params={}),
            PreprocessingStep(method="sg_smooth", params={"window": 11, "order": 2}),
            PreprocessingStep(method="mean_center", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is True
        assert reason == ""

    def test_empty_steps_passes(self):
        """An empty pipeline is valid (no preprocessing)."""
        is_valid, _reason = validate_pipeline([])
        assert is_valid is True

    def test_unknown_method_fails(self):
        """A method not in PRESTEP_METHODS fails."""
        steps = [PreprocessingStep(method="unknown_method", params={})]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "未知预处理方法" in reason

    def test_snv_and_msc_mutually_exclusive(self):
        """SNV and MSC cannot coexist."""
        steps = [
            PreprocessingStep(method="snv", params={}),
            PreprocessingStep(method="msc", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "互斥方法" in reason

    def test_all_scatter_corrections_are_mutually_exclusive(self):
        steps = [
            PreprocessingStep(method="robust_snv", params={}),
            PreprocessingStep(method="emsc", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "互斥方法" in reason

    def test_derivative1_and_derivative2_mutually_exclusive(self):
        """derivative1 and derivative2 cannot coexist."""
        steps = [
            PreprocessingStep(method="derivative1", params={"window": 11, "order": 2}),
            PreprocessingStep(method="derivative2", params={"window": 11, "order": 3}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "互斥方法" in reason

    def test_sg_and_norris_derivatives_are_mutually_exclusive(self):
        steps = [
            PreprocessingStep(method="derivative1", params={"window": 11, "order": 2}),
            PreprocessingStep(
                method="norris_derivative1", params={"gap": 3, "segment": 5}
            ),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "互斥方法" in reason

    def test_despike_must_precede_scatter_correction(self):
        steps = [
            PreprocessingStep(method="snv", params={}),
            PreprocessingStep(method="despike", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "顺序不当" in reason

    def test_scaling_methods_mutually_exclusive(self):
        """mean_center, autoscale, and normalize cannot coexist."""
        steps = [
            PreprocessingStep(method="mean_center", params={}),
            PreprocessingStep(method="autoscale", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "互斥方法" in reason

    def test_wrong_order_fails(self):
        """Scaling before scatter correction fails (wrong stage order)."""
        steps = [
            PreprocessingStep(method="mean_center", params={}),
            PreprocessingStep(method="snv", params={}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "顺序不当" in reason

    def test_sg_window_even_fails(self):
        """SG window must be odd."""
        steps = [
            PreprocessingStep(method="sg_smooth", params={"window": 10, "order": 2}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "奇数" in reason

    def test_sg_window_out_of_range_fails(self):
        """SG window outside 5-15 fails."""
        steps = [
            PreprocessingStep(method="sg_smooth", params={"window": 3, "order": 2}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "超出范围" in reason

    def test_airpls_lambda_out_of_range_fails(self):
        """airPLS lambda outside 1e3-1e8 fails."""
        steps = [
            PreprocessingStep(method="airpls", params={"lambda_": 1e10}),
        ]
        is_valid, reason = validate_pipeline(steps)
        assert is_valid is False
        assert "超出范围" in reason

    def test_valid_default_candidates_pass(self):
        """All DEFAULT_CANDIDATE_PIPELINES should pass validation."""
        for pipe in DEFAULT_CANDIDATE_PIPELINES:
            is_valid, reason = validate_pipeline(pipe.steps)
            assert is_valid, f"Default candidate failed: {reason}"

    def test_same_stage_methods_allowed(self):
        """Multiple methods in the same stage are allowed if not exclusive."""
        steps = [
            PreprocessingStep(method="airpls", params={"lambda_": 1e6}),
            PreprocessingStep(method="detrend", params={}),
            PreprocessingStep(method="snv", params={}),
        ]
        is_valid, _reason = validate_pipeline(steps)
        assert is_valid is True
