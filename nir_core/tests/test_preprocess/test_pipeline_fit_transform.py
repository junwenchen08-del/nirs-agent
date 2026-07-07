"""Tests for the leakage-safe fit/transform protocol of PreprocessingPipeline.

These tests verify that stateful steps (mean_center, autoscale, MSC) compute
their statistics from *training* data only and reuse them when transforming
validation / test data -- the property that prevents preprocessing leakage in
nested cross-validation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline


def test_mean_center_fit_transform_uses_train_statistics():
    """transform() must subtract the *train* mean, not the input's own mean."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(5.0, 1.0, size=(50, 20))  # mean ~5
    X_val = rng.normal(10.0, 1.0, size=(10, 20))   # mean ~10

    p = PreprocessingPipeline(
        [PreprocessingStep(method="mean_center")]
    ).fit(X_train)

    out = p.transform(X_val)
    # If val's own mean were used, out.mean() ~ 0. With train mean (~5),
    # out.mean() ~ 10 - 5 = 5.
    assert out.mean(axis=0).mean() == pytest.approx(5.0, abs=0.5)
    # And NOT close to 0 (which would indicate val-mean leakage).
    assert abs(out.mean()) > 2.0


def test_autoscale_fit_transform_uses_train_stats():
    """transform() must use train mean AND train std."""
    rng = np.random.default_rng(1)
    X_train = rng.normal(0.0, 2.0, size=(60, 15))  # std ~2
    X_val = rng.normal(0.0, 5.0, size=(10, 15))    # std ~5

    p = PreprocessingPipeline(
        [PreprocessingStep(method="autoscale")]
    ).fit(X_train)

    out = p.transform(X_val)
    # With train std (~2), val (std~5) scaled by /2 -> std ~2.5, not 1.
    val_std = out.std(axis=0, ddof=0).mean()
    assert val_std == pytest.approx(2.5, abs=0.6)
    # If val's own std were used, val_std would be ~1.0 (leakage).
    assert val_std > 1.5


def test_msc_fit_transform_uses_train_reference():
    """transform() must use the train mean spectrum as MSC reference."""
    rng = np.random.default_rng(2)
    # Train and val have very different mean spectra.
    wv = np.arange(15)
    base_train = np.linspace(0, 1, 15)
    base_val = np.linspace(1, 0, 15)  # opposite trend
    X_train = base_train[None, :] + rng.normal(0, 0.01, (40, 15))
    X_val = base_val[None, :] + rng.normal(0, 0.01, (8, 15))

    p = PreprocessingPipeline(
        [PreprocessingStep(method="msc")]
    ).fit(X_train)

    out_val = p.transform(X_val)
    # Compare against a manual MSC using the TRAIN reference.
    ref = X_train.mean(axis=0)
    A = np.column_stack([np.ones(15), ref])
    coeffs, *_ = np.linalg.lstsq(A, X_val.T, rcond=None)
    a = coeffs[0:1, :]
    b = coeffs[1:2, :]
    expected = (X_val.T - a) / np.where(b == 0, 1.0, b)
    expected = expected.T
    np.testing.assert_allclose(out_val, expected, atol=1e-6)


def test_transform_without_fit_warns_and_falls_back():
    """transform() on an unfitted stateful pipeline warns and uses apply()."""
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (20, 10))
    p = PreprocessingPipeline(
        [PreprocessingStep(method="mean_center")]
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = p.transform(X)
        assert any(issubclass(rec.category, RuntimeWarning) for rec in w)
    # Fallback to apply(): subtracts X's own mean -> ~0.
    assert out.mean() == pytest.approx(0.0, abs=1e-9)


def test_apply_unchanged_backward_compatible():
    """apply() keeps the original stateless behaviour (X's own stats)."""
    rng = np.random.default_rng(4)
    X = rng.normal(3.0, 1.0, (30, 12))
    p = PreprocessingPipeline(
        [PreprocessingStep(method="mean_center")]
    )
    out = p.apply(X)
    # apply uses X's own mean -> centered to ~0.
    assert out.mean() == pytest.approx(0.0, abs=1e-9)


def test_apply_remains_stateless_after_fit():
    """apply() keeps stateless behaviour even after fit() (backward compat).

    Callers wanting fitted behaviour must use transform(); apply() always
    uses the input's own statistics so existing one-off usage is unchanged.
    """
    rng = np.random.default_rng(5)
    X_train = rng.normal(5.0, 1.0, (40, 10))
    X_val = rng.normal(8.0, 1.0, (6, 10))
    p = PreprocessingPipeline(
        [PreprocessingStep(method="mean_center")]
    ).fit(X_train)
    # apply() is stateless -> uses val's own mean -> centered to ~0.
    out_apply = p.apply(X_val)
    assert out_apply.mean() == pytest.approx(0.0, abs=1e-9)
    # transform() uses train mean -> val (8) - train (5) ~ 3.
    out_transform = p.transform(X_val)
    assert abs(out_transform.mean()) > 2.0


def test_mixed_stateful_stateless_pipeline():
    """A pipeline with both stateful and stateless steps fits correctly."""
    rng = np.random.default_rng(6)
    X_train = rng.normal(2.0, 3.0, (50, 25))
    X_val = rng.normal(4.0, 3.0, (8, 25))

    p = PreprocessingPipeline(
        [
            PreprocessingStep(method="snv"),  # stateless
            PreprocessingStep(method="sg_smooth", params={"window": 5, "order": 2}),  # stateless
            PreprocessingStep(method="mean_center"),  # stateful
        ]
    ).fit(X_train)

    out = p.transform(X_val)
    assert out.shape == X_val.shape
    assert np.isfinite(out).all()
    # mean_center used train stats: val (mean~4 after snv~0) minus train mean.
    # Just verify no crash and finite output; exact value depends on snv+sg.


def test_fitted_flag():
    p = PreprocessingPipeline(
        [PreprocessingStep(method="mean_center")]
    )
    assert not p.fitted()
    p.fit(np.ones((10, 5)))
    assert p.fitted()


def test_nested_cv_no_inner_leakage_signal():
    """Sanity check: nested_cv_preprocessing runs and returns a pipeline.

    This is a smoke test that the refit-per-fold inner CV does not raise and
    produces finite scores (the rigorous leakage proof is structural: each
    fold refits the pipeline on inner-train only).
    """
    from nir_core.tests.generators import generate_synthetic_spectra
    from nir_core.model.evaluation import nested_cv_preprocessing, split_dataset

    data = generate_synthetic_spectra(
        n_samples=120, n_wavelengths=80, n_components=2, random_state=42
    )
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
        data.X, data.y, test_ratio=0.2, val_ratio=0.15, random_state=42
    )
    best_pipe, results = nested_cv_preprocessing(
        X_tr, y_tr, X_val, y_val, inner_folds=3, max_components=5,
        random_state=42,
    )
    assert best_pipe is not None
    assert "best" in results
    cand = results["candidates"]
    assert len(cand) >= 1
    for desc, info in cand.items():
        assert np.isfinite(info["score"])
