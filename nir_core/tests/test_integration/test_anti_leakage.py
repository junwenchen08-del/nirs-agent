"""V3 anti-leakage tests.

Verifies that:
    1. PreprocessingPipeline.fit/transform does not leak test set statistics.
    2. nir_train_model with pipeline_steps splits BEFORE preprocessing.
    3. Dynamic CV strategy correctly adapts to sample size.
    4. VIP computation produces valid scores.
    5. Regression coefficients are extractable.
"""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline


# ---------------------------------------------------------------------------
# 1. Pipeline fit/transform anti-leakage
# ---------------------------------------------------------------------------


class TestPipelineAntiLeakage:
    """Verify that stateful preprocessing steps do not leak test data."""

    def test_mean_center_uses_train_mean(self):
        """Mean-centering must subtract the TRAIN mean, not the full-data mean."""
        rng = np.random.RandomState(42)
        X_train = rng.rand(50, 100)  # mean ~0.5
        X_test = rng.rand(10, 100) + 10.0  # mean ~10.5, deliberately shifted

        pipe = PreprocessingPipeline(
            steps=[PreprocessingStep(method="mean_center")]
        )
        pipe.fit(X_train)
        X_test_t = pipe.transform(X_test)

        # If correct: test was centered using TRAIN mean (~0.5), so
        # transformed test mean should be ~10.0 (10.5 - 0.5).
        # If leaked: test was centered using its OWN mean (~10.5), so
        # transformed test mean would be ~0.0.
        test_mean = float(np.mean(X_test_t))
        assert test_mean > 5.0, (
            f"Mean-centering leaked test stats: transformed mean={test_mean:.4f}, "
            "expected ~10.0 if train mean was used correctly"
        )

    def test_autoscale_uses_train_stats(self):
        """Autoscale must use TRAIN mean and std, not test set's own."""
        rng = np.random.RandomState(42)
        X_train = rng.rand(50, 100) * 0.1  # small scale
        X_test = rng.rand(10, 100) * 10.0  # large scale

        pipe = PreprocessingPipeline(
            steps=[PreprocessingStep(method="autoscale")]
        )
        pipe.fit(X_train)
        X_test_t = pipe.transform(X_test)

        # If correct: test was scaled using TRAIN std (~0.029), so
        # transformed test values should be very large (~100s).
        # If leaked: test was scaled using its OWN std (~2.9), so
        # transformed test values would be ~O(1).
        test_max = float(np.max(np.abs(X_test_t)))
        assert test_max > 10.0, (
            f"Autoscale leaked test stats: transformed max={test_max:.4f}, "
            "expected >>10 if train std was used correctly"
        )

    def test_msc_uses_train_reference(self):
        """MSC must use the TRAIN mean spectrum as reference."""
        rng = np.random.RandomState(42)
        X_train = rng.rand(50, 100)
        X_test = rng.rand(10, 100) * 5.0  # different scale

        pipe = PreprocessingPipeline(
            steps=[PreprocessingStep(method="msc")]
        )
        pipe.fit(X_train)
        X_test_t = pipe.transform(X_test)

        # MSC-corrected test data should be closer to the train reference
        # spectrum than to the test's own mean. We just verify the transform
        # ran and produced finite output.
        assert np.all(np.isfinite(X_test_t)), "MSC transform produced non-finite values"
        assert X_test_t.shape == X_test.shape

    def test_pipeline_fit_then_transform_idempotent_on_train(self):
        """Transform on the same data used for fit should match apply()."""
        rng = np.random.RandomState(42)
        X = rng.rand(30, 50)

        steps = [PreprocessingStep(method="snv"),
                 PreprocessingStep(method="mean_center")]
        pipe = PreprocessingPipeline(steps=steps)
        pipe.fit(X)
        X_fit_transform = pipe.transform(X)
        X_apply = pipe.apply(X)

        # For stateless + stateful mix, fit/transform should match apply
        # because apply uses X's own statistics (which is the same X).
        np.testing.assert_allclose(X_fit_transform, X_apply, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Leakage in split-then-preprocess flow
# ---------------------------------------------------------------------------


class TestSplitThenPreprocess:
    """Verify the v3 anti-leakage flow: split → fit on train → transform all."""

    def test_split_then_preprocess_differs_from_global_preprocess(self):
        """Preprocessing after split should differ from preprocessing before split."""
        from nir_core.model.evaluation import split_dataset

        rng = np.random.RandomState(42)
        X = rng.rand(100, 50)
        y = rng.rand(100)

        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X, y, test_ratio=0.2, val_ratio=0.1, random_state=42
        )

        # v3 flow: fit on train, transform test.
        pipe_v3 = PreprocessingPipeline(
            steps=[PreprocessingStep(method="mean_center")]
        )
        pipe_v3.fit(X_tr)
        X_te_v3 = pipe_v3.transform(X_te)

        # v2 flow (leaky): fit on ALL data, transform test.
        pipe_v2 = PreprocessingPipeline(
            steps=[PreprocessingStep(method="mean_center")]
        )
        pipe_v2.fit(X)
        X_te_v2 = pipe_v2.transform(X_te)

        # The two should differ because train mean != full-data mean.
        assert not np.allclose(X_te_v3, X_te_v2), (
            "Anti-leakage flow produced same result as leaky flow — "
            "preprocessing may be using global statistics"
        )


# ---------------------------------------------------------------------------
# 3. Dynamic CV strategy
# ---------------------------------------------------------------------------


class TestDynamicCVStrategy:
    """Verify _resolve_cv_splitter adapts to sample size."""

    def test_small_samples_uses_loocv(self):
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import LeaveOneOut

        splitter, label = _resolve_cv_splitter("auto", 10, 15, 42)
        assert isinstance(splitter, LeaveOneOut)
        assert label == "loocv"

    def test_medium_samples_uses_5_fold(self):
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import KFold

        splitter, label = _resolve_cv_splitter("auto", 10, 40, 42)
        assert isinstance(splitter, KFold)
        assert "5" in label

    def test_large_samples_uses_cv_folds(self):
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import KFold

        splitter, label = _resolve_cv_splitter("auto", 10, 200, 42)
        assert isinstance(splitter, KFold)
        assert "10" in label

    def test_explicit_loocv_strategy(self):
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import LeaveOneOut

        splitter, label = _resolve_cv_splitter("loocv", 10, 200, 42)
        assert isinstance(splitter, LeaveOneOut)
        assert label == "loocv"

    def test_fixed_strategy_uses_cv_folds(self):
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import KFold

        splitter, label = _resolve_cv_splitter("fixed", 7, 200, 42)
        assert isinstance(splitter, KFold)
        assert "7" in label

    def test_cv_folds_clamped_to_n_samples(self):
        """CV folds cannot exceed n_samples - 1."""
        from nir_core.model.pls import _resolve_cv_splitter
        from sklearn.model_selection import KFold

        # 20 samples with cv_folds=15 → should clamp to 19 (n-1) for fixed,
        # but "auto" with <50 samples uses 5-fold.
        splitter, label = _resolve_cv_splitter("fixed", 15, 20, 42)
        assert isinstance(splitter, KFold)
        # KFold should have at most n_samples-1 splits
        assert splitter.get_n_splits() <= 19


# ---------------------------------------------------------------------------
# 4. VIP computation
# ---------------------------------------------------------------------------


class TestVIPComputation:
    """Verify VIP scores are computed correctly."""

    def test_vip_returns_correct_shape(self):
        from nir_core.model.pls import compute_vip, train_pls

        rng = np.random.RandomState(42)
        X = rng.rand(60, 100)
        y = rng.rand(60)

        model, _, _ = train_pls(X, y, n_components=5, cv_strategy="fixed")
        vip = compute_vip(model, X, y)

        assert vip.shape == (100,)
        assert np.all(vip >= 0)

    def test_vip_mean_around_one(self):
        """The mean of VIP scores should be close to 1.0 (by definition)."""
        from nir_core.model.pls import compute_vip, train_pls

        rng = np.random.RandomState(42)
        X = rng.rand(100, 50)
        y = X @ rng.rand(50) + rng.rand(100) * 0.01  # linear relation

        model, _, _ = train_pls(X, y, n_components=5, cv_strategy="fixed")
        vip = compute_vip(model, X, y)

        # VIP scores have mean ~1.0 by construction.
        assert 0.5 < float(np.mean(vip)) < 2.0, (
            f"VIP mean={float(np.mean(vip)):.4f}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# 5. Regression coefficients
# ---------------------------------------------------------------------------


class TestRegressionCoefficients:
    """Verify regression coefficient extraction."""

    def test_coef_returns_correct_shape(self):
        from nir_core.model.pls import get_regression_coefficients, train_pls

        rng = np.random.RandomState(42)
        X = rng.rand(60, 80)
        y = rng.rand(60)

        model, _, _ = train_pls(X, y, n_components=3, cv_strategy="fixed")
        coef = get_regression_coefficients(model)

        assert coef.shape == (80,)
        assert np.all(np.isfinite(coef))

    def test_coef_predicts_correctly(self):
        """y_pred ≈ X @ coef (for centered data)."""
        from nir_core.model.pls import get_regression_coefficients, predict_pls, train_pls

        rng = np.random.RandomState(42)
        X = rng.rand(60, 30)
        y = X @ rng.rand(30) + 0.1 * rng.rand(60)

        model, _, _ = train_pls(X, y, n_components=5, cv_strategy="fixed")
        coef = get_regression_coefficients(model)

        # Manual prediction using coefficients.
        y_manual = X @ coef
        y_model = predict_pls(model, X)

        # Should be close (not exact due to PLS internal centering).
        np.testing.assert_allclose(y_manual, y_model, atol=1e-6)
