"""Tests for extended multiplicative scatter correction (EMSC)."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline
from nir_core.preprocess.scatter import emsc


def _reference(wv: np.ndarray) -> np.ndarray:
    return (
        0.8 * np.exp(-(((wv - 1200.0) / 45.0) ** 2))
        + 0.5 * np.exp(-(((wv - 1650.0) / 70.0) ** 2))
        + 0.2
    )


def test_emsc_removes_additive_multiplicative_and_quadratic_effects():
    wv = np.linspace(900.0, 1800.0, 240)
    ref = _reference(wv)
    t = (wv - wv.mean()) / wv.std()
    X = np.vstack(
        [
            0.3 + 1.4 * ref + 0.08 * t - 0.04 * t**2,
            -0.2 + 0.7 * ref - 0.05 * t + 0.02 * t**2,
        ]
    )

    corrected = emsc(X, reference=ref, wv=wv, polynomial_order=2)

    np.testing.assert_allclose(corrected, np.tile(ref, (2, 1)), atol=1e-10)


def test_emsc_pipeline_uses_training_reference_for_validation(rng):
    wv = np.linspace(1000.0, 2000.0, 120)
    ref = _reference(wv)
    X_train = np.vstack(
        [
            0.1 + scale * ref + rng.normal(0, 1e-4, ref.size)
            for scale in (0.8, 1.0, 1.2, 1.4)
        ]
    )
    X_val = np.vstack([0.5 + 0.6 * ref, -0.4 + 1.8 * ref])
    pipe = PreprocessingPipeline(
        [PreprocessingStep(method="emsc", params={"polynomial_order": 1})]
    ).fit(X_train, wv)

    transformed = pipe.transform(X_val, wv)
    expected = emsc(
        X_val,
        reference=X_train.mean(axis=0),
        wv=wv,
        polynomial_order=1,
    )
    np.testing.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)


def test_emsc_pipeline_rejects_prediction_axis_mismatch(rng):
    wv = np.linspace(1000.0, 2000.0, 120)
    ref = _reference(wv)
    X = np.vstack(
        [0.1 + scale * ref + rng.normal(0, 1e-4, ref.size) for scale in (0.8, 1.0, 1.2)]
    )
    pipe = PreprocessingPipeline([PreprocessingStep(method="emsc")]).fit(X, wv)

    with pytest.raises(ValueError, match="does not match"):
        pipe.transform(X, wv + 0.01)


def test_emsc_accepts_1d_and_does_not_mutate():
    wv = np.linspace(900.0, 1700.0, 100)
    ref = _reference(wv)
    x = 0.2 + 1.3 * ref
    original = x.copy()
    out = emsc(x, reference=ref, wv=wv)
    assert out.shape == x.shape
    np.testing.assert_array_equal(x, original)


def test_emsc_rejects_axis_mismatch():
    X = np.ones((2, 20))
    with pytest.raises(ValueError, match="wv length"):
        emsc(X, reference=np.ones(20), wv=np.arange(19))


@pytest.mark.parametrize("polynomial_order", [-1, 4, 1.5])
def test_emsc_rejects_invalid_polynomial_order(polynomial_order):
    with pytest.raises(ValueError, match="polynomial_order"):
        emsc(
            np.ones((2, 20)),
            reference=np.linspace(0, 1, 20),
            polynomial_order=polynomial_order,
        )


def test_emsc_rejects_near_zero_multiplicative_coefficient():
    ref = np.sin(np.linspace(0.0, 4.0, 100))
    constant = np.ones((1, 100))
    with pytest.raises(ValueError, match="multiplicative"):
        emsc(constant, reference=ref)
