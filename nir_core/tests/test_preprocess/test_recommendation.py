"""Deterministic, bounded preprocessing recommendation tests."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.recommendation import recommend_preprocessing


def _clean_spectra(*, n_samples: int = 40, n_wavelengths: int = 101) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, n_wavelengths)
    peak = np.exp(-(((axis - 0.2) / 0.18) ** 2))
    return np.vstack(
        [
            1.0 + (0.9 + index / 200.0) * peak + 0.01 * np.sin(8 * axis + index)
            for index in range(n_samples)
        ]
    )


def test_recommendation_is_deterministic_bounded_and_keeps_raw_baseline():
    X = _clean_spectra()
    first = recommend_preprocessing(X, budget="standard")
    second = recommend_preprocessing(X.copy(), budget="standard")

    assert first.as_dict() == second.as_dict()
    assert first.candidates[0].steps == ()
    assert first.candidates[0].candidate_id == "raw"
    assert len(first.candidates) <= 8
    assert all(len(candidate.steps) <= 3 for candidate in first.candidates)
    assert len({candidate.candidate_id for candidate in first.candidates}) == len(
        first.candidates
    )


def test_budget_caps_candidate_count_and_step_count():
    X = _clean_spectra()

    small = recommend_preprocessing(X, budget="small")
    extended = recommend_preprocessing(X, budget="extended")

    assert len(small.candidates) <= 4
    assert all(len(candidate.steps) <= 2 for candidate in small.candidates)
    assert len(extended.candidates) <= 12
    assert all(len(candidate.steps) <= 3 for candidate in extended.candidates)


def test_baseline_drift_adds_a_bounded_baseline_candidate():
    X = _clean_spectra() + 3.0 * np.linspace(-1.0, 1.0, 101)

    recommendation = recommend_preprocessing(X, budget="standard")

    assert "baseline_drift" in recommendation.profile.tags
    methods = [
        step.method
        for candidate in recommendation.candidates
        for step in candidate.steps
    ]
    assert any(method in {"asls", "detrend", "airpls"} for method in methods)


def test_extended_baseline_drift_can_compare_arpls_but_not_rubberband():
    X = _clean_spectra() + 3.0 * np.linspace(-1.0, 1.0, 101)

    recommendation = recommend_preprocessing(X, budget="extended")
    methods = [
        step.method
        for candidate in recommendation.candidates
        for step in candidate.steps
    ]

    assert "arpls" in methods
    assert "rubberband" not in methods
    assert "rubberband" in recommendation.manual_recommendations


def test_spikes_are_detected_but_explicit_only_despike_is_not_auto_selected():
    X = _clean_spectra()
    X[::2, 50] += 20.0

    recommendation = recommend_preprocessing(X, budget="extended")

    assert "isolated_spikes" in recommendation.profile.tags
    assert "despike" in recommendation.manual_recommendations
    assert "median_filter" in recommendation.manual_recommendations
    assert all(
        step.method not in {"despike", "median_filter"}
        for candidate in recommendation.candidates
        for step in candidate.steps
    )


def test_high_frequency_noise_can_compare_whittaker_smoothing():
    rng = np.random.default_rng(771)
    X = _clean_spectra() + rng.normal(0.0, 0.2, size=(40, 101))

    recommendation = recommend_preprocessing(X, budget="extended")
    methods = {
        step.method
        for candidate in recommendation.candidates
        for step in candidate.steps
    }

    assert "high_frequency_noise" in recommendation.profile.tags
    assert "whittaker_smooth" in methods


def test_extended_scatter_variation_can_compare_rnv():
    X = _clean_spectra()
    X = X * np.linspace(0.3, 2.5, X.shape[0])[:, None]

    recommendation = recommend_preprocessing(X, budget="extended")
    methods = {
        step.method
        for candidate in recommendation.candidates
        for step in candidate.steps
    }

    assert "scatter_variation" in recommendation.profile.tags
    assert "rnv" in methods


def test_low_signal_to_noise_never_adds_derivative_candidates():
    rng = np.random.default_rng(77)
    X = rng.normal(0.0, 1.0, size=(48, 101))

    recommendation = recommend_preprocessing(X, budget="extended")

    assert "low_signal_to_noise" in recommendation.profile.tags
    assert all(
        not step.method.startswith("derivative")
        for candidate in recommendation.candidates
        for step in candidate.steps
    )


def test_recommendation_does_not_depend_on_an_unpassed_holdout():
    calibration = _clean_spectra(n_samples=32)
    holdout_a = np.zeros((8, calibration.shape[1]))
    holdout_b = np.full((8, calibration.shape[1]), 1e9)

    first = recommend_preprocessing(calibration, budget="standard")
    _ = holdout_a
    second = recommend_preprocessing(calibration, budget="standard")
    _ = holdout_b

    assert first.as_dict() == second.as_dict()


def test_unknown_budget_fails_closed():
    with pytest.raises(ValueError, match="budget"):
        recommend_preprocessing(_clean_spectra(), budget="unbounded")
