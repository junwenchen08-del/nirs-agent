"""Tests for nir_core.utils.drift."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.models import ModelResult
from nir_core.utils.drift import (
    compute_drift_index,
    compute_mahalanobis_drift,
    compute_reference_drift,
    fit_monitoring_reference,
)


def _make_train(n: int = 100, n_wv: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(n, n_wv))
    return base


def test_mahalanobis_drift_flags_injected_outlier():
    X_train = _make_train()
    rng = np.random.default_rng(1)
    X_new = rng.normal(0.0, 1.0, size=(10, X_train.shape[1]))
    # Inject one clearly drifted sample.
    X_new[0] += 10.0
    res = compute_mahalanobis_drift(X_train, X_new, threshold=3.0)
    assert 0 in res["flagged_indices"]
    assert res["drift_score"] > 0.0


def test_mahalanobis_drift_score_in_unit_interval():
    X_train = _make_train()
    rng = np.random.default_rng(2)
    X_new = rng.normal(0.0, 1.0, size=(20, X_train.shape[1]))
    res = compute_mahalanobis_drift(X_train, X_new, threshold=3.0)
    assert 0.0 <= res["drift_score"] <= 1.0


def test_mahalanobis_drift_distances_length():
    X_train = _make_train()
    X_new = _make_train(n=15, seed=5)
    res = compute_mahalanobis_drift(X_train, X_new)
    assert res["distances"].shape == (15,)


def test_mahalanobis_drift_heatmap_shape():
    X_train = _make_train(n=80, n_wv=12)
    X_new = _make_train(n=7, n_wv=12, seed=9)
    res = compute_mahalanobis_drift(X_train, X_new)
    assert res["heatmap_data"].shape == (7, 12)


def test_mahalanobis_drift_empty_new():
    X_train = _make_train()
    X_new = np.empty((0, X_train.shape[1]))
    res = compute_mahalanobis_drift(X_train, X_new)
    assert res["drift_score"] == 0.0
    assert res["flagged_indices"].size == 0


def test_mahalanobis_drift_mismatched_wavelengths_raises():
    X_train = _make_train(n_wv=10)
    X_new = _make_train(n=5, n_wv=12, seed=3)
    with pytest.raises(ValueError):
        compute_mahalanobis_drift(X_train, X_new)


def test_compute_drift_index_in_unit_interval():
    X_new = _make_train(n=20, n_wv=15, seed=4)
    mr = ModelResult(method="pls", metrics={"RPD": 4.0, "R2": 0.9})
    idx = compute_drift_index(mr, X_new)
    assert 0.0 <= idx <= 1.0


def test_compute_drift_index_zero_for_empty():
    X_new = np.empty((0, 5))
    mr = ModelResult(method="pls", metrics={"RPD": 4.0})
    assert compute_drift_index(mr, X_new) == 0.0


def test_compute_drift_index_higher_for_drifted_batch():
    rng = np.random.default_rng(7)
    X_clean = rng.normal(0.0, 1.0, size=(40, 10))
    X_drifted = rng.normal(5.0, 1.0, size=(40, 10))  # shifted mean
    mr = ModelResult(method="pls", metrics={"RPD": 3.0, "R2": 0.9})
    idx_clean = compute_drift_index(mr, X_clean)
    idx_drifted = compute_drift_index(mr, X_drifted)
    assert idx_drifted >= idx_clean


def test_compute_drift_index_fallback_without_rpd():
    X_new = _make_train(n=20, n_wv=10, seed=2)
    mr = ModelResult(method="pls", metrics={"R2": 0.9})  # no RPD
    idx = compute_drift_index(mr, X_new)
    assert 0.0 <= idx <= 1.0


def test_reference_drift_detects_shift_against_training_domain():
    X_train = _make_train(n=160, n_wv=16, seed=11)
    reference = fit_monitoring_reference(X_train)
    clean = _make_train(n=30, n_wv=16, seed=12)
    shifted = clean + 6.0

    clean_result = compute_reference_drift(reference, clean)
    shifted_result = compute_reference_drift(reference, shifted)

    assert clean_result["available"] is True
    assert shifted_result["drift_score"] > clean_result["drift_score"]
    assert shifted_result["drift_score"] >= 0.9


def test_reference_drift_rejects_wrong_feature_count():
    reference = fit_monitoring_reference(_make_train(n=80, n_wv=10))
    with pytest.raises(ValueError, match="expects 10"):
        compute_reference_drift(reference, _make_train(n=5, n_wv=9, seed=3))
