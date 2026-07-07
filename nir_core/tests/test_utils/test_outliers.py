"""Tests for outlier detection and leakage checks in nir_core.utils.validation."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.utils.validation import (
    check_train_test_split_leakage,
    detect_outliers_mahalanobis,
    detect_outliers_pca_t2,
)


def test_pca_t2_detects_injected_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(60, 15))
    X[5] += 15.0  # strong outlier
    flagged = detect_outliers_pca_t2(X, n_components=5, alpha=0.95)
    assert 5 in flagged
    assert flagged.dtype == int


def test_pca_t2_no_outliers_for_clean_data():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(100, 10))
    flagged = detect_outliers_pca_t2(X, n_components=3, alpha=0.99)
    # Should not flag the bulk; allow at most a couple of false positives
    # at the very strict 0.99 level, but assert the injected index never
    # appears since none was injected.
    assert flagged.size <= 5


def test_pca_t2_handles_tiny_input():
    X = np.array([[1.0, 2.0], [2.0, 3.0]])
    flagged = detect_outliers_pca_t2(X, n_components=2)
    assert flagged.size == 0


def test_mahalanobis_detects_injected_outlier():
    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, size=(80, 8))
    X[10] += 8.0
    flagged = detect_outliers_mahalanobis(X, threshold=3.0)
    assert 10 in flagged


def test_mahalanobis_returns_empty_for_single_row():
    X = np.array([[1.0, 2.0, 3.0]])
    flagged = detect_outliers_mahalanobis(X, threshold=3.0)
    assert flagged.size == 0


def test_train_test_leakage_detects_duplicate_row():
    rng = np.random.default_rng(4)
    X_train = rng.normal(0.0, 1.0, size=(20, 5))
    X_test = rng.normal(0.0, 1.0, size=(5, 5))
    # Insert an exact duplicate of a training row into the test set.
    X_test[2] = X_train[3]
    assert check_train_test_split_leakage(X_train, X_test) is True


def test_train_test_leakage_no_false_positive():
    rng = np.random.default_rng(5)
    X_train = rng.normal(0.0, 1.0, size=(20, 5))
    X_test = rng.normal(0.0, 1.0, size=(5, 5)) + 100.0  # clearly different
    assert check_train_test_split_leakage(X_train, X_test) is False


def test_train_test_leakage_mismatched_features_returns_false():
    X_train = np.zeros((5, 3))
    X_test = np.zeros((5, 4))
    assert check_train_test_split_leakage(X_train, X_test) is False


def test_validation_functions_return_int_arrays():
    rng = np.random.default_rng(6)
    X = rng.normal(0.0, 1.0, size=(40, 6))
    f1 = detect_outliers_pca_t2(X, n_components=3)
    f2 = detect_outliers_mahalanobis(X, threshold=3.0)
    assert f1.dtype == int
    assert f2.dtype == int
