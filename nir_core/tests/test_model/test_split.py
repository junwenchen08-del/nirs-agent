"""Tests for split_dataset."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.evaluation import split_dataset


def _row_overlap(A: np.ndarray, B: np.ndarray) -> int:
    """Count overlapping rows between two arrays (treating rows as tuples)."""
    a = set(map(tuple, A.tolist()))
    b = set(map(tuple, B.tolist()))
    return len(a & b)


def test_split_returns_three_disjoint_sets(synthetic_data):
    """train/val/test must have no overlapping rows."""
    X, y = synthetic_data.X, synthetic_data.y
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
        X, y, test_ratio=0.20, val_ratio=0.10, random_state=42
    )
    n_total = X.shape[0]
    # Proportions approximately correct.
    assert abs(X_te.shape[0] / n_total - 0.20) < 0.05
    assert abs(X_val.shape[0] / n_total - 0.10) < 0.05
    # Train + val + test == total.
    assert X_tr.shape[0] + X_val.shape[0] + X_te.shape[0] == n_total
    # No overlap between any pair.
    assert _row_overlap(X_tr, X_val) == 0
    assert _row_overlap(X_tr, X_te) == 0
    assert _row_overlap(X_val, X_te) == 0


def test_split_reproducible(synthetic_data):
    """Same random_state -> identical splits."""
    X, y = synthetic_data.X, synthetic_data.y
    s1 = split_dataset(X, y, random_state=42)
    s2 = split_dataset(X, y, random_state=42)
    for a, b in zip(s1, s2):
        assert np.allclose(a[0], b[0])
        assert np.allclose(a[1], b[1])


def test_split_different_seeds_differ(synthetic_data):
    """Different seeds should (almost certainly) produce different splits."""
    X, y = synthetic_data.X, synthetic_data.y
    s1 = split_dataset(X, y, random_state=1)
    s2 = split_dataset(X, y, random_state=2)
    # Train sets should differ.
    assert not np.allclose(s1[0][0], s2[0][0])


def test_split_invalid_ratios_raise(synthetic_data):
    X, y = synthetic_data.X, synthetic_data.y
    with pytest.raises(ValueError):
        split_dataset(X, y, test_ratio=0.0)
    with pytest.raises(ValueError):
        split_dataset(X, y, test_ratio=1.0)
    with pytest.raises(ValueError):
        split_dataset(X, y, val_ratio=-0.1)
    with pytest.raises(ValueError):
        split_dataset(X, y, test_ratio=0.6, val_ratio=0.5)


def test_split_shape_mismatch_raises(synthetic_data):
    X = synthetic_data.X
    y_short = np.zeros(X.shape[0] - 1)
    with pytest.raises(ValueError):
        split_dataset(X, y_short)


def test_split_y_alignment(synthetic_data):
    """Each row's X and y must stay aligned after the split."""
    X, y = synthetic_data.X, synthetic_data.y
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
        X, y, random_state=42
    )
    # For each split row, find its index in the original X and confirm y matches.
    for X_split, y_split in [(X_tr, y_tr), (X_val, y_val), (X_te, y_te)]:
        for row, label in zip(X_split, y_split):
            # Find matching row in the original X.
            matches = np.where(np.all(X == row, axis=1))[0]
            assert len(matches) == 1
            assert y[matches[0]] == label
