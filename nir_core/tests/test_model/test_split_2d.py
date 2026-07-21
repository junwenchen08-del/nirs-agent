"""Shape compatibility tests for multi-target dataset splitting."""

from __future__ import annotations

import numpy as np

from nir_core.model.evaluation import split_dataset


def test_split_preserves_two_dimensional_targets() -> None:
    X = np.arange(240, dtype=float).reshape(40, 6)
    y = np.column_stack((np.arange(40), np.arange(40) * 2, np.arange(40) * -1))

    splits = split_dataset(X, y, random_state=7)

    assert [target.shape[1] for _, target in splits] == [3, 3, 3]
    for X_part, y_part in splits:
        for row, targets in zip(X_part, y_part):
            original_index = int(row[0] // 6)
            np.testing.assert_array_equal(targets, y[original_index])


def test_split_keeps_legacy_one_dimensional_targets_one_dimensional() -> None:
    X = np.arange(240, dtype=float).reshape(40, 6)
    y = np.arange(40, dtype=float)

    splits = split_dataset(X, y, random_state=7)

    assert all(target.ndim == 1 for _, target in splits)
