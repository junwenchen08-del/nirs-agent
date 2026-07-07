"""Tests for :mod:`nir_core.io.validators`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.io.validators import validate_reference_values, validate_spectra


def test_validate_spectra_clean() -> None:
    """A clean all-positive matrix should yield no problems."""
    X = np.ones((5, 10)) + 0.1
    assert validate_spectra(X) == []


def test_validate_spectra_nan() -> None:
    X = np.ones((3, 3))
    X[0, 0] = np.nan
    problems = validate_spectra(X)
    assert any("NaN" in p for p in problems)


def test_validate_spectra_inf() -> None:
    X = np.ones((3, 3))
    X[1, 1] = np.inf
    problems = validate_spectra(X)
    assert any("infinite" in p for p in problems)


def test_validate_spectra_negative() -> None:
    X = np.ones((3, 3))
    X[0, 0] = -0.5
    problems = validate_spectra(X)
    assert any("negative absorbance" in p for p in problems)


def test_validate_spectra_zero_row() -> None:
    X = np.ones((4, 5))
    X[2, :] = 0.0
    problems = validate_spectra(X)
    assert any("all-zero row" in p for p in problems)


def test_validate_spectra_zero_col() -> None:
    X = np.ones((4, 5))
    X[:, 1] = 0.0
    problems = validate_spectra(X)
    assert any("all-zero column" in p for p in problems)


def test_validate_spectra_not_2d() -> None:
    with pytest.raises(ValueError):
        validate_spectra(np.zeros(5))


def test_validate_reference_values_clean() -> None:
    y = np.linspace(0, 10, 50)
    assert validate_reference_values(y) == []


def test_validate_reference_values_nan() -> None:
    y = np.linspace(0, 10, 50)
    y[3] = np.nan
    problems = validate_reference_values(y)
    assert any("NaN" in p for p in problems)


def test_validate_reference_values_inf() -> None:
    y = np.linspace(0, 10, 50)
    y[3] = np.inf
    problems = validate_reference_values(y)
    assert any("infinite" in p for p in problems)


def test_validate_reference_values_constant() -> None:
    y = np.full(20, 5.0)
    problems = validate_reference_values(y)
    assert any("constant" in p for p in problems)


def test_validate_reference_values_outlier() -> None:
    """A value beyond 5 std should be flagged as an extreme outlier."""
    y = np.linspace(0, 1, 100)
    y[0] = 100.0  # huge outlier
    problems = validate_reference_values(y)
    assert any("outlier" in p for p in problems)


def test_validate_reference_values_not_1d() -> None:
    with pytest.raises(ValueError):
        validate_reference_values(np.zeros((3, 3)))
