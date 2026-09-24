"""Validators for spectral matrices and reference values.

These are pure functions returning a list of human-readable problem
descriptions. An empty list means "no problems detected".
"""

from __future__ import annotations

import numpy as np


def validate_spectra(X: np.ndarray) -> list[str]:
    """Validate a spectral matrix and report issues.

    Checks performed:
      - NaN values present.
      - Infinite (non-finite) values present.
      - Negative absorbance values (physically implausible for absorbance).
      - Rows that are entirely zero (dead samples).
      - Columns that are entirely zero (dead wavelengths).

    Args:
        X: Spectral matrix of shape ``(n_samples, n_wavelengths)``.

    Returns:
        List of problem-description strings. Empty if no issues found.

    Raises:
        ValueError: If ``X`` is not a 2D array.
    """
    if X.ndim != 2:
        raise ValueError(f"validate_spectra expects a 2D array, got shape {X.shape!r}")

    problems: list[str] = []
    nan_mask = np.isnan(X)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        problems.append(f"Matrix contains {n_nan} NaN value(s).")

    # Inf check (also covers -inf). Use np.isinf to be precise.
    inf_mask = np.isinf(X)
    if inf_mask.any():
        n_inf = int(inf_mask.sum())
        problems.append(f"Matrix contains {n_inf} infinite value(s).")

    # Negative absorbance: only meaningful for finite values.
    finite_mask = np.isfinite(X)
    if (X < 0).any() and finite_mask.any():
        n_neg = int(((X < 0) & finite_mask).sum())
        if n_neg > 0:
            problems.append(
                f"Matrix contains {n_neg} negative absorbance value(s); "
                "absorbance should be non-negative."
            )

    # All-zero rows.
    zero_rows = np.where(np.all(X == 0, axis=1))[0]
    if zero_rows.size > 0:
        sample = ", ".join(str(int(i)) for i in zero_rows[:5])
        more = "" if zero_rows.size <= 5 else f" ... (+{zero_rows.size - 5} more)"
        problems.append(
            f"{int(zero_rows.size)} all-zero row(s) detected (indices: {sample}{more})."
        )

    # All-zero columns.
    zero_cols = np.where(np.all(X == 0, axis=0))[0]
    if zero_cols.size > 0:
        sample = ", ".join(str(int(i)) for i in zero_cols[:5])
        more = "" if zero_cols.size <= 5 else f" ... (+{zero_cols.size - 5} more)"
        problems.append(
            f"{int(zero_cols.size)} all-zero column(s) detected "
            f"(indices: {sample}{more})."
        )

    return problems


def validate_reference_values(y: np.ndarray) -> list[str]:
    """Validate a 1D reference-values vector and report issues.

    Checks performed:
      - NaN values present.
      - Infinite values present.
      - Constant vector (zero variance) -- uninformative for calibration.
      - Extreme outliers: values more than 5 standard deviations from the
        mean (computed on finite values only).

    Args:
        y: Reference values of shape ``(n_samples,)``.

    Returns:
        List of problem-description strings. Empty if no issues found.

    Raises:
        ValueError: If ``y`` is not 1D.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(
            f"validate_reference_values expects a 1D array, got shape {y.shape!r}"
        )

    problems: list[str] = []

    nan_mask = np.isnan(y)
    if nan_mask.any():
        n_nan = int(nan_mask.sum())
        problems.append(f"Reference vector contains {n_nan} NaN value(s).")

    inf_mask = np.isinf(y)
    if inf_mask.any():
        n_inf = int(inf_mask.sum())
        problems.append(f"Reference vector contains {n_inf} infinite value(s).")

    finite = y[np.isfinite(y)]
    if finite.size == 0:
        problems.append("Reference vector has no finite values.")
        return problems

    # Constant check (zero variance).
    if finite.size > 1:
        std = float(np.std(finite))
        if std == 0.0:
            problems.append(
                "Reference vector is constant (zero variance); unsuitable "
                "for calibration."
            )
        else:
            # Extreme outliers: |z| > 5.
            mean = float(np.mean(finite))
            z = (finite - mean) / std
            outlier_mask = np.abs(z) > 5.0
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                problems.append(
                    f"Reference vector contains {n_outliers} extreme "
                    "outlier(s) (|z| > 5)."
                )

    return problems
