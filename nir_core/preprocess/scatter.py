"""Scatter-correction algorithms.

Implements Standard Normal Variate (SNV) and Multiplicative Scatter
Correction (MSC). Both reduce multiplicative scatter effects caused by
particle-size / path-length variation in diffuse-reflectance NIR.

All functions are pure (input is never mutated) and accept either a single
spectrum (1D array) or a batch of spectra (2D array ``(n_samples, n_wavelengths)``).
"""

from __future__ import annotations

import numpy as np


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a 2D view of ``X`` plus a flag indicating whether the input was 1D.

    The returned array shares memory with ``X``; callers must not mutate it
    in place. Use this to normalize the handling of 1D / 2D inputs.
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr[np.newaxis, :], True
    if X_arr.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D array, got {X_arr.ndim}D."
        )
    return X_arr, False


def snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) scatter correction.

    For each spectrum (row) ``x`` computes ``(x - mean(x)) / std(x)`` using
    population std (ddof=0). Rows whose std is zero are returned as zeros
    (constant-offset spectra carry no relative scatter information).

    Args:
        X: Input spectra of shape ``(n_samples, n_wavelengths)`` or a single
            spectrum of shape ``(n_wavelengths,)``.

    Returns:
        SNV-corrected spectra with the same shape as ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True, ddof=0)
    # Protect against division by zero: where std==0 -> output row is 0.
    safe_std = np.where(std == 0.0, 1.0, std)
    out = (arr - mean) / safe_std
    out = np.where(std == 0.0, 0.0, out)
    if was_1d:
        return out[0]
    return out


def msc(X: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction (MSC).

    For each spectrum ``x`` fits the linear model ``x = a + b * reference``
    (ordinary least squares) and returns the corrected spectrum
    ``(x - a) / b``. When ``b`` is zero the reference is returned for that
    spectrum (no meaningful multiplicative correction possible).

    Args:
        X: Input spectra of shape ``(n_samples, n_wavelengths)`` or a single
            spectrum ``(n_wavelengths,)``.
        reference: Reference spectrum of shape ``(n_wavelengths,)``. If
            ``None`` the column-wise mean of ``X`` is used.

    Returns:
        MSC-corrected spectra with the same shape as ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    n_wavelengths = arr.shape[1]

    if reference is None:
        ref = arr.mean(axis=0)
    else:
        ref = np.asarray(reference, dtype=float).ravel()
        if ref.shape[0] != n_wavelengths:
            raise ValueError(
                f"reference length {ref.shape[0]} does not match "
                f"n_wavelengths {n_wavelengths}."
            )

    # Design matrix [1, ref] for OLS of x = a + b*ref.
    # Solve per-row via lstsq on the same design matrix.
    A = np.column_stack([np.ones(n_wavelengths), ref])
    # coeffs shape (n_samples, 2): column 0 = a (intercept), column 1 = b (slope).
    coeffs, *_ = np.linalg.lstsq(A, arr.T, rcond=None)
    coeffs = coeffs.T  # (n_samples, 2)
    a = coeffs[:, 0][:, np.newaxis]
    b = coeffs[:, 1][:, np.newaxis]

    safe_b = np.where(b == 0.0, 1.0, b)
    out = (arr - a) / safe_b
    # Where b==0, fall back to the reference spectrum (broadcast over rows).
    zero_b_mask = (b == 0.0).ravel()
    if zero_b_mask.any():
        out[zero_b_mask] = ref

    if was_1d:
        return out[0]
    return out
