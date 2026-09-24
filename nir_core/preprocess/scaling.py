"""Scaling and normalization utilities.

- :func:`mean_center` -- subtract column means.
- :func:`autoscale`   -- mean-center and scale to unit column std.
- :func:`normalize`   -- row-wise L1 / L2 / max normalization.

All functions are pure and operate on 2D arrays (1D inputs are promoted and
returned with their original shape).
"""

from __future__ import annotations

import numpy as np


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a 2D float view of ``X`` plus a flag indicating 1D input."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr[np.newaxis, :], True
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D.")
    return X_arr, False


def mean_center(X: np.ndarray) -> np.ndarray:
    """Subtract the column-wise mean from each spectrum.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.

    Returns:
        Mean-centered spectra with the same shape as ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    out = arr - arr.mean(axis=0, keepdims=True)
    return out[0] if was_1d else out


def autoscale(X: np.ndarray) -> np.ndarray:
    """Mean-center and scale to unit column standard deviation.

    Columns whose std is zero are returned as zeros (constant columns carry
    no variance information).

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.

    Returns:
        Autoscaled spectra with the same shape as ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True, ddof=0)
    safe_std = np.where(std == 0.0, 1.0, std)
    out = (arr - mean) / safe_std
    out = np.where(std == 0.0, 0.0, out)
    return out[0] if was_1d else out


def normalize(X: np.ndarray, norm: str = "l2") -> np.ndarray:
    """Row-wise normalization (L1, L2, or max).

    Each row is scaled so that its L1 norm, L2 norm, or maximum absolute
    value equals 1, depending on ``norm``. Rows whose norm is zero are
    returned as zeros.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        norm: One of ``{"l1", "l2", "max"}``.

    Returns:
        Normalized spectra with the same shape as ``X``.

    Raises:
        ValueError: If ``norm`` is not one of the allowed strings.
    """
    arr, was_1d = _ensure_2d(X)
    if norm == "l1":
        scale = np.abs(arr).sum(axis=1, keepdims=True)
    elif norm == "l2":
        scale = np.sqrt((arr**2).sum(axis=1, keepdims=True))
    elif norm == "max":
        scale = np.abs(arr).max(axis=1, keepdims=True)
    else:
        raise ValueError(f"norm must be one of 'l1', 'l2', 'max'; got {norm!r}.")
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    out = arr / safe_scale
    out = np.where(scale == 0.0, 0.0, out)
    return out[0] if was_1d else out
