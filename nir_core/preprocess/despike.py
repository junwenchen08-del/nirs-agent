"""Robust removal of isolated detector spikes from spectra."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        return arr[np.newaxis, :], True
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")
    return arr, False


def despike(
    X: np.ndarray,
    *,
    window: int = 5,
    z_threshold: float = 6.0,
) -> np.ndarray:
    """Replace isolated spectral spikes with a local median estimate.

    A median-filtered spectrum is used as the local reference. Residuals are
    scored with a robust MAD scale plus a small standard-deviation floor so
    smooth peak curvature is not mistaken for a detector spike.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or one spectrum.
        window: Odd local median window, at least 3.
        z_threshold: Positive residual threshold in robust-scale units.

    Returns:
        Corrected spectra with the same shape as ``X``.
    """
    if not isinstance(window, int) or isinstance(window, bool) or window < 3:
        raise ValueError(f"window must be an odd integer >= 3, got {window!r}.")
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}.")
    if not np.isfinite(z_threshold) or z_threshold <= 0:
        raise ValueError(
            f"z_threshold must be a positive finite number, got {z_threshold!r}."
        )

    arr, was_1d = _ensure_2d(X)
    local_median = median_filter(arr, size=(1, window), mode="nearest")
    residual = arr - local_median
    residual_center = np.median(residual, axis=1, keepdims=True)
    mad = np.median(np.abs(residual - residual_center), axis=1, keepdims=True)
    robust_scale = 1.4826 * mad
    std_floor = 0.1 * residual.std(axis=1, keepdims=True, ddof=0)
    # A small fraction of the spectrum's dynamic range protects broad,
    # smoothly curved absorption maxima whose median residual can otherwise
    # look extreme when the rest of a noiseless synthetic spectrum is flat.
    dynamic_floor = 1e-3 * np.ptp(arr, axis=1, keepdims=True)
    scale = np.maximum(np.maximum(robust_scale, std_floor), dynamic_floor)
    threshold = float(z_threshold) * scale
    mask = (scale > 0.0) & (np.abs(residual - residual_center) > threshold)
    out = np.where(mask, local_median, arr)
    return out[0] if was_1d else out
