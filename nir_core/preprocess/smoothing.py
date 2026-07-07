"""Savitzky-Golay smoothing and derivative computation.

Thin, validated wrappers around :func:`scipy.signal.savgol_filter` that
operate on 2D spectral matrices (and 1D single spectra) and raise clear
errors on invalid parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a 2D float view of ``X`` plus a flag indicating 1D input."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr[np.newaxis, :], True
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D.")
    return X_arr, False


def _validate_sg_window(window: int, order: int) -> None:
    """Validate Savitzky-Golay window and polynomial order."""
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}.")
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}.")
    if order < 0 or order >= window:
        raise ValueError(
            f"order must satisfy 0 <= order < window; got order={order}, "
            f"window={window}."
        )


def sg_smooth(X: np.ndarray, window: int = 11, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing.

    Applies :func:`scipy.signal.savgol_filter` with ``deriv=0`` along the
    wavelength axis (last axis) of every spectrum.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        window: Odd positive integer, the smoothing window length. Must be
            greater than ``order``.
        order: Polynomial order used for local fitting. Must be ``< window``.

    Returns:
        Smoothed spectra with the same shape as ``X``.

    Raises:
        ValueError: If ``window`` is even, non-positive, or ``<= order``.
    """
    _validate_sg_window(window, order)
    arr, was_1d = _ensure_2d(X)
    out = savgol_filter(arr, window_length=window, polyorder=order, deriv=0, axis=1)
    return out[0] if was_1d else out


def sg_derivative(
    X: np.ndarray, window: int = 11, order: int = 2, deriv: int = 1
) -> np.ndarray:
    """Savitzky-Golay derivative.

    Computes the ``deriv``-th derivative along the wavelength axis via
    :func:`scipy.signal.savgol_filter`.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        window: Odd positive integer smoothing window. Must be ``> order``.
        order: Polynomial order used for local fitting. Must satisfy
            ``deriv <= order < window``.
        deriv: Derivative order (1 for first derivative, 2 for second).

    Returns:
        Differentiated spectra with the same shape as ``X``.

    Raises:
        ValueError: If window/order invalid or ``deriv > order``.
    """
    _validate_sg_window(window, order)
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}.")
    if deriv > order:
        raise ValueError(
            f"deriv must be <= order; got deriv={deriv}, order={order}."
        )
    arr, was_1d = _ensure_2d(X)
    out = savgol_filter(
        arr, window_length=window, polyorder=order, deriv=deriv, axis=1
    )
    return out[0] if was_1d else out
