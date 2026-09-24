"""Savitzky-Golay smoothing and derivative computation.

Thin, validated wrappers around :func:`scipy.signal.savgol_filter` that
operate on 2D spectral matrices (and 1D single spectra) and raise clear
errors on invalid parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
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
    X: np.ndarray,
    window: int = 11,
    order: int = 2,
    deriv: int = 1,
    delta: float = 1.0,
    spacing_mode: str = "index",
    wv: np.ndarray | None = None,
) -> np.ndarray:
    """Savitzky-Golay derivative.

    Computes the ``deriv``-th derivative along the wavelength axis via
    :func:`scipy.signal.savgol_filter`.

    By default the derivative is expressed on the *index scale* (``delta=1.0``),
    which is the legacy, backward-compatible behaviour. To obtain a physical
    derivative (per wavelength/wavenumber unit), pass ``spacing_mode="wavelength"``
    together with an equally spaced ``wv``; the spacing ``delta`` is then derived
    from ``wv`` and the derivative is scaled accordingly. A non-uniform ``wv`` is
    rejected so an average ``delta`` is never silently substituted for a real,
    regular axis.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        window: Odd positive integer smoothing window. Must be ``> order``.
        order: Polynomial order used for local fitting. Must satisfy
            ``deriv <= order < window``.
        deriv: Derivative order (1 for first derivative, 2 for second).
        delta: Sampling interval used to scale the derivative. Defaults to 1.0
            (index scale). Ignored when ``spacing_mode="wavelength"``.
        spacing_mode: ``"index"`` (default) or ``"wavelength"``.
        wv: Wavelength axis; required when ``spacing_mode="wavelength"``.

    Returns:
        Differentiated spectra with the same shape as ``X``.

    Raises:
        ValueError: If window/order invalid, ``deriv > order``, ``spacing_mode``
            unknown, or a missing / non-uniform wavelength axis is given in
            wavelength mode.
    """
    _validate_sg_window(window, order)
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}.")
    if deriv > order:
        raise ValueError(f"deriv must be <= order; got deriv={deriv}, order={order}.")
    if spacing_mode not in {"index", "wavelength"}:
        raise ValueError(
            f"spacing_mode must be 'index' or 'wavelength', got {spacing_mode!r}."
        )
    arr, was_1d = _ensure_2d(X)
    if spacing_mode == "wavelength":
        if wv is None:
            raise ValueError(
                "spacing_mode='wavelength' requires a wavelength axis (wv)."
            )
        wv_arr = np.asarray(wv, dtype=float).ravel()
        if wv_arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"wavelength axis length {wv_arr.shape[0]} does not match "
                f"spectrum width {arr.shape[1]}."
            )
        diffs = np.diff(wv_arr)
        if diffs.size == 0:
            raise ValueError("wavelength axis must contain at least two points.")
        if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-9):
            raise ValueError(
                "spacing_mode='wavelength' requires an equally spaced wavelength "
                "axis; resample to a regular axis first (e.g. nir_align_wavelengths)."
            )
        delta = abs(float(diffs[0]))
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta must be positive and finite, got {delta!r}.")
    out = savgol_filter(
        arr, window_length=window, polyorder=order, deriv=deriv, axis=1, delta=delta
    )
    return out[0] if was_1d else out


def norris_williams_derivative(
    X: np.ndarray,
    *,
    gap: int = 3,
    segment: int = 5,
    deriv: int = 1,
    delta: float = 1.0,
) -> np.ndarray:
    """Norris-Williams gap-segment derivative with shape-preserving edges.

    Spectra are first averaged over an odd ``segment`` window. A centered
    finite difference separated by ``gap`` points then estimates the first or
    second derivative. Edge values, where a centered estimate is unavailable,
    are filled with the nearest valid derivative so the wavelength count is
    preserved for downstream model artifacts.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or one spectrum.
        gap: Positive integer half-gap used by the centered difference.
        segment: Odd positive moving-average window. Use 1 for no averaging.
        deriv: Derivative order, either 1 or 2.
        delta: Physical spacing between adjacent wavelength points. Keep 1.0
            for the legacy index scale.

    Returns:
        Derivative spectra with the same shape as ``X``.
    """
    if not isinstance(gap, int) or isinstance(gap, bool) or gap < 1:
        raise ValueError(f"gap must be a positive integer, got {gap!r}.")
    if (
        not isinstance(segment, int)
        or isinstance(segment, bool)
        or segment < 1
        or segment % 2 == 0
    ):
        raise ValueError(f"segment must be an odd positive integer, got {segment!r}.")
    if deriv not in {1, 2}:
        raise ValueError(f"deriv must be 1 or 2, got {deriv!r}.")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta must be positive and finite, got {delta!r}.")

    arr, was_1d = _ensure_2d(X)
    if arr.shape[1] <= 2 * gap:
        raise ValueError(
            f"spectrum is too short ({arr.shape[1]} points) for gap={gap}."
        )
    smoothed = uniform_filter1d(arr, size=segment, axis=1, mode="nearest")
    out = np.empty_like(smoothed)
    left = smoothed[:, : -2 * gap]
    center = smoothed[:, gap:-gap]
    right = smoothed[:, 2 * gap :]
    if deriv == 1:
        interior = (right - left) / (2.0 * gap * float(delta))
    else:
        spacing = gap * float(delta)
        interior = (right - 2.0 * center + left) / (spacing**2)
    out[:, gap:-gap] = interior
    out[:, :gap] = interior[:, :1]
    out[:, -gap:] = interior[:, -1:]
    return out[0] if was_1d else out
