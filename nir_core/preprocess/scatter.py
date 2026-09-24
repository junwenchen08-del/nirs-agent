"""Scatter-correction algorithms.

Implements Standard Normal Variate (SNV), robust SNV, Multiplicative Scatter
Correction (MSC), and Extended Multiplicative Scatter Correction (EMSC).
These methods reduce additive/multiplicative scatter effects caused by
particle-size and path-length variation in diffuse-reflectance NIR.

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
        raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D.")
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


def robust_snv(X: np.ndarray, *, consistency: float = 1.4826) -> np.ndarray:
    """Robust Standard Normal Variate using the row median and MAD.

    Compared with classical SNV, the median and median absolute deviation
    (MAD) are less sensitive to isolated detector spikes and heavy-tailed
    noise. When a row has zero MAD the ordinary standard deviation is used as
    a deterministic fallback; a fully constant row becomes zeros.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or one spectrum.
        consistency: Gaussian consistency factor applied to MAD. The default
            ``1.4826`` makes the robust scale comparable with standard
            deviation for normally distributed values.

    Returns:
        Robustly standardized spectra with the same shape as ``X``.
    """
    if not np.isfinite(consistency) or consistency <= 0:
        raise ValueError(
            f"consistency must be a positive finite number, got {consistency!r}."
        )
    arr, was_1d = _ensure_2d(X)
    median = np.median(arr, axis=1, keepdims=True)
    centered = arr - median
    mad = np.median(np.abs(centered), axis=1, keepdims=True)
    robust_scale = float(consistency) * mad
    std = arr.std(axis=1, keepdims=True, ddof=0)
    scale = np.where(robust_scale > 0.0, robust_scale, std)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    out = centered / safe_scale
    out = np.where(scale > 0.0, out, 0.0)
    return out[0] if was_1d else out


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


def emsc(
    X: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    wv: np.ndarray | None = None,
    polynomial_order: int = 2,
    min_multiplicative: float = 1e-12,
) -> np.ndarray:
    """Extended Multiplicative Scatter Correction (EMSC).

    Each spectrum is modelled as a constant offset, a multiplicative copy of
    a reference spectrum, and optional low-order polynomial baseline terms.
    The fitted baseline is removed and the remaining signal is divided by the
    multiplicative coefficient.

    When ``reference`` is omitted, the input mean spectrum is used. Production
    model pipelines should instead fit the reference on training data and
    reuse it for validation and prediction; :class:`PreprocessingPipeline`
    implements that leakage-safe behaviour.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or one spectrum.
        reference: Fixed reference spectrum. Defaults to the input mean.
        wv: Optional wavelength axis. Column indices are used when omitted.
        polynomial_order: Baseline polynomial order from 0 through 3.
        min_multiplicative: Minimum absolute multiplicative coefficient. A
            smaller fitted value is considered non-identifiable and fails.

    Returns:
        EMSC-corrected spectra with the same shape as ``X``.
    """
    if (
        not isinstance(polynomial_order, int)
        or isinstance(polynomial_order, bool)
        or not 0 <= polynomial_order <= 3
    ):
        raise ValueError(
            "polynomial_order must be an integer between 0 and 3; "
            f"got {polynomial_order!r}."
        )
    if not np.isfinite(min_multiplicative) or min_multiplicative <= 0:
        raise ValueError("min_multiplicative must be a positive finite number.")

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
    if not np.isfinite(ref).all():
        raise ValueError("reference must contain only finite values.")

    if wv is None:
        axis = np.arange(n_wavelengths, dtype=float)
    else:
        axis = np.asarray(wv, dtype=float).ravel()
        if axis.shape[0] != n_wavelengths:
            raise ValueError(
                f"wv length {axis.shape[0]} does not match n_wavelengths "
                f"{n_wavelengths}."
            )
        if not np.isfinite(axis).all():
            raise ValueError("wv must contain only finite values.")
    axis_std = float(axis.std())
    axis_scaled = (axis - axis.mean()) / (axis_std if axis_std > 0 else 1.0)

    columns = [np.ones(n_wavelengths), ref]
    columns.extend(axis_scaled**power for power in range(1, polynomial_order + 1))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, arr.T, rcond=None)
    multiplicative = coefficients[1]
    invalid = np.abs(multiplicative) <= float(min_multiplicative)
    if np.any(invalid):
        rows = np.flatnonzero(invalid).tolist()
        raise ValueError(
            f"EMSC multiplicative coefficient is too close to zero for rows {rows}."
        )

    baseline = np.outer(np.ones(n_wavelengths), coefficients[0])
    for power in range(1, polynomial_order + 1):
        baseline += np.outer(axis_scaled**power, coefficients[power + 1])
    out = ((arr.T - baseline) / multiplicative).T
    return out[0] if was_1d else out
