"""Baseline-removal algorithms.

Implements:

- :func:`airpls`  -- adaptive iteratively reweighted penalized least squares
  (airPLS, Zhang et al. 2010).
- :func:`asls`    -- asymmetric least squares (Eilers & Boelins 2005).
- :func:`detrend` -- quadratic-polynomial detrending.

All three operate on each spectrum (column of a 2D matrix, i.e. each row)
independently and return ``X - baseline``.

The penalized-least-squares variants use :mod:`scipy.sparse` to construct a
second-order difference penalty matrix and solve the sparse linear system
via :func:`scipy.sparse.linalg.spsolve`.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _ensure_2d(X: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a 2D float view of ``X`` plus a flag indicating 1D input."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        return X_arr[np.newaxis, :], True
    if X_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got {X_arr.ndim}D.")
    return X_arr, False


def _second_difference_matrix(n: int) -> sparse.csc_matrix:
    """Build the (n-2) x n second-order difference operator ``D``.

    ``D @ z`` produces the second differences of ``z``. ``D.T @ D`` is the
    standard discrete roughness penalty. Requires ``n >= 3``; callers must
    guard shorter signals before invoking this.
    """
    # Use diags for efficient construction.
    D = sparse.diags(
        [1.0, -2.0, 1.0],
        offsets=[0, 1, 2],
        shape=(n - 2, n),
        format="csc",
    )
    return D


def _asls_baseline(y: np.ndarray, lambda_: float, p: float, max_iters: int) -> np.ndarray:
    """Compute asLS baseline of a 1D signal ``y``.

    Args:
        y: 1D signal of length ``n``.
        lambda_: Smoothness penalty weight.
        p: Asymmetry weight for points above the baseline (0 < p < 1).
        max_iters: Maximum number of weight-update iterations.

    Returns:
        Baseline vector of length ``n``.
    """
    n = y.shape[0]
    if n < 3:
        # Too few points to build a second-difference penalty; return a zero
        # baseline so the signal is left unchanged after subtraction.
        return np.zeros(n)
    D = _second_difference_matrix(n)
    # Penalty matrix H = D.T @ D (n x n), sparse.
    H = (D.T @ D).tocsc()
    # Weight matrix W is diagonal; iterate.
    w = np.ones(n)
    base = np.zeros(n)
    for _ in range(max_iters):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        Z = W + lambda_ * H
        # Solve Z @ base = w * y.
        base = spsolve(Z.tocsc(), w * y)
        # Update weights: above baseline -> p, below -> 1-p.
        new_w = np.where(y > base, p, 1.0 - p)
        if np.allclose(new_w, w):
            w = new_w
            break
        w = new_w
    return base


def _airpls_baseline(
    y: np.ndarray, lambda_: float, porder: int, max_iters: int
) -> np.ndarray:
    """Compute airPLS baseline of a 1D signal ``y`` (Zhang 2010).

    Args:
        y: 1D signal of length ``n``.
        lambda_: Smoothness penalty weight.
        porder: Deprecated, retained only for API compatibility. The original
            airPLS paper does not use this parameter; weights follow the
            standard formula ``exp(it * |r_i| / |sum_neg|)``.
        max_iters: Maximum number of iterations.

    Returns:
        Baseline vector of length ``n``.
    """
    n = y.shape[0]
    if n < 3:
        # Too few points to build a second-difference penalty.
        return np.zeros(n)
    D = _second_difference_matrix(n)
    H = (D.T @ D).tocsc()
    w = np.ones(n)
    base = np.zeros(n)
    for it in range(1, max_iters + 1):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        Z = W + lambda_ * H
        base = spsolve(Z.tocsc(), w * y)
        residual = y - base
        # Points above baseline (residual > 0) -> signal, weight 0.
        # Points below baseline (residual < 0) -> re-weighted by magnitude.
        neg_mask = residual < 0
        if not neg_mask.any():
            # All residuals non-negative: converged.
            break
        neg_residual = residual[neg_mask]
        # Sum of negative residuals (a measure of baseline overshoot).
        sum_neg = np.sum(neg_residual)
        # Convergence criterion (|sum_neg| / |y| < 0.1% of signal range).
        if abs(sum_neg) / (abs(y).sum() + 1e-12) < 1e-3:
            break
        # airPLS weight update (Zhang 2010, Analyst 135:1138-1146):
        #   positive residuals (signal above baseline) -> weight 0
        #   negative residuals -> weight = exp(it * |r_i| / |sum_neg|)
        # Normalising by |sum_neg| (not max|r|) follows the original paper
        # and keeps the weight scale stable across iterations.
        sum_neg_abs = abs(float(sum_neg))
        safe_denom = sum_neg_abs if sum_neg_abs > 1e-12 else 1e-12
        new_w = np.zeros(n)
        rn = -residual[neg_mask] / safe_denom
        new_w[neg_mask] = np.exp(it * rn)
        # Guard against overflow in late iterations (exp can grow large).
        new_w = np.clip(new_w, 0.0, 1e10)
        # Guard against all-zero weights (would make the system singular).
        if new_w.sum() == 0:
            break
        if np.allclose(new_w, w):
            w = new_w
            break
        w = new_w
    return base


def airpls(
    X: np.ndarray,
    lambda_: float = 1e7,
    porder: int = 1,
    max_iters: int = 100,
) -> np.ndarray:
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).

    Reference: Zhang, Z.-M.; Tong, S.; Chen, Y.; Liang, X.-Z. *Baseline
    correction for infrared spectroscopy by adaptive iteratively
    reweighted penalized least squares*, **Analyst 2010, 135, 1138-1146**.

    The baseline of each spectrum is estimated iteratively: positive
    residuals (signal above baseline) are given zero weight, while negative
    residuals are weighted by a magnitude-dependent exponential. A
    second-order-difference smoothness penalty keeps the baseline smooth.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or a single spectrum
            ``(n_wavelengths,)``.
        lambda_: Smoothness penalty (larger -> smoother baseline).
        porder: Deprecated, retained for API compatibility (no effect on the
            computation).
        max_iters: Maximum number of reweighting iterations per spectrum.

    Returns:
        Baseline-corrected spectra ``X - baseline`` with the same shape as
        ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        base = _airpls_baseline(arr[i], lambda_, porder, max_iters)
        out[i] = arr[i] - base
    return out[0] if was_1d else out


def asls(
    X: np.ndarray, lambda_: float = 1e5, p: float = 0.001
) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens 2005).

    Iteratively fits a smooth baseline where points above the current
    baseline are down-weighted by ``p`` and points below by ``1 - p``
    (so the baseline hugs the lower envelope of the signal).

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        lambda_: Smoothness penalty (larger -> smoother baseline).
        p: Asymmetry parameter, ``0 < p < 1``. Small ``p`` (e.g. 0.001)
            makes the baseline track the lower envelope.

    Returns:
        Baseline-corrected spectra ``X - baseline`` with the same shape.
    """
    arr, was_1d = _ensure_2d(X)
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        base = _asls_baseline(arr[i], lambda_, p, max_iters=20)
        out[i] = arr[i] - base
    return out[0] if was_1d else out


def detrend(X: np.ndarray, wv: np.ndarray | None = None) -> np.ndarray:
    """Quadratic-polynomial detrending.

    For each spectrum fits ``x = c0 + c1 * t + c2 * t**2`` where ``t`` is the
    wavelength axis (or column index when ``wv`` is ``None``) and returns the
    residual ``x - fit``.

    Args:
        X: Input spectra ``(n_samples, n_wavelengths)`` or single spectrum
            ``(n_wavelengths,)``.
        wv: Wavelength values of shape ``(n_wavelengths,)``. If ``None`` the
            integer column indices ``0..n_wavelengths-1`` are used.

    Returns:
        Detrended spectra with the same shape as ``X``.
    """
    arr, was_1d = _ensure_2d(X)
    n_samples, n_wavelengths = arr.shape
    if wv is None:
        t = np.arange(n_wavelengths, dtype=float)
    else:
        t = np.asarray(wv, dtype=float).ravel()
        if t.shape[0] != n_wavelengths:
            raise ValueError(
                f"wv length {t.shape[0]} does not match n_wavelengths "
                f"{n_wavelengths}."
            )
    # Build design matrix with columns [1, t, t**2].
    # Normalize t to avoid ill-conditioning for large wavelength values.
    t_norm = (t - t.mean()) / (t.std() if t.std() > 0 else 1.0)
    A = np.column_stack([np.ones(n_wavelengths), t_norm, t_norm ** 2])
    # Least-squares fit for all rows at once: coeffs shape (3, n_samples).
    coeffs, *_ = np.linalg.lstsq(A, arr.T, rcond=None)
    fit = (A @ coeffs).T  # (n_samples, n_wavelengths)
    out = arr - fit
    return out[0] if was_1d else out
