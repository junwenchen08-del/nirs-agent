"""Drift detection utilities for spectral monitoring.

Provides Mahalanobis-distance based drift detection between a training
reference set and a new (production) batch, plus a composite drift index
derived from a ``ModelResult``.
"""

from __future__ import annotations

import numpy as np

from nir_core.models import ModelResult


def compute_mahalanobis_drift(
    X_train: np.ndarray,
    X_new: np.ndarray,
    threshold: float = 3.0,
) -> dict:
    """Compute Mahalanobis-distance drift for new samples relative to train.

    Estimates the mean ``mu`` and covariance ``Sigma`` from ``X_train``,
    then for each new sample computes
    ``d_i = sqrt((x_i - mu)^T Sigma^+ (x_i - mu))`` where ``Sigma^+`` is
    the Moore-Penrose pseudo-inverse (stable for singular / high-dim
    covariance). Samples with ``d_i > threshold`` are flagged.

    The heatmap contribution per wavelength is approximated by the
    squared centered residual weighted by the pseudo-inverse diagonal,
    giving a per-wavelength contribution to the T^2 statistic. The
    returned ``heatmap_data`` has shape ``(n_new, n_wavelengths)``.

    Args:
        X_train: Reference spectral matrix, shape (n_train, n_wavelengths).
        X_new: New spectral matrix, shape (n_new, n_wavelengths).
        threshold: Mahalanobis distance threshold for flagging.

    Returns:
        Dict with keys:
        - ``distances``: ndarray shape (n_new,).
        - ``flagged_indices``: int ndarray of flagged sample indices.
        - ``drift_score``: float fraction of flagged samples in [0, 1].
        - ``heatmap_data``: ndarray shape (n_new, n_wavelengths) of
          per-wavelength T^2 contributions.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    if X_train.ndim != 2 or X_new.ndim != 2:
        raise ValueError("X_train and X_new must be 2-D arrays.")
    if X_train.shape[1] != X_new.shape[1]:
        raise ValueError(
            "X_train and X_new must share the wavelength dimension."
        )
    n_new = X_new.shape[0]
    if n_new == 0:
        return {
            "distances": np.array([], dtype=float),
            "flagged_indices": np.array([], dtype=int),
            "drift_score": 0.0,
            "heatmap_data": np.empty((0, X_train.shape[1]), dtype=float),
        }

    mu = np.mean(X_train, axis=0)
    centered_train = X_train - mu
    cov = np.cov(centered_train, rowvar=False)
    cov = np.atleast_2d(cov)
    cov_inv = np.linalg.pinv(cov)

    diff = X_new - mu  # (n_new, n_wavelengths)
    # Per-sample Mahalanobis distance.
    mahal_sq = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
    mahal_sq = np.maximum(mahal_sq, 0.0)
    distances = np.sqrt(mahal_sq)

    flagged_indices = np.where(distances > threshold)[0].astype(int)
    drift_score = float(flagged_indices.size / n_new) if n_new > 0 else 0.0

    # Per-wavelength contribution to T^2: d_i^2 contribution from each
    # wavelength j is diff_ij * (cov_inv @ diff_i)_j. Shape (n_new, n_w).
    weighted = diff @ cov_inv  # (n_new, n_wavelengths)
    heatmap_data = diff * weighted  # element-wise; sum over j == mahal_sq

    return {
        "distances": distances,
        "flagged_indices": flagged_indices,
        "drift_score": drift_score,
        "heatmap_data": heatmap_data,
    }


def compute_drift_index(model_result: ModelResult, X_new: np.ndarray) -> float:
    """Composite drift index in [0, 1] for a new batch.

    A simple, robust implementation: if the model's metrics contain a
    reference RPD (used to scale the acceptable deviation), the index is
    ``min(1, mean_mahal / (3 * RPD))`` where ``mean_mahal`` is the mean
    Mahalanobis distance of ``X_new`` from its own centroid. When RPD is
    unavailable the index falls back to ``min(1, mean_mahal / 9.0)``.

    A value above 0.3 suggests re-calibration is advisable.

    Args:
        model_result: A ``ModelResult`` whose ``metrics`` may carry RPD.
        X_new: New spectral matrix, shape (n_new, n_wavelengths).

    Returns:
        Drift index float in [0, 1].
    """
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2 or X_new.shape[0] == 0:
        return 0.0

    mu = np.mean(X_new, axis=0)
    diff = X_new - mu
    cov = np.cov(diff, rowvar=False)
    cov = np.atleast_2d(cov)
    cov_inv = np.linalg.pinv(cov)
    mahal_sq = np.maximum(
        np.einsum("ij,jk,ik->i", diff, cov_inv, diff), 0.0
    )
    mean_mahal = float(np.mean(np.sqrt(mahal_sq)))

    metrics = getattr(model_result, "metrics", {}) or {}
    rpd_val = None
    for key in ("RPD", "RPD_val"):
        v = metrics.get(key)
        if v is not None:
            try:
                rpd_val = float(v)
            except (TypeError, ValueError):
                rpd_val = None
                continue
            break

    if rpd_val is not None and rpd_val > 0:
        denom = 3.0 * rpd_val
    else:
        denom = 9.0
    if denom <= 0:
        return 0.0
    return float(min(1.0, mean_mahal / denom))
