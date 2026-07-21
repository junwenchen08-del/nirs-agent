"""Drift detection utilities for spectral monitoring.

Provides Mahalanobis-distance based drift detection between a training
reference set and a new (production) batch, plus a composite drift index
derived from a ``ModelResult``.
"""

from __future__ import annotations

import numpy as np

from nir_core.models import ModelResult


def fit_monitoring_reference(
    X_train: np.ndarray,
    *,
    explained_variance: float = 0.99,
    max_components: int = 20,
    limit_quantile: float = 0.99,
) -> dict:
    """Fit a compact PCA applicability-domain reference in model space."""
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2 or X_train.shape[0] < 3 or X_train.shape[1] < 1:
        raise ValueError("X_train must contain at least 3 rows and 1 feature")
    if not 0.5 <= explained_variance <= 1.0:
        raise ValueError("explained_variance must be in [0.5, 1.0]")
    if not 0.5 < limit_quantile < 1.0:
        raise ValueError("limit_quantile must be in (0.5, 1.0)")

    mean = np.mean(X_train, axis=0)
    centered = X_train - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    variances = singular_values**2
    total = float(np.sum(variances))
    if total <= np.finfo(float).eps:
        n_components = 1
    else:
        cumulative = np.cumsum(variances) / total
        n_components = int(np.searchsorted(cumulative, explained_variance) + 1)
    n_components = max(
        1,
        min(n_components, int(max_components), X_train.shape[0] - 1, X_train.shape[1]),
    )
    components = vt[:n_components]
    scores = centered @ components.T
    score_covariance = np.atleast_2d(np.cov(scores, rowvar=False))
    score_covariance_inv = np.linalg.pinv(score_covariance)
    t2 = np.maximum(np.einsum("ij,jk,ik->i", scores, score_covariance_inv, scores), 0.0)
    residual = centered - scores @ components
    q_residual = np.sum(residual**2, axis=1)
    epsilon = float(np.finfo(float).eps)
    return {
        "schema_version": 1,
        "method": "pca_t2_q",
        "n_train": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "n_components": int(n_components),
        "mean": mean,
        "components": components,
        "score_covariance_inv": score_covariance_inv,
        "t2_limit": max(float(np.quantile(t2, limit_quantile)), epsilon),
        "q_limit": max(float(np.quantile(q_residual, limit_quantile)), epsilon),
        "limit_quantile": float(limit_quantile),
        "explained_variance_retained": float(np.sum(variances[:n_components]) / total)
        if total > epsilon
        else 1.0,
    }


def compute_reference_drift(reference: dict, X_new: np.ndarray) -> dict:
    """Score new spectra against a fitted PCA T²/Q reference."""
    if not isinstance(reference, dict) or reference.get("schema_version") != 1:
        raise ValueError("Unsupported monitoring reference")
    X_new = np.asarray(X_new, dtype=float)
    if X_new.ndim != 2:
        raise ValueError("X_new must be a 2-D array")
    expected_features = int(reference["n_features"])
    if X_new.shape[1] != expected_features:
        raise ValueError(
            f"X_new has {X_new.shape[1]} features; monitoring reference expects {expected_features}"
        )

    mean = np.asarray(reference["mean"], dtype=float)
    components = np.asarray(reference["components"], dtype=float)
    covariance_inv = np.asarray(reference["score_covariance_inv"], dtype=float)
    centered = X_new - mean
    scores = centered @ components.T
    t2 = np.maximum(np.einsum("ij,jk,ik->i", scores, covariance_inv, scores), 0.0)
    residual = centered - scores @ components
    q_residual = np.sum(residual**2, axis=1)
    t2_flags = t2 > float(reference["t2_limit"])
    q_flags = q_residual > float(reference["q_limit"])
    flagged = t2_flags | q_flags
    return {
        "method": "pca_t2_q",
        "available": True,
        "n_samples": int(X_new.shape[0]),
        "drift_score": float(np.mean(flagged)) if flagged.size else 0.0,
        "flagged_indices": np.flatnonzero(flagged).astype(int),
        "t2_flagged_indices": np.flatnonzero(t2_flags).astype(int),
        "q_flagged_indices": np.flatnonzero(q_flags).astype(int),
        "t2": t2,
        "q_residual": q_residual,
        "t2_limit": float(reference["t2_limit"]),
        "q_limit": float(reference["q_limit"]),
    }


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
        raise ValueError("X_train and X_new must share the wavelength dimension.")
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
    mahal_sq = np.maximum(np.einsum("ij,jk,ik->i", diff, cov_inv, diff), 0.0)
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
