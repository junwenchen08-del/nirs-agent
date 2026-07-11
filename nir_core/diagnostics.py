"""Residual diagnostics for NIR model evaluation.

Provides :func:`compute_residual_diagnostics` — a "pathology report" for the
LLM agent to reason about *why* a model underperforms and *what* preprocessing
adjustment to try next.

All thresholds are **relative** (normalised to the scale of ``y_true``) so the
diagnostics work equally well for moisture (0–100 %), protein (5–25 %), or pH
(4–10) without re-tuning.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_residual_diagnostics"]


def compute_residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute residual-based diagnostics for LLM-driven preprocessing decisions.

    Produces three diagnostic signals, all using **relative thresholds** so they
    are scale-invariant:

    1. ``residual_trend`` — Pearson correlation between residuals and
       reference values. Values: ``"upward"`` / ``"downward"`` / ``"flat"``.
       This is **scale-invariant** (correlation is unaffected by linear
       scaling of ``y``).
       - ``"upward"``: predictions are systematically low for high reference
         values → suggests baseline drift → try airPLS / asLS.
       - ``"downward"``: predictions are systematically high for high reference
         values → suggests multiplicative scatter → try SNV / MSC.

    2. ``residual_variance`` — ratio of residual variance to ``var(y_true)``.
       Values: ``"high"`` (ratio > 0.25, model explains < 75 % of variance) /
       ``"low"``. High variance suggests noise → try SG smoothing.

    3. ``outlier_ratio`` — fraction of samples whose absolute residual exceeds
       2× the residual standard deviation. High ratio (> 0.1) suggests
       outlier contamination → try ``remove_outliers=True``.

    Args:
        y_true: Reference values, shape ``(n_samples,)``.
        y_pred: Predicted values, shape ``(n_samples,)``.

    Returns:
        Dict with keys ``residual_trend``, ``residual_variance``,
        ``outlier_ratio``, ``outlier_count``, and ``residual_std``.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    n = y_true.shape[0]
    if n < 2:
        return {
            "residual_trend": "flat",
            "residual_variance": "low",
            "outlier_ratio": 0.0,
            "outlier_count": 0,
            "residual_std": 0.0,
        }

    residuals = y_true - y_pred
    y_std = float(np.std(y_true))
    y_var = float(np.var(y_true))

    # Guard against zero-variance y (all reference values identical).
    if y_std < 1e-12:
        return {
            "residual_trend": "flat",
            "residual_variance": "low" if np.var(residuals) < 1e-12 else "high",
            "outlier_ratio": 0.0,
            "outlier_count": 0,
            "residual_std": float(np.std(residuals)),
        }

    # 1. Residual trend: Pearson correlation between residuals and y_true.
    #    |r| > 0.3 indicates a systematic trend (scale-invariant).
    #    r > 0.3 → upward (residuals grow with y → under-prediction at high y)
    #    r < -0.3 → downward (over-prediction at high y)
    resid_std = float(np.std(residuals))
    if resid_std > 1e-12:
        r_matrix = np.corrcoef(y_true, residuals)
        r = float(r_matrix[0, 1])
    else:
        r = 0.0
    if r > 0.3:
        trend = "upward"
    elif r < -0.3:
        trend = "downward"
    else:
        trend = "flat"

    # 2. Residual variance ratio: var(residuals) / var(y_true).
    #    ratio > 0.25 → high (model explains < 75% of variance)
    var_ratio = float(np.var(residuals)) / y_var if y_var > 1e-12 else 0.0
    variance_level = "high" if var_ratio > 0.25 else "low"

    # 3. Outlier detection: |residual| > 2 * std(residuals).
    if resid_std > 1e-12:
        outlier_mask = np.abs(residuals) > 2.0 * resid_std
        outlier_count = int(np.sum(outlier_mask))
    else:
        outlier_count = 0
    outlier_ratio = outlier_count / n

    return {
        "residual_trend": trend,
        "residual_variance": variance_level,
        "outlier_ratio": round(outlier_ratio, 4),
        "outlier_count": outlier_count,
        "residual_std": round(resid_std, 6),
    }
