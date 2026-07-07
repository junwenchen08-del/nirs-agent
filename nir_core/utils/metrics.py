"""Calibration quality metrics and quality-gate evaluation.

All metric functions accept numpy arrays and return floats. ``compute_metrics``
returns a dict suitable for ``ModelResult.metrics``. ``evaluate_quality``
maps a metrics dict onto domain-aware thresholds (via ``NirConfig``) and
produces a grade plus an action recommendation.
"""

from __future__ import annotations

import numpy as np

from nir_core.config import get_nir_config


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error.

    Args:
        y_true: Reference values, shape (n,).
        y_pred: Predicted values, shape (n,).

    Returns:
        RMSE as a non-negative float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2.

    Uses the manual formula ``1 - SS_res / SS_tot`` to avoid the sklearn
    dependency in this pure-metric module. Returns 0.0 when ``SS_tot == 0``
    (constant reference) to avoid division by zero.

    Args:
        y_true: Reference values.
        y_pred: Predicted values.

    Returns:
        R^2 float (may be negative for poor predictions).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rpd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Ratio of Performance to Deviation = std(y_true) / RMSE.

    Returns ``inf`` when RMSE is 0 (perfect prediction) and 0.0 when the
    reference standard deviation is 0.

    Args:
        y_true: Reference values.
        y_pred: Predicted values.

    Returns:
        RPD float (higher is better).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    std_y = float(np.std(y_true, ddof=1)) if y_true.size > 1 else 0.0
    err = rmse(y_true, y_pred)
    if err == 0.0:
        return float("inf") if std_y > 0 else 0.0
    if std_y == 0.0:
        return 0.0
    return std_y / err


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Systematic bias = mean(y_pred - y_true).

    Args:
        y_true: Reference values.
        y_pred: Predicted values.

    Returns:
        Bias float (0.0 = unbiased).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_pred - y_true))


def slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Least-squares regression slope of y_pred on y_true.

    Fits ``y_pred = slope * y_true + intercept``. A slope of 1.0 indicates
    no multiplicative systematic error.

    Args:
        y_true: Reference values (independent variable).
        y_pred: Predicted values (dependent variable).

    Returns:
        Slope float. Returns 0.0 when y_true has zero variance.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))) / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error.

    Args:
        y_true: Reference values.
        y_pred: Predicted values.

    Returns:
        MAE as a non-negative float.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute the full standard metric set used by the NIR agent.

    Args:
        y_true: Reference values.
        y_pred: Predicted values.

    Returns:
        Dict with keys ``{"RMSE", "R2", "RPD", "bias", "slope", "MAE"}``.
    """
    return {
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "RPD": rpd(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "slope": slope(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
    }


def _extract_r2(metrics: dict) -> float | None:
    """Pull R2 from a metrics dict, tolerating ``R2_val`` or ``R2`` keys."""
    for key in ("R2_val", "R2"):
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    # Nested under "test" sub-dict (some callers pass test-set metrics there).
    test_block = metrics.get("test")
    if isinstance(test_block, dict):
        for key in ("R2_val", "R2"):
            val = test_block.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
    return None


def _extract_rpd(metrics: dict) -> float | None:
    """Pull RPD from a metrics dict."""
    for key in ("RPD", "RPD_val"):
        val = metrics.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    test_block = metrics.get("test")
    if isinstance(test_block, dict):
        for key in ("RPD", "RPD_val"):
            val = test_block.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None
    return None


def _extract_bias(metrics: dict) -> float:
    """Pull bias (default 0.0) from a metrics dict, preferring ``test.bias``."""
    test_block = metrics.get("test")
    if isinstance(test_block, dict) and test_block.get("bias") is not None:
        try:
            return float(test_block["bias"])
        except (TypeError, ValueError):
            pass
    if metrics.get("bias") is not None:
        try:
            return float(metrics["bias"])
        except (TypeError, ValueError):
            pass
    return 0.0


def evaluate_quality(
    metrics: dict,
    domain: str = "default",
    n_samples: int | None = None,
) -> dict:
    """Evaluate model quality against domain-aware thresholds.

    The ``metrics`` dict may use either ``R2_val`` or ``R2`` for the
    coefficient of determination (``R2_val`` wins when both are present)
    and either ``RMSEP`` or ``RMSE`` for the test error. ``RMSECV`` is
    optional and used only for overfitting detection.

    Grading:
        - excellent: r2 >= min_r2 + 0.10 AND rpd >= min_rpd + 1.0
        - good:      r2 >= min_r2        AND rpd >= min_rpd
        - fair:      r2 >= min_r2 - 0.20 AND rpd >= min_rpd - 1.5
        - poor:      otherwise

    Overfitting is flagged when ``RMSECV`` is present and
    ``RMSEP > 2 * RMSECV``. Bias issue is flagged when the test bias is
    non-zero and ``abs(bias) > 0.1 * abs(reference_central_value)``; the
    reference central value defaults to the mean of ``y`` if provided,
    otherwise 1.0 (so the threshold becomes 0.1 absolute).

    Args:
        metrics: Metric dict, possibly nested (e.g. ``test.bias``).
        domain: Application domain key (e.g. ``"soil"``, ``"food_moisture"``).
        n_samples: Number of calibration samples; small (<100) relaxes
            thresholds via ``QualityThresholds.get_thresholds``.

    Returns:
        Dict with keys:
        - ``grade``: one of ``{"excellent","good","fair","poor"}``.
        - ``passed``: True when grade is excellent or good.
        - ``recommendation``: Chinese recommendation string.
        - ``action``: one of ``{"proceed","retry_preprocessing",
          "investigate_data"}``.
        - ``thresholds_used``: The thresholds dict applied.
        - ``details``: ``{R2_status, RPD_status, overfitting_risk,
          bias_issue}``.
    """
    cfg = get_nir_config()
    thresholds = cfg.quality.get_thresholds(domain=domain, n_samples=n_samples)
    min_r2 = float(thresholds["min_r2"])
    min_rpd = float(thresholds["min_rpd"])

    r2 = _extract_r2(metrics)
    rpd_val = _extract_rpd(metrics)

    # R2 / RPD status strings.
    if r2 is None:
        r2_status = "missing"
    elif r2 >= min_r2 + 0.10:
        r2_status = "excellent"
    elif r2 >= min_r2:
        r2_status = "good"
    elif r2 >= min_r2 - 0.20:
        r2_status = "fair"
    else:
        r2_status = "poor"

    if rpd_val is None:
        rpd_status = "missing"
    elif rpd_val >= min_rpd + 1.0:
        rpd_status = "excellent"
    elif rpd_val >= min_rpd:
        rpd_status = "good"
    elif rpd_val >= min_rpd - 1.5:
        rpd_status = "fair"
    else:
        rpd_status = "poor"

    # Grade: take the worse of the two, but require both to meet a tier.
    def _tier(status: str) -> int:
        return {"excellent": 4, "good": 3, "fair": 2, "poor": 1, "missing": 0}.get(
            status, 0
        )

    if r2 is None or rpd_val is None:
        grade = "poor"
    else:
        grade_tier = min(_tier(r2_status), _tier(rpd_status))
        grade = {4: "excellent", 3: "good", 2: "fair", 1: "poor"}.get(grade_tier, "poor")

    # Overfitting detection.
    rmsep = metrics.get("RMSEP")
    if rmsep is None:
        rmsep = metrics.get("RMSE")
    rmsecv = metrics.get("RMSECV")
    overfitting = False
    if rmsep is not None and rmsecv is not None:
        try:
            rmsep_f = float(rmsep)
            rmsecv_f = float(rmsecv)
            if rmsecv_f > 0 and rmsep_f > 2.0 * rmsecv_f:
                overfitting = True
        except (TypeError, ValueError):
            overfitting = False

    # Bias issue detection (conservative; never flags a zero bias).
    test_bias = _extract_bias(metrics)
    y_vals = metrics.get("y")
    if y_vals is not None:
        try:
            ref_central = float(np.mean(np.asarray(y_vals, dtype=float)))
        except (TypeError, ValueError):
            ref_central = 1.0
    else:
        ref_central = 1.0
    if ref_central == 0.0:
        ref_central = 1.0
    bias_issue = (test_bias != 0.0) and (abs(test_bias) > 0.1 * abs(ref_central))

    # Action mapping.
    if grade in ("excellent", "good") and not overfitting:
        action = "proceed"
    elif overfitting:
        action = "retry_preprocessing"
    elif grade == "fair":
        action = "retry_preprocessing"
    else:
        action = "investigate_data"

    # Chinese recommendations.
    rec_map = {
        "excellent": "模型质量优秀，R²与RPD均显著超过阈值，可直接进入下一阶段。",
        "good": "模型质量达标，可继续后续流程；建议关注 bias 与过拟合风险。",
        "fair": "模型质量处于临界水平，建议更换预处理组合后重试。",
        "poor": "模型质量不达标，建议检查数据质量、异常值与划分方式。",
    }
    recommendation = rec_map[grade]
    if overfitting:
        recommendation += " 检测到过拟合风险(RMSEP>2*RMSECV)，建议简化模型或增加样本。"
    if bias_issue:
        recommendation += " 检测到显著系统偏差，建议检查基准校正或仪器漂移。"

    return {
        "grade": grade,
        "passed": grade in ("excellent", "good"),
        "recommendation": recommendation,
        "action": action,
        "thresholds_used": thresholds,
        "details": {
            "R2_status": r2_status,
            "RPD_status": rpd_status,
            "overfitting_risk": overfitting,
            "bias_issue": bias_issue,
        },
    }
