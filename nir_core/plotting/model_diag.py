"""Model diagnostic plots.

Provides predicted-vs-reference scatter plots, residual diagnostics, and
drift detection heatmaps. All functions return base64-encoded PNG strings
and run on a headless matplotlib backend.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_base64(fig: plt.Figure) -> str:
    """Serialize a matplotlib figure to a base64-encoded PNG string.

    Closes the figure after serialization to prevent memory leaks.

    Args:
        fig: A matplotlib Figure instance.

    Returns:
        ASCII string of base64-encoded PNG bytes.
    """
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _compute_metrics(y_ref: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute R2, RMSEP, and RPD from reference and predicted arrays."""
    n = y_ref.size
    ss_res = float(np.sum((y_ref - y_pred) ** 2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    rmsep = float(np.sqrt(ss_res / n)) if n > 0 else 0.0
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    std_ref = float(np.std(y_ref)) if n > 0 else 0.0
    rpd = (std_ref / rmsep) if rmsep > 0 else float("inf")
    return {"RMSEP": rmsep, "R2": r2, "RPD": rpd}


def plot_predicted_vs_reference(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs Reference",
) -> str:
    """Scatter plot of predicted vs reference values with 1:1 line and RMSEP band.

    Args:
        y_ref: Reference values (1-D array).
        y_pred: Predicted values (1-D array, same length as y_ref).
        title: Plot title (Chinese or English supported).

    Returns:
        Base64-encoded PNG string.

    Raises:
        ValueError: If y_ref and y_pred have different lengths.
    """
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_ref.size != y_pred.size:
        raise ValueError(
            f"Length mismatch: y_ref has {y_ref.size} elements, "
            f"y_pred has {y_pred.size} elements."
        )

    metrics = _compute_metrics(y_ref, y_pred)
    rmsep = metrics["RMSEP"]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Determine axis limits from combined data.
    all_vals = np.concatenate([y_ref, y_pred])
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lim = (lo - pad, hi + pad)

    # 1:1 reference line.
    ax.plot(lim, lim, "k--", linewidth=1.2, label="1:1 line")

    # ±RMSEP band around the 1:1 line.
    band_x = np.linspace(lim[0], lim[1], 100)
    ax.fill_between(
        band_x,
        band_x - rmsep,
        band_x + rmsep,
        color="orange",
        alpha=0.2,
        label=f"±RMSEP band ({rmsep:.4f})",
    )

    # Scatter of predictions.
    ax.scatter(y_ref, y_pred, s=30, alpha=0.7, edgecolors="none", c="steelblue")

    # Metrics text box.
    text = (
        f"R² = {metrics['R2']:.4f}\n"
        f"RMSEP = {metrics['RMSEP']:.4f}\n"
        f"RPD = {metrics['RPD']:.4f}"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Predicted")
    ax.set_title(title, fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    return _fig_to_base64(fig)


def plot_residuals(y_ref: np.ndarray, y_pred: np.ndarray) -> str:
    """Plot residual diagnostics: residuals vs predicted (left) and histogram (right).

    Args:
        y_ref: Reference values (1-D array).
        y_pred: Predicted values (1-D array, same length as y_ref).

    Returns:
        Base64-encoded PNG string.

    Raises:
        ValueError: If y_ref and y_pred have different lengths.
    """
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_ref.size != y_pred.size:
        raise ValueError(
            f"Length mismatch: y_ref has {y_ref.size} elements, "
            f"y_pred has {y_pred.size} elements."
        )

    residuals = y_pred - y_ref

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))

    # Left: residuals vs predicted.
    axes[0].scatter(y_pred, residuals, s=30, alpha=0.7, edgecolors="none", c="steelblue")
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual (pred - ref)")
    axes[0].set_title("Residuals vs Predicted", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Right: residual histogram.
    axes[1].hist(residuals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Histogram", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_cv_curve(
    n_components: list[int],
    mean_rmse_cv: list[float],
    best_n: int,
    std_rmse_cv: list[float] | None = None,
) -> str:
    """Plot PLS/PCR cross-validation RMSE curve with the selected component.

    Args:
        n_components: List of component counts tested.
        mean_rmse_cv: Mean RMSECV for each component count.
        best_n: Selected component count (highlighted).
        std_rmse_cv: Optional standard deviation of RMSECV (for error bars).

    Returns:
        Base64-encoded PNG string.
    """
    n_components = np.asarray(n_components, dtype=int)
    mean_rmse_cv = np.asarray(mean_rmse_cv, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    if std_rmse_cv is not None:
        std_rmse_cv = np.asarray(std_rmse_cv, dtype=float)
        ax.errorbar(
            n_components,
            mean_rmse_cv,
            yerr=std_rmse_cv,
            fmt="o-",
            color="steelblue",
            capsize=4,
            label="RMSECV",
        )
    else:
        ax.plot(n_components, mean_rmse_cv, "o-", color="steelblue", label="RMSECV")

    ax.axvline(
        best_n,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Best = {best_n}",
    )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("RMSECV")
    ax.set_title("Cross-Validation Component Selection", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    return _fig_to_base64(fig)


def plot_drift_heatmap(
    distances: np.ndarray,
    wv: np.ndarray,
    flagged: np.ndarray,
) -> str:
    """Plot sample drift distances with flagged anomalies highlighted.

    Renders a bar chart of per-sample Mahalanobis distances, with a threshold
    line (mean + 3*std) and flagged anomalies marked in red.

    Args:
        distances: 1-D array of per-sample drift distances.
        wv: Wavelength array (used for axis labeling / context; not plotted
            directly but kept for API symmetry with future heatmap variants).
        flagged: 1-D array of integer sample indices flagged as drift anomalies.

    Returns:
        Base64-encoded PNG string.
    """
    distances = np.asarray(distances, dtype=float).ravel()
    flagged = np.asarray(flagged, dtype=int).ravel()
    n = distances.size

    fig, ax = plt.subplots(figsize=(10, 5))

    sample_idx = np.arange(n)
    colors = np.where(np.isin(sample_idx, flagged), "crimson", "steelblue")

    ax.bar(sample_idx, distances, color=colors, edgecolor="none", alpha=0.85)

    # Threshold line: mean + 3*std (defensive against zero variance).
    mean_d = float(np.mean(distances)) if n > 0 else 0.0
    std_d = float(np.std(distances)) if n > 0 else 0.0
    threshold = mean_d + 3.0 * std_d
    ax.axhline(
        threshold,
        color="darkorange",
        linestyle="--",
        linewidth=1.3,
        label=f"阈值 Threshold (mean+3σ = {threshold:.3f})",
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("Drift Detection", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.legend(loc="best", fontsize=9)

    return _fig_to_base64(fig)
