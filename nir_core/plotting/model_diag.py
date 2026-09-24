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
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
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
    axes[0].scatter(
        y_pred, residuals, s=30, alpha=0.7, edgecolors="none", c="steelblue"
    )
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


def _multi_plot_inputs(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    names: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ref = np.asarray(y_ref, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if ref.ndim != 2 or pred.ndim != 2:
        raise ValueError("y_ref and y_pred must both be 2D")
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_ref {ref.shape}, y_pred {pred.shape}")
    target_names = names or [f"y{i}" for i in range(ref.shape[1])]
    if len(target_names) != ref.shape[1]:
        raise ValueError("names length must match the number of targets")
    return ref, pred, target_names


def plot_multi_predicted_vs_reference(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    names: list[str] | None = None,
) -> str:
    """Plot predicted-versus-reference panels for multiple components."""
    ref, pred, target_names = _multi_plot_inputs(y_ref, y_pred, names)
    n_targets = ref.shape[1]
    n_cols = min(2, n_targets)
    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )
    for index, ax in enumerate(axes.ravel()):
        if index >= n_targets:
            ax.set_visible(False)
            continue
        reference = ref[:, index]
        prediction = pred[:, index]
        metrics = _compute_metrics(reference, prediction)
        all_values = np.concatenate([reference, prediction])
        lo, hi = float(np.min(all_values)), float(np.max(all_values))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        limits = (lo - pad, hi + pad)
        ax.plot(limits, limits, "k--", linewidth=1.0)
        ax.scatter(
            reference, prediction, s=24, alpha=0.75, edgecolors="none", c="steelblue"
        )
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel="Reference",
            ylabel="Predicted",
            title=target_names[index],
        )
        ax.text(
            0.04,
            0.96,
            f"R² = {metrics['R2']:.4f}\nRMSEP = {metrics['RMSEP']:.4f}\nRPD = {metrics['RPD']:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
        ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return _fig_to_base64(fig)


def plot_multi_residuals(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    names: list[str] | None = None,
) -> str:
    """Plot residual-versus-predicted panels for multiple components."""
    ref, pred, target_names = _multi_plot_inputs(y_ref, y_pred, names)
    n_targets = ref.shape[1]
    n_cols = min(2, n_targets)
    n_rows = int(np.ceil(n_targets / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False
    )
    for index, ax in enumerate(axes.ravel()):
        if index >= n_targets:
            ax.set_visible(False)
            continue
        residuals = pred[:, index] - ref[:, index]
        ax.scatter(
            pred[:, index],
            residuals,
            s=24,
            alpha=0.75,
            edgecolors="none",
            c="steelblue",
        )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
        ax.set(
            xlabel="Predicted",
            ylabel="Residual (pred - ref)",
            title=target_names[index],
        )
        ax.grid(True, linestyle="--", alpha=0.35)
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


def plot_vip(
    vip_scores: np.ndarray,
    wv: np.ndarray | None = None,
    threshold: float = 1.0,
    top_n: int = 20,
) -> str:
    """Plot Variable Importance in Projection (VIP) scores vs wavelength.

    VIP scores > 1.0 indicate above-average contribution to the PLS model.
    The top-N most important wavelengths are highlighted.

    Args:
        vip_scores: 1-D array of VIP scores (n_wavelengths,).
        wv: Optional wavelength axis. If None, column indices are used.
        threshold: VIP threshold line (default 1.0).
        top_n: Number of top wavelengths to highlight with markers.

    Returns:
        Base64-encoded PNG string.
    """
    vip_scores = np.asarray(vip_scores, dtype=float).ravel()
    n_wv = vip_scores.size

    if wv is not None:
        x = np.asarray(wv, dtype=float).ravel()
        if x.size != n_wv:
            x = np.arange(n_wv, dtype=float)
        x_label = "Wavelength (nm)"
    else:
        x = np.arange(n_wv, dtype=float)
        x_label = "Variable index"

    fig, ax = plt.subplots(figsize=(10, 5))

    # VIP curve.
    ax.plot(x, vip_scores, color="steelblue", linewidth=1.0, alpha=0.8)

    # Fill above threshold.
    above = np.where(vip_scores >= threshold, vip_scores, threshold)
    ax.fill_between(x, threshold, above, color="steelblue", alpha=0.2)

    # Threshold line.
    ax.axhline(
        threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"Threshold = {threshold}",
    )

    # Highlight top-N variables.
    top_n = min(top_n, n_wv)
    if top_n > 0:
        top_idx = np.argsort(vip_scores)[-top_n:]
        ax.scatter(
            x[top_idx],
            vip_scores[top_idx],
            color="crimson",
            s=30,
            zorder=5,
            label=f"Top {top_n} variables",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("VIP score")
    ax.set_title("Variable Importance in Projection (VIP)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    return _fig_to_base64(fig)


def plot_regression_coefficients(
    coefficients: np.ndarray,
    wv: np.ndarray | None = None,
) -> str:
    """Plot PLS regression coefficients vs wavelength.

    Positive coefficients indicate a positive contribution to the prediction;
    negative coefficients indicate a negative contribution.

    Args:
        coefficients: 1-D array of regression coefficients (n_wavelengths,).
        wv: Optional wavelength axis. If None, column indices are used.

    Returns:
        Base64-encoded PNG string.
    """
    coef = np.asarray(coefficients, dtype=float).ravel()
    n_wv = coef.size

    if wv is not None:
        x = np.asarray(wv, dtype=float).ravel()
        if x.size != n_wv:
            x = np.arange(n_wv, dtype=float)
        x_label = "Wavelength (nm)"
    else:
        x = np.arange(n_wv, dtype=float)
        x_label = "Variable index"

    fig, ax = plt.subplots(figsize=(10, 5))

    # Positive coefficients in blue, negative in red.
    ax.bar(
        x,
        coef,
        color=np.where(coef >= 0, "steelblue", "crimson"),
        edgecolor="none",
        width=(x[1] - x[0]) if n_wv > 1 else 1.0,
        alpha=0.8,
    )

    ax.axhline(0.0, color="k", linewidth=0.8)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Regression coefficient")
    ax.set_title("PLS Regression Coefficients", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    return _fig_to_base64(fig)
