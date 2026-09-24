"""Spectral plotting utilities.

Renders raw and preprocessed NIR spectra as base64-encoded PNG strings using
a headless matplotlib backend (Agg) so plots can be generated in server
environments without a display.
"""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # headless backend, no display required
import matplotlib.pyplot as plt
import numpy as np

from nir_core.models import SpectralData


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


def _wavelength_axis(data: SpectralData) -> np.ndarray:
    """Return the x-axis values for plotting (wavelengths or column indices)."""
    if data.wv is not None:
        return np.asarray(data.wv, dtype=float)
    return np.arange(data.X.shape[1], dtype=float)


def plot_raw_spectra(data: SpectralData, n_highlight: int = 5) -> str:
    """Plot raw NIR spectra with a few highlighted samples.

    All spectra are drawn in light gray; the first ``n_highlight`` samples are
    overlaid in distinct colors from a colormap for visual reference.

    Args:
        data: SpectralData containing the spectra to plot.
        n_highlight: Number of spectra to highlight with distinct colors.

    Returns:
        Base64-encoded PNG string of the figure.
    """
    X = np.asarray(data.X)
    n_samples = X.shape[0]
    x = _wavelength_axis(data)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Background: all spectra in light gray.
    ax.plot(x, X.T, color="lightgray", linewidth=0.6, alpha=0.8)

    # Highlight: pick up to n_highlight samples using a colormap.
    n_highlight = max(0, min(n_highlight, n_samples))
    if n_highlight > 0:
        cmap = plt.get_cmap("tab10")
        highlight_idx = np.arange(n_highlight)
        for i, idx in enumerate(highlight_idx):
            ax.plot(
                x,
                X[idx],
                color=cmap(i % 10),
                linewidth=1.3,
                label=f"Sample {idx}",
            )
        ax.legend(loc="best", fontsize=8, ncol=min(n_highlight, 5))

    ax.set_title("Raw Spectra", fontsize=13)
    ax.set_xlabel("Wavelength (nm)" if data.wv is not None else "Column index")
    ax.set_ylabel("Absorbance")
    ax.grid(True, linestyle="--", alpha=0.4)

    return _fig_to_base64(fig)


def plot_preprocessed_comparison(
    raw: SpectralData,
    processed: SpectralData,
    wv_range: tuple[float, float] | None = None,
) -> str:
    """Compare raw and preprocessed spectra in stacked subplots.

    Args:
        raw: Original SpectralData.
        processed: Preprocessed SpectralData (same number of wavelengths).
        wv_range: Optional (min, max) wavelength range to restrict the x-axis.
            Applied to both subplots.

    Returns:
        Base64-encoded PNG string of the figure.
    """
    x_raw = _wavelength_axis(raw)
    x_proc = _wavelength_axis(processed)
    X_raw = np.asarray(raw.X)
    X_proc = np.asarray(processed.X)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=False)

    # Top: raw spectra.
    axes[0].plot(x_raw, X_raw.T, color="lightgray", linewidth=0.6)
    axes[0].set_title("Raw Spectra", fontsize=12)
    axes[0].set_ylabel("Absorbance")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Bottom: preprocessed spectra.
    axes[1].plot(x_proc, X_proc.T, color="steelblue", linewidth=0.6, alpha=0.7)
    axes[1].set_title("Preprocessed Spectra", fontsize=12)
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Absorbance")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # Apply wavelength range restriction if requested.
    if wv_range is not None:
        lo, hi = float(wv_range[0]), float(wv_range[1])
        axes[0].set_xlim(lo, hi)
        axes[1].set_xlim(lo, hi)

    fig.tight_layout()
    return _fig_to_base64(fig)
