"""Tests for nir_core.plotting.spectra."""

from __future__ import annotations

import base64

import numpy as np

from nir_core.models import SpectralData
from nir_core.plotting.spectra import plot_preprocessed_comparison, plot_raw_spectra
from nir_core.tests.generators import generate_synthetic_spectra


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png_b64(b64_str: str) -> None:
    """Decode a base64 string and assert the bytes start with the PNG magic."""
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0
    raw = base64.b64decode(b64_str)
    assert raw[:8] == PNG_MAGIC, "Decoded bytes are not a PNG."


def test_plot_raw_spectra_returns_png(synthetic_data: SpectralData) -> None:
    """plot_raw_spectra should return a non-empty base64 PNG string."""
    b64 = plot_raw_spectra(synthetic_data, n_highlight=5)
    _assert_valid_png_b64(b64)


def test_plot_raw_spectra_highlight_more_than_samples() -> None:
    """n_highlight larger than n_samples should not raise."""
    data = generate_synthetic_spectra(n_samples=10, n_wavelengths=50, random_state=1)
    b64 = plot_raw_spectra(data, n_highlight=100)
    _assert_valid_png_b64(b64)


def test_plot_raw_spectra_without_wv() -> None:
    """plot_raw_spectra should work when wv is None (column index axis)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 40))
    data = SpectralData(X=X, wv=None)
    b64 = plot_raw_spectra(data, n_highlight=3)
    _assert_valid_png_b64(b64)


def test_plot_preprocessed_comparison_returns_png(
    synthetic_data: SpectralData,
) -> None:
    """plot_preprocessed_comparison returns a valid base64 PNG."""
    # Construct a "processed" version by mean-centering the raw spectra.
    raw = synthetic_data
    processed = SpectralData(
        X=raw.X - raw.X.mean(axis=0, keepdims=True),
        y=raw.y,
        wv=raw.wv,
        sample_names=raw.sample_names,
        source_file=raw.source_file,
        original_format=raw.original_format,
    )
    b64 = plot_preprocessed_comparison(raw, processed)
    _assert_valid_png_b64(b64)


def test_plot_preprocessed_comparison_with_wv_range(
    synthetic_data: SpectralData,
) -> None:
    """wv_range restriction should not raise and should return a valid PNG."""
    raw = synthetic_data
    processed = SpectralData(
        X=raw.X * 1.0 - raw.X.mean(axis=0, keepdims=True),
        y=raw.y,
        wv=raw.wv,
    )
    lo, hi = float(raw.wv.min()), float(raw.wv.max())
    mid = (lo + hi) / 2.0
    b64 = plot_preprocessed_comparison(raw, processed, wv_range=(lo, mid))
    _assert_valid_png_b64(b64)


def test_plot_preprocessed_comparison_different_params() -> None:
    """Comparison of spectra generated with different params should work."""
    raw = generate_synthetic_spectra(
        n_samples=30, n_wavelengths=100, n_components=2, noise_level=0.02, random_state=1
    )
    processed = generate_synthetic_spectra(
        n_samples=30, n_wavelengths=100, n_components=2, noise_level=0.05, random_state=2
    )
    b64 = plot_preprocessed_comparison(raw, processed)
    _assert_valid_png_b64(b64)
