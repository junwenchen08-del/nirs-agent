"""Tests for :mod:`nir_core.io.loaders.load_csv`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nir_core.io.loaders import load_csv
from nir_core.io.writers import save_csv
from nir_core.models import SpectralData


def test_load_csv_roundtrip_no_y_no_wv(tmp_path: Path, synthetic_data: SpectralData) -> None:
    """Round-trip a spectra-only CSV and verify shapes/values."""
    src = synthetic_data
    # Save without y / wv to exercise the plain-matrix path.
    data = SpectralData(
        X=src.X[:10],
        y=None,
        wv=None,
        sample_names=[],
        source_file="",
        original_format="",
    )
    fp = tmp_path / "plain.csv"
    save_csv(data, str(fp))

    loaded = load_csv(str(fp))
    assert loaded.original_format == "csv"
    assert loaded.X.shape == data.X.shape
    assert loaded.y is None
    assert loaded.wv is None
    np.testing.assert_allclose(loaded.X, data.X)


def test_load_csv_roundtrip_with_y_and_wv(
    tmp_path: Path, synthetic_data: SpectralData
) -> None:
    """Round-trip a full CSV (y + wv) and verify all three arrays."""
    src = synthetic_data
    data = SpectralData(
        X=src.X[:20],
        y=src.y[:20] if src.y is not None else None,
        wv=src.wv,
        sample_names=[],
        source_file="",
        original_format="",
    )
    fp = tmp_path / "full.csv"
    save_csv(data, str(fp))

    loaded = load_csv(str(fp), y_col=0, wv_row=0)
    assert loaded.X.shape == data.X.shape
    assert loaded.y is not None and loaded.y.shape == data.y.shape
    assert loaded.wv is not None and loaded.wv.shape == data.wv.shape
    np.testing.assert_allclose(loaded.X, data.X, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(loaded.y, data.y, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(loaded.wv, data.wv, rtol=1e-6, atol=1e-8)


def test_load_csv_missing_file(tmp_path: Path) -> None:
    """A missing file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_csv(str(tmp_path / "does_not_exist.csv"))


def test_load_csv_auto_delimiter_tab(tmp_path: Path) -> None:
    """Tab-separated files should be auto-detected."""
    X = np.arange(12, dtype=float).reshape(3, 4)
    fp = tmp_path / "tab.txt"
    with open(fp, "w", encoding="utf-8") as fh:
        for row in X:
            fh.write("\t".join(f"{v:g}" for v in row))
            fh.write("\n")
    loaded = load_csv(str(fp))
    assert loaded.X.shape == (3, 4)
    np.testing.assert_allclose(loaded.X, X)
