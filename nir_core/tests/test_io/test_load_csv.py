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


def test_load_csv_roundtrip_with_y_and_wv(tmp_path: Path, synthetic_data: SpectralData) -> None:
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


def test_load_csv_auto_detects_nir_style_labeled_layout(tmp_path: Path) -> None:
    """A CSV with empty corner, wavelength header row, and y column should
    auto-split into X / y / wv without explicit y_col / wv_row.

    Mirrors the corn_moisture.csv layout:
        ,1100,1102,...,2500
        12.5,0.31,0.30,...,0.42
        ...
    """
    wv = np.linspace(1100, 2500, 51)
    rng = np.random.default_rng(42)
    y = np.linspace(5, 25, 20)  # moisture percentages, 5-25%
    X = rng.uniform(0.1, 1.5, size=(20, 51))  # absorbance 0.1-1.5

    fp = tmp_path / "nir_style.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        # Header row: empty corner + wavelengths.
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(y, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp))
    assert loaded.X.shape == (20, 51)
    assert loaded.y is not None and loaded.y.shape == (20,)
    assert loaded.wv is not None and loaded.wv.shape == (51,)
    # ``:g`` formatting in the writer trims to 6 significant digits, so the
    # round-trip precision is bounded by the file format rather than FP64.
    np.testing.assert_allclose(loaded.wv, wv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded.y, y, rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(loaded.X, X, rtol=1e-5, atol=1e-5)


def test_load_csv_auto_layout_disabled(tmp_path: Path) -> None:
    """auto_layout=False keeps prior behaviour: whole block is X, no y/wv."""
    wv = np.linspace(1100, 2500, 51)
    rng = np.random.default_rng(42)
    y = np.linspace(5, 25, 20)
    X = rng.uniform(0.1, 1.5, size=(20, 51))

    fp = tmp_path / "nir_style.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(y, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), auto_layout=False)
    # Block has 21 rows (header + 20) and 52 cols (y col + 51 wv).
    # With auto_layout=False, nothing is split out.
    assert loaded.y is None
    assert loaded.wv is None
    assert loaded.X.shape == (21, 52)


def test_load_csv_auto_layout_rejects_non_nir_first_row(tmp_path: Path) -> None:
    """A first row with values outside the NIR wavelength range should NOT
    trigger auto-layout — protects against false positives on plain
    numeric tables that happen to have an empty corner cell.
    """
    # First row is small integers (0..10), not wavelengths.
    arr = np.arange(11, dtype=float).reshape(1, 11)
    arr[0, 0] = np.nan
    fp = tmp_path / "not_nir.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        for row in arr:
            fh.write(",".join("" if np.isnan(v) else f"{v:g}" for v in row) + "\n")
        # Add a second row so the loader produces 2D.
        fh.write("1," + ",".join("0.5" for _ in range(10)) + "\n")

    loaded = load_csv(str(fp))
    # Auto-layout must refuse; the block stays intact.
    assert loaded.y is None
    assert loaded.wv is None


def test_load_csv_auto_layout_accepts_similar_scale_y_and_spectra(tmp_path: Path) -> None:
    """Reference values and absorbances with similar numeric ranges must
    still be auto-separated when signals 1 (corner NaN) and 2 (NIR
    wavelength row) are confident.

    Regression: the previous range-ratio heuristic rejected layouts where
    ``y_range / inner_range`` fell in [0.3, 3.0]. pH 6–8 (range 2.0) vs
    absorbance 0.2–1.5 (range 1.3) gives ratio ≈ 1.54 and was silently
    misclassified, leaving ``data.y = None`` and y merged into X.
    """
    wv = np.linspace(1100, 2500, 51)
    rng = np.random.default_rng(7)
    y = np.linspace(6.0, 8.0, 20)  # pH, range 2.0
    X = rng.uniform(0.2, 1.5, size=(20, 51))  # absorbance, range 1.3

    fp = tmp_path / "ph_absorbance.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(y, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp))
    assert loaded.y is not None and loaded.y.shape == (20,)
    assert loaded.wv is not None and loaded.wv.shape == (51,)
    assert loaded.X.shape == (20, 51)
    np.testing.assert_allclose(loaded.y, y, rtol=1e-5, atol=1e-4)


def test_load_csv_auto_layout_rejects_wavelength_first_column(tmp_path: Path) -> None:
    """When the first column's values fall in the NIR wavelength band
    themselves, the column is treated as wavelengths (samples-in-columns
    layout) rather than reference values, so auto-layout refuses.
    """
    wv = np.linspace(1100, 2500, 51)
    # First column also holds NIR-range values → looks like wavelengths.
    first_col = np.linspace(1200, 2400, 20)
    rng = np.random.default_rng(3)
    X = rng.uniform(0.1, 1.0, size=(20, 51))

    fp = tmp_path / "wavelength_col.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(first_col, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp))
    # First column is in the NIR band → not treated as reference values.
    assert loaded.y is None


# ---------------------------------------------------------------------------
# Explicit y_col / wv_row override
# ---------------------------------------------------------------------------


def test_load_csv_explicit_y_col_and_wv_row_on_canonical_layout(tmp_path: Path) -> None:
    """Explicit y_col=0, wv_row=0 produces the same result as auto-detection
    on a canonical NIR CSV (empty corner + wavelength header + y column).
    """
    wv = np.linspace(1100, 2500, 51)
    rng = np.random.default_rng(42)
    y = np.linspace(5, 25, 20)
    X = rng.uniform(0.1, 1.5, size=(20, 51))

    fp = tmp_path / "nir_style.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(y, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), y_col=0, wv_row=0)
    assert loaded.X.shape == (20, 51)
    assert loaded.y is not None and loaded.y.shape == (20,)
    assert loaded.wv is not None and loaded.wv.shape == (51,)
    np.testing.assert_allclose(loaded.wv, wv, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(loaded.y, y, rtol=1e-5, atol=1e-4)


def test_load_csv_explicit_y_col_only(tmp_path: Path) -> None:
    """Passing only y_col (no wv_row) splits y but leaves wv=None."""
    rng = np.random.default_rng(7)
    y = np.linspace(10, 50, 15)
    X = rng.uniform(0, 1, size=(15, 30))

    fp = tmp_path / "plain.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        for yi, row in zip(y, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), y_col=0)
    assert loaded.X.shape == (15, 30)
    assert loaded.y is not None and loaded.y.shape == (15,)
    assert loaded.wv is None
    np.testing.assert_allclose(loaded.y, y, rtol=1e-5, atol=1e-4)


def test_load_csv_explicit_wv_row_only(tmp_path: Path) -> None:
    """Passing only wv_row (no y_col) splits wv but leaves y=None."""
    wv = np.linspace(1000, 2500, 30)
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(15, 30))

    fp = tmp_path / "plain.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(",".join(f"{w:g}" for w in wv) + "\n")
        for row in X:
            fh.write(",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), wv_row=0)
    assert loaded.X.shape == (15, 30)
    assert loaded.y is None
    assert loaded.wv is not None and loaded.wv.shape == (30,)
    # ``:g`` formatting trims to 6 significant digits — relax tolerance.
    np.testing.assert_allclose(loaded.wv, wv, rtol=1e-5, atol=1e-2)


def test_load_csv_explicit_override_bypasses_auto_detection(tmp_path: Path) -> None:
    """When explicit y_col/wv_row are given, auto-detection is skipped even
    if the first column would normally be rejected (e.g. NIR-range values).
    This gives the agent an escape hatch for edge cases.
    """
    wv = np.linspace(1100, 2500, 51)
    # First column holds NIR-range values — auto-detection would reject.
    first_col = np.linspace(1200, 2400, 20)
    rng = np.random.default_rng(3)
    X = rng.uniform(0.1, 1.0, size=(20, 51))

    fp = tmp_path / "wavelength_col.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("," + ",".join(f"{w:g}" for w in wv) + "\n")
        for yi, row in zip(first_col, X):
            fh.write(f"{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    # Auto-detection rejects (first col in NIR band).
    auto_loaded = load_csv(str(fp))
    assert auto_loaded.y is None

    # Explicit override forces y_col=0 despite the NIR-band values.
    override_loaded = load_csv(str(fp), y_col=0, wv_row=0)
    assert override_loaded.y is not None
    assert override_loaded.y.shape == (20,)
    assert override_loaded.wv is not None
    assert override_loaded.X.shape == (20, 51)


def test_load_csv_explicit_y_col_out_of_range_raises(tmp_path: Path) -> None:
    """An out-of-range y_col raises ValueError (not silently ignored)."""
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, size=(5, 3))
    fp = tmp_path / "small.csv"
    np.savetxt(str(fp), X, delimiter=",")

    with pytest.raises(ValueError, match="y_col.*out of range"):
        load_csv(str(fp), y_col=10)


# ────────────────────────────────────────────────────────────────────────────
# ★ v3.8: x_cols selector tests (skip metadata columns)
# ────────────────────────────────────────────────────────────────────────────


def test_load_csv_x_cols_slice_from_n(tmp_path: Path) -> None:
    """x_cols="8:" selects columns 8 to end, skipping metadata prefix."""
    rng = np.random.default_rng(1)
    n_samples, n_spec = 20, 30
    n_meta = 8
    X_spec = rng.uniform(0, 1, size=(n_samples, n_spec))
    # Metadata columns: non-numeric strings that genfromtxt turns into NaN.
    meta_cols = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp"]
    fp = tmp_path / "mango_like.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        # Header
        fh.write(",".join(meta_cols) + "," + ",".join(f"w{i}" for i in range(n_spec)) + "\n")
        # Data rows: 8 string cells + 30 numeric cells
        for row in X_spec:
            fh.write(",".join(meta_cols) + "," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), x_cols="8:")
    assert loaded.X.shape == (n_samples, n_spec)
    np.testing.assert_allclose(loaded.X, X_spec, rtol=1e-5, atol=1e-4)
    assert loaded.y is None
    assert loaded.wv is None


def test_load_csv_x_cols_with_y_col(tmp_path: Path) -> None:
    """x_cols="9:" with y_col=8 skips metadata (0-7) and y (8), keeps 9+."""
    rng = np.random.default_rng(2)
    n_samples, n_spec = 15, 25
    n_meta = 8
    y = rng.uniform(5, 25, size=n_samples)  # DM values
    X_spec = rng.uniform(0, 1, size=(n_samples, n_spec))
    meta_cols = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp"]
    fp = tmp_path / "mango_with_y.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        # Header: 8 meta + DM + 25 spectra
        fh.write(",".join(meta_cols) + ",DM," + ",".join(f"w{i}" for i in range(n_spec)) + "\n")
        # Data rows: 8 strings + y + 30 spectra values
        for yi, row in zip(y, X_spec):
            fh.write(",".join(meta_cols) + f",{yi:g}," + ",".join(f"{v:g}" for v in row) + "\n")

    loaded = load_csv(str(fp), y_col=8, x_cols="9:")
    # X should be the 25 spectra columns (col 9 onwards)
    assert loaded.X.shape == (n_samples, n_spec)
    np.testing.assert_allclose(loaded.X, X_spec, rtol=1e-5, atol=1e-4)
    # y should be the DM column (col 8)
    assert loaded.y is not None
    assert loaded.y.shape == (n_samples,)
    np.testing.assert_allclose(loaded.y, y, rtol=1e-5, atol=1e-4)


def test_load_csv_x_cols_explicit_list(tmp_path: Path) -> None:
    """x_cols="8,9,10" selects exactly those three columns."""
    rng = np.random.default_rng(3)
    n_samples, n_cols = 10, 15
    X = rng.uniform(0, 1, size=(n_samples, n_cols))
    fp = tmp_path / "explicit.csv"
    np.savetxt(str(fp), X, delimiter=",")

    loaded = load_csv(str(fp), x_cols="8,9,10")
    assert loaded.X.shape == (n_samples, 3)
    np.testing.assert_allclose(loaded.X[:, 0], X[:, 8], rtol=1e-6)
    np.testing.assert_allclose(loaded.X[:, 1], X[:, 9], rtol=1e-6)
    np.testing.assert_allclose(loaded.X[:, 2], X[:, 10], rtol=1e-6)


def test_load_csv_x_cols_half_open_range(tmp_path: Path) -> None:
    """x_cols="8:12" selects columns 8,9,10,11 (Python half-open)."""
    rng = np.random.default_rng(4)
    n_samples, n_cols = 10, 20
    X = rng.uniform(0, 1, size=(n_samples, n_cols))
    fp = tmp_path / "range.csv"
    np.savetxt(str(fp), X, delimiter=",")

    loaded = load_csv(str(fp), x_cols="8:12")
    assert loaded.X.shape == (n_samples, 4)
    np.testing.assert_allclose(loaded.X, X[:, 8:12], rtol=1e-6)


def test_load_csv_x_cols_negative_start(tmp_path: Path) -> None:
    """x_cols="-5:" selects the last 5 columns."""
    rng = np.random.default_rng(5)
    n_samples, n_cols = 10, 20
    X = rng.uniform(0, 1, size=(n_samples, n_cols))
    fp = tmp_path / "neg.csv"
    np.savetxt(str(fp), X, delimiter=",")

    loaded = load_csv(str(fp), x_cols="-5:")
    assert loaded.X.shape == (n_samples, 5)
    np.testing.assert_allclose(loaded.X, X[:, 15:], rtol=1e-6)


def test_load_csv_x_cols_y_col_inside_range(tmp_path: Path) -> None:
    """When y_col is inside x_cols range, y_col is auto-excluded from X."""
    rng = np.random.default_rng(6)
    n_samples, n_cols = 10, 10
    X = rng.uniform(0, 1, size=(n_samples, n_cols))
    fp = tmp_path / "y_inside.csv"
    np.savetxt(str(fp), X, delimiter=",")

    # x_cols=":" means all columns; y_col=5 should be excluded from X
    loaded = load_csv(str(fp), y_col=5, x_cols=":")
    assert loaded.y is not None
    assert loaded.y.shape == (n_samples,)
    np.testing.assert_allclose(loaded.y, X[:, 5], rtol=1e-6)
    # X should have 9 columns (all except col 5)
    assert loaded.X.shape == (n_samples, 9)
    # Verify col 5 is NOT in X by checking remaining column indices
    expected = np.delete(X, 5, axis=1)
    np.testing.assert_allclose(loaded.X, expected, rtol=1e-6)


def test_load_csv_x_cols_out_of_range_raises(tmp_path: Path) -> None:
    """An out-of-range x_cols index raises ValueError."""
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, size=(5, 3))
    fp = tmp_path / "small.csv"
    np.savetxt(str(fp), X, delimiter=",")

    with pytest.raises(ValueError, match="x_cols.*out of range|selected 0 columns"):
        load_csv(str(fp), x_cols="10:")


def test_load_csv_x_cols_invalid_syntax_raises(tmp_path: Path) -> None:
    """An malformed x_cols spec raises ValueError."""
    rng = np.random.default_rng(8)
    X = rng.uniform(0, 1, size=(5, 3))
    fp = tmp_path / "small.csv"
    np.savetxt(str(fp), X, delimiter=",")

    with pytest.raises(ValueError, match="Invalid x_cols"):
        load_csv(str(fp), x_cols="not_a_number")


def test_load_csv_x_cols_with_wv_row(tmp_path: Path) -> None:
    """x_cols works together with wv_row (wavelength header in first row)."""
    rng = np.random.default_rng(9)
    n_samples, n_spec = 12, 20
    n_meta = 3
    wv = np.linspace(1000, 2500, n_spec)
    X_spec = rng.uniform(0, 1, size=(n_samples, n_spec))
    meta_cols = ["Set", "Season", "Region"]
    fp = tmp_path / "meta_with_wv.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        # Header: 3 meta + 20 wavelength values
        fh.write(",".join(meta_cols) + "," + ",".join(f"{w:g}" for w in wv) + "\n")
        # Data: 3 strings + 20 spectra
        for row in X_spec:
            fh.write(",".join(meta_cols) + "," + ",".join(f"{v:g}" for v in row) + "\n")

    # wv_row=0 picks the wavelength header; x_cols="3:" skips metadata cols
    loaded = load_csv(str(fp), wv_row=0, x_cols="3:")
    assert loaded.wv is not None
    assert loaded.wv.shape == (n_spec,)
    np.testing.assert_allclose(loaded.wv, wv, rtol=1e-5, atol=1e-2)
    assert loaded.X.shape == (n_samples, n_spec)
    np.testing.assert_allclose(loaded.X, X_spec, rtol=1e-5, atol=1e-4)
