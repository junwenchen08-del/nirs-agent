"""Multi-target CSV loading and NPZ persistence tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nir_core.io.loaders import load_csv
from nir_core.io.writers import save_npz


def test_load_csv_multiple_reference_columns_with_names(tmp_path: Path) -> None:
    path = tmp_path / "multi.csv"
    path.write_text(
        "protein,moisture,oil,1100,1200,1300\n"
        "10.1,12.0,4.1,0.10,0.20,0.30\n"
        "10.8,11.5,4.4,0.15,0.25,0.35\n",
        encoding="utf-8",
    )

    data = load_csv(str(path), y_cols=[0, 1, 2], x_cols="3:")

    assert data.X.shape == (2, 3)
    assert data.y is not None and data.y.shape == (2, 3)
    assert data.y_names == ["protein", "moisture", "oil"]
    np.testing.assert_allclose(data.wv, [1100.0, 1200.0, 1300.0])
    assert data.summary()["n_components"] == 3
    assert data.summary()["y_ranges"]["moisture"] == [11.5, 12.0]
    # 2D y must NOT report a scalar y_range (cross-component mix is misleading);
    # consumers should use per-component y_ranges instead.
    assert data.summary()["y_range"] is None

    output = tmp_path / "multi.npz"
    save_npz(data, str(output))
    archive = np.load(output, allow_pickle=False)
    assert archive["y"].shape == (2, 3)
    assert archive["y_names"].tolist() == ["protein", "moisture", "oil"]


def test_load_csv_keeps_wavelengths_aligned_with_selected_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "selected.csv"
    path.write_text(
        ",,1000,1100,1200,1300\n"
        "10.1,12.0,0.10,0.20,0.30,0.40\n"
        "10.8,11.5,0.15,0.25,0.35,0.45\n",
        encoding="utf-8",
    )

    data = load_csv(
        str(path),
        y_cols=[0, 1],
        x_cols="4:6",
        wv_row=0,
    )

    np.testing.assert_allclose(data.X, [[0.30, 0.40], [0.35, 0.45]])
    np.testing.assert_allclose(data.y, [[10.1, 12.0], [10.8, 11.5]])
    np.testing.assert_allclose(data.wv, [1200.0, 1300.0])


def test_load_csv_rejects_y_col_and_y_cols_together(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    np.savetxt(path, np.arange(20, dtype=float).reshape(5, 4), delimiter=",")

    try:
        load_csv(str(path), y_col=0, y_cols=[0, 1])
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected y_col/y_cols conflict to fail")


def test_summary_y_range_1d_returns_scalar_range() -> None:
    """1D y must still populate y_range (backward compatibility); only 2D y
    suppresses it in favour of per-component y_ranges."""
    from nir_core.models import SpectralData

    data = SpectralData(
        X=np.array([[0.1, 0.2], [0.3, 0.4]]),
        y=np.array([8.0, 12.0]),
        wv=np.array([1100.0, 1200.0]),
        y_names=["ref"],
    )
    summary = data.summary()
    assert summary["n_components"] == 1
    assert summary["y_range"] == [8.0, 12.0]
    # y_ranges still populated for consistency (single-entry dict).
    assert summary["y_ranges"] == {"ref": [8.0, 12.0]}
