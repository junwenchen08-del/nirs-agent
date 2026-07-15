"""Tests for :mod:`nir_core.io.sniffers` and :mod:`nir_core.io.loaders.auto_detect_and_load`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from nir_core.io.loaders import auto_detect_and_load
from nir_core.io.sniffers import detect_format, detect_structure, inspect_file
from nir_core.io.writers import save_csv
from nir_core.models import SpectralData


def test_detect_format_extensions(tmp_path: Path) -> None:
    """detect_format should map extensions correctly."""
    (tmp_path / "a.mat").write_bytes(b"MATLAB 5.0 MAT-file")
    (tmp_path / "b.csv").write_text("a,b,c\n1,2,3\n")
    (tmp_path / "c.txt").write_text("1 2 3\n4 5 6\n")

    assert detect_format(str(tmp_path / "a.mat")) == "mat"
    assert detect_format(str(tmp_path / "b.csv")) == "csv"
    assert detect_format(str(tmp_path / "c.txt")) == "txt"


def test_detect_format_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_format(str(tmp_path / "ghost.csv"))


def test_detect_structure_rows() -> None:
    """Tall matrices -> samples_in_rows."""
    X = np.arange(200 * 50).reshape(200, 50).astype(float)
    assert detect_structure(X) == "samples_in_rows"


def test_detect_structure_columns() -> None:
    """Wide matrices -> samples_in_columns."""
    X = np.arange(50 * 200).reshape(50, 200).astype(float)
    assert detect_structure(X) == "samples_in_columns"


def test_detect_structure_default_square() -> None:
    """Square / near-square matrices default to samples_in_rows."""
    X = np.ones((10, 10))
    assert detect_structure(X) == "samples_in_rows"


def test_detect_structure_not_2d() -> None:
    with pytest.raises(ValueError):
        detect_structure(np.zeros(5))


def test_auto_detect_and_load_csv(
    tmp_path: Path, small_synthetic_data: SpectralData
) -> None:
    """auto_detect_and_load should round-trip a spectra-only CSV cleanly."""
    src = small_synthetic_data
    # Save without y/wv so auto_detect_and_load (no selectors) recovers X.
    data = SpectralData(X=src.X)
    fp = tmp_path / "auto.csv"
    save_csv(data, str(fp))

    loaded = auto_detect_and_load(str(fp))
    assert loaded.original_format == "csv"
    assert loaded.X.shape == src.X.shape
    np.testing.assert_allclose(loaded.X, src.X, rtol=1e-6, atol=1e-8)


def test_auto_detect_and_load_mat(tmp_path: Path) -> None:
    """auto_detect_and_load should dispatch .mat to load_mat."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 20))
    fp = tmp_path / "auto.mat"
    savemat(str(fp), {"X": X})

    loaded = auto_detect_and_load(str(fp))
    assert loaded.original_format == "mat"
    np.testing.assert_allclose(loaded.X, X)


def test_inspect_file_csv(tmp_path: Path) -> None:
    """inspect_file on a CSV should return a JSON string with expected keys."""
    import json

    # 10 rows x 3 cols -> ratio 3.33, n_rows > n_cols -> samples_in_rows.
    X = np.linspace(0, 1, 30).reshape(10, 3)
    fp = tmp_path / "inspect.csv"
    save_csv(SpectralData(X=X), str(fp))

    payload = inspect_file(str(fp))
    info = json.loads(payload)
    assert info["format"] == "csv"
    assert info["shape"] == [10, 3]
    assert info["estimated_samples"] == 10
    assert info["estimated_wavelengths"] == 3
    assert info["structure"] == "samples_in_rows"
    assert info["value_range"] is not None
    assert info["has_nan"] is False


def test_inspect_file_csv_detects_non_numeric_columns(tmp_path: Path) -> None:
    """★ v3.8: inspect_file should detect non-numeric metadata columns.

    Simulates the Anderson 2020 mango CSV layout: 8 string metadata columns
    (Set/Season/Region/...) followed by numeric spectra columns. The string
    columns become NaN under genfromtxt(dtype=float); inspect should flag
    them in ``non_numeric_columns`` so the agent can pass x_cols to skip.
    """
    import json

    n_samples, n_spec = 20, 10
    rng = np.random.default_rng(42)
    X_spec = rng.uniform(0, 1, size=(n_samples, n_spec))
    meta_cols = ["Set", "Season", "Region", "Date", "Type", "Cultivar", "Pop", "Temp"]
    fp = tmp_path / "mango_like.csv"
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(",".join(meta_cols) + "," + ",".join(f"w{i}" for i in range(n_spec)) + "\n")
        for row in X_spec:
            fh.write(",".join(meta_cols) + "," + ",".join(f"{v:g}" for v in row) + "\n")

    info = json.loads(inspect_file(str(fp)))
    assert info["format"] == "csv"
    assert "non_numeric_columns" in info
    assert info["non_numeric_columns"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert info["has_nan"] is True


def test_inspect_file_csv_no_non_numeric_columns_when_all_numeric(tmp_path: Path) -> None:
    """★ v3.8: A purely numeric CSV should report non_numeric_columns=[]."""
    import json

    X = np.linspace(0, 1, 30).reshape(10, 3)
    fp = tmp_path / "numeric.csv"
    np.savetxt(str(fp), X, delimiter=",")

    info = json.loads(inspect_file(str(fp)))
    assert info["non_numeric_columns"] == []


def test_inspect_file_mat(tmp_path: Path) -> None:
    import json

    # 10 rows x 4 cols -> ratio 2.5, n_rows > n_cols -> samples_in_rows.
    X = np.linspace(0, 5, 40).reshape(10, 4)
    fp = tmp_path / "inspect.mat"
    savemat(str(fp), {"X": X})

    info = json.loads(inspect_file(str(fp)))
    assert info["format"] == "mat"
    assert info["shape"] == [10, 4]
    assert info["structure"] == "samples_in_rows"
