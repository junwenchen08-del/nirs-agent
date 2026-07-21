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
        fh.write(
            ",".join(meta_cols) + "," + ",".join(f"w{i}" for i in range(n_spec)) + "\n"
        )
        for row in X_spec:
            fh.write(",".join(meta_cols) + "," + ",".join(f"{v:g}" for v in row) + "\n")

    info = json.loads(inspect_file(str(fp)))
    assert info["format"] == "csv"
    assert "non_numeric_columns" in info
    assert info["non_numeric_columns"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert info["has_nan"] is True


def test_inspect_file_csv_no_non_numeric_columns_when_all_numeric(
    tmp_path: Path,
) -> None:
    """★ v3.8: A purely numeric CSV should report non_numeric_columns=[]."""
    import json

    X = np.linspace(0, 1, 30).reshape(10, 3)
    fp = tmp_path / "numeric.csv"
    np.savetxt(str(fp), X, delimiter=",")

    info = json.loads(inspect_file(str(fp)))
    assert info["non_numeric_columns"] == []


def test_inspect_csv_reports_dialect_and_high_confidence_schema(tmp_path: Path) -> None:
    """Semicolon/decimal-comma files should be mapped without ad-hoc code."""
    import json

    fp = tmp_path / "european.csv"
    fp.write_text(
        "Sample;Protein;1100;1102;1104\n"
        "S1;10,5;0,10;0,20;0,30\n"
        "S2;11,0;0,11;0,21;0,31\n"
        "S3;11,5;0,12;0,22;0,32\n",
        encoding="utf-8",
    )

    info = json.loads(inspect_file(str(fp)))
    assert info["dialect"]["delimiter"] == ";"
    assert info["dialect"]["decimal"] == ","
    mapping = info["schema_mapping"]
    assert mapping["status"] == "auto"
    assert mapping["confidence"] >= 0.8
    assert mapping["sample_id_columns"] == [0]
    assert mapping["target_columns"] == [1]
    assert mapping["spectral_columns"] == [2, 3, 4]


def test_inspect_csv_requests_mapping_when_column_roles_are_ambiguous(
    tmp_path: Path,
) -> None:
    """Unknown numeric fields must not be silently assigned to X or y."""
    import json

    fp = tmp_path / "ambiguous.csv"
    fp.write_text(
        "id,value_a,value_b,value_c\nS1,1,2,3\nS2,4,5,6\nS3,7,8,9\n",
        encoding="utf-8",
    )

    mapping = json.loads(inspect_file(str(fp)))["schema_mapping"]
    assert mapping["status"] == "needs_user_mapping"
    assert mapping["confidence"] < 0.8
    assert mapping["candidate_numeric_columns"] == [1, 2, 3]
    assert mapping["clarification_required"] is True


def test_inspect_csv_recognizes_dry_matter_abbreviation(tmp_path: Path) -> None:
    import json

    fp = tmp_path / "dm_target.csv"
    fp.write_text(
        "Set,DM,900,910,920\n"
        "Cal,12.1,0.1,0.2,0.3\n"
        "Cal,13.2,0.2,0.3,0.4\n"
        "Test,14.3,0.3,0.4,0.5\n",
        encoding="utf-8",
    )

    mapping = json.loads(inspect_file(str(fp)))["schema_mapping"]
    assert mapping["status"] == "auto"
    assert mapping["target_columns"] == [1]
    assert mapping["spectral_columns"] == [2, 3, 4]


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
