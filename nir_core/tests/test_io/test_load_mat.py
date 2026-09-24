"""Tests for :mod:`nir_core.io.loaders.load_mat`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nir_core.io.loaders import load_mat
from nir_core.models import SpectralData


def _h5py_available() -> bool:
    try:
        # Force a real use to catch binary-incompat issues like the
        # numpy.dtype size mismatch we saw on this machine.
        import io

        import h5py  # type: ignore  # noqa: F401

        buf = io.BytesIO()
        with h5py.File(buf, "w") as fh:
            fh.create_dataset("x", data=np.zeros(2))
        return True
    except Exception:
        return False


def test_load_mat_v5_roundtrip(
    tmp_path: Path, small_synthetic_data: SpectralData
) -> None:
    """Save a v5 .mat with scipy.io.savemat and load it back."""
    from scipy.io import savemat

    src = small_synthetic_data
    fp = tmp_path / "data_v5.mat"
    savemat(
        str(fp),
        {
            "X": src.X,
            "y": src.y if src.y is not None else np.zeros(src.X.shape[0]),
            "wv": src.wv if src.wv is not None else np.zeros(src.X.shape[1]),
        },
        do_compression=False,
    )

    loaded = load_mat(str(fp))
    assert loaded.original_format == "mat"
    assert loaded.X.shape == src.X.shape
    np.testing.assert_allclose(loaded.X, src.X)
    assert loaded.y is not None
    np.testing.assert_allclose(
        loaded.y, src.y if src.y is not None else np.zeros(src.X.shape[0])
    )
    assert loaded.wv is not None
    np.testing.assert_allclose(
        loaded.wv, src.wv if src.wv is not None else np.zeros(src.X.shape[1])
    )


def test_load_mat_v5_explicit_vars(tmp_path: Path) -> None:
    """Explicit x_var / y_var / wv_var should override heuristics."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    X = rng.standard_normal((15, 30))
    y = rng.standard_normal(15)
    wv = np.linspace(1100, 2500, 30)
    fp = tmp_path / "named.mat"
    savemat(
        str(fp),
        {"spectra": X, "moisture": y, "wavelengths": wv},
    )

    loaded = load_mat(
        str(fp),
        x_var="spectra",
        y_var="moisture",
        wv_var="wavelengths",
    )
    assert loaded.X.shape == (15, 30)
    np.testing.assert_allclose(loaded.X, X)
    np.testing.assert_allclose(loaded.y, y)
    np.testing.assert_allclose(loaded.wv, wv)


def test_load_mat_labeled_matrix_bundle_extracts_target_metadata_and_spectra(
    tmp_path: Path,
) -> None:
    """PLS-Toolbox Matrix/VarLabels/ObjLabels files load without scripts."""
    from scipy.io import savemat

    rng = np.random.default_rng(17)
    n_samples = 18
    wavelengths = np.array([7398.337, 7406.051, 7413.766, 7421.481])
    y = np.linspace(4.6, 9.7, n_samples)
    sample_type = np.repeat([1.0, 2.0, 3.0], 6)
    scale = np.zeros(n_samples)
    spectra = rng.normal(size=(n_samples, wavelengths.size))
    matrix = np.column_stack([y, sample_type, scale, spectra])
    var_labels = np.array(
        ["active (%w/w)", "Type", "Scale", *[f"{value:.3f}" for value in wavelengths]],
        dtype=object,
    )
    obj_labels = np.array(
        [f"{index:02d}_01" for index in range(n_samples)], dtype=object
    )
    source = tmp_path / "tablets.MAT"
    savemat(
        source, {"Matrix": matrix, "ObjLabels": obj_labels, "VarLabels": var_labels}
    )

    loaded = load_mat(str(source))

    np.testing.assert_allclose(loaded.X, spectra)
    np.testing.assert_allclose(loaded.y, y)
    np.testing.assert_allclose(loaded.wv, wavelengths, atol=5e-4)
    assert loaded.y_names == ["active (%w/w)"]
    assert loaded.sample_names == obj_labels.tolist()


def test_inspect_mat_labeled_matrix_bundle_reports_auto_load_layout(
    tmp_path: Path,
) -> None:
    """Inspection exposes target/metadata/spectral columns without Python fallback."""
    import json

    from scipy.io import savemat

    from nir_core.io.sniffers import inspect_file

    matrix = np.arange(35, dtype=float).reshape(5, 7)
    source = tmp_path / "labeled.mat"
    savemat(
        source,
        {
            "Matrix": matrix,
            "ObjLabels": np.array(
                [f"sample-{index}" for index in range(5)], dtype=object
            ),
            "VarLabels": np.array(
                [
                    "active (%w/w)",
                    "Type",
                    "Scale",
                    "7421.481",
                    "7413.766",
                    "7406.051",
                    "7398.337",
                ],
                dtype=object,
            ),
        },
    )

    info = json.loads(inspect_file(str(source)))

    assert info["labeled_matrix"] is True
    assert info["shape"] == [5, 4]
    assert info["target_columns"] == [{"index": 0, "name": "active (%w/w)"}]
    assert info["metadata_columns"] == [
        {"index": 1, "name": "Type"},
        {"index": 2, "name": "Scale"},
    ]
    assert info["spectral_column_indices"] == [3, 4, 5, 6]
    assert info["wavelength_range"] == [7398.337, 7421.481]
    assert info["axis_first"] == 7421.481
    assert info["axis_last"] == 7398.337
    assert info["axis_direction"] == "descending"


def test_load_mat_recursively_infers_unknown_fields_and_transposes(
    tmp_path: Path,
) -> None:
    import json

    from scipy.io import savemat

    from nir_core.io.sniffers import inspect_file

    rng = np.random.default_rng(2026)
    expected_X = rng.normal(size=(12, 30))
    expected_y = np.linspace(2.0, 8.0, 12)
    expected_wv = np.linspace(900.0, 1700.0, 30)
    fp = tmp_path / "nested_unknown_fields.mat"
    savemat(
        str(fp),
        {
            "Study": {
                "Measurements": {
                    "Signal": expected_X.T,
                    "Chemistry": expected_y,
                    "Axis": expected_wv,
                }
            }
        },
    )

    loaded = load_mat(str(fp))
    info = json.loads(inspect_file(str(fp)))

    np.testing.assert_allclose(loaded.X, expected_X)
    np.testing.assert_allclose(loaded.y, expected_y)
    np.testing.assert_allclose(loaded.wv, expected_wv)
    mapping = info["schema_mapping"]
    assert mapping["status"] == "auto"
    assert mapping["transpose"] is True
    assert mapping["x_variable"].endswith("Signal")
    assert mapping["y_variable"].endswith("Chemistry")
    assert mapping["wv_variable"].endswith("Axis")
    assert info["axis_first"] == 900.0
    assert info["axis_last"] == 1700.0
    assert info["axis_direction"] == "ascending"


def test_inspect_mat_reports_ambiguous_equal_matrix_candidates(tmp_path: Path) -> None:
    import json

    from scipy.io import savemat

    from nir_core.io.sniffers import inspect_file

    rng = np.random.default_rng(44)
    fp = tmp_path / "ambiguous_matrices.mat"
    savemat(
        str(fp),
        {
            "block_a": rng.normal(size=(10, 20)),
            "block_b": rng.normal(size=(10, 20)),
            "response": np.linspace(1.0, 2.0, 10),
        },
    )

    info = json.loads(inspect_file(str(fp)))

    assert info["schema_mapping"]["status"] == "needs_user_mapping"
    assert set(info["schema_mapping"]["x_candidates"]) == {"block_a", "block_b"}
    assert info["schema_mapping"]["clarification_required"] is True


def test_load_mat_accepts_explicit_dotted_mapping_for_nested_fields(
    tmp_path: Path,
) -> None:
    from scipy.io import savemat

    rng = np.random.default_rng(91)
    expected = rng.normal(size=(10, 20))
    other = rng.normal(size=(10, 20))
    response = np.linspace(3.0, 7.0, 10)
    fp = tmp_path / "manual_nested_mapping.mat"
    savemat(
        str(fp),
        {
            "Experiment": {
                "block_a": other.T,
                "block_b": expected.T,
                "response": response,
            }
        },
    )

    loaded = load_mat(
        str(fp),
        x_var="Experiment.block_b",
        y_var="Experiment.response",
        transpose=True,
    )

    np.testing.assert_allclose(loaded.X, expected)
    np.testing.assert_allclose(loaded.y, response)


def test_load_mat_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mat(str(tmp_path / "nope.mat"))


@pytest.mark.skipif(
    not _h5py_available(),
    reason="h5py not installed or binary-incompatible (v7.3 support optional)",
)
def test_load_mat_v73_roundtrip(tmp_path: Path) -> None:
    """Round-trip a v7.3 .mat file via h5py (skipped if h5py unavailable)."""
    import h5py  # type: ignore

    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 25)).astype(float)
    y = rng.standard_normal(12).astype(float)
    wv = np.linspace(1000, 2000, 25).astype(float)
    fp = tmp_path / "data_v73.mat"
    with h5py.File(str(fp), "w") as fh:
        fh.create_dataset("X", data=X)
        fh.create_dataset("y", data=y)
        fh.create_dataset("wv", data=wv)

    loaded = load_mat(str(fp))
    assert loaded.original_format == "mat"
    np.testing.assert_allclose(loaded.X, X)
    assert loaded.y is not None
    np.testing.assert_allclose(loaded.y, y)
    assert loaded.wv is not None
    np.testing.assert_allclose(loaded.wv, wv)


@pytest.mark.skipif(
    not _h5py_available(),
    reason="h5py not installed or binary-incompatible (v7.3 support optional)",
)
def test_load_mat_v73_recurses_nested_groups_and_infers_orientation(
    tmp_path: Path,
) -> None:
    import json

    import h5py  # type: ignore

    from nir_core.io.sniffers import inspect_file

    rng = np.random.default_rng(73)
    X = rng.normal(size=(9, 24))
    y = np.linspace(1.0, 5.0, 9)
    wv = np.linspace(1000.0, 2200.0, 24)
    fp = tmp_path / "nested_v73.mat"
    with h5py.File(str(fp), "w") as handle:
        group = handle.create_group("Study/Measurements")
        group.create_dataset("Signal", data=X.T)
        group.create_dataset("Chemistry", data=y)
        group.create_dataset("Axis", data=wv)

    loaded = load_mat(str(fp))
    info = json.loads(inspect_file(str(fp)))

    np.testing.assert_allclose(loaded.X, X)
    np.testing.assert_allclose(loaded.y, y)
    np.testing.assert_allclose(loaded.wv, wv)
    assert info["schema_mapping"]["transpose"] is True


# ---------------------------------------------------------------------------
# v3.7: MATLAB struct .mat file support
# ---------------------------------------------------------------------------
# Public NIR datasets (Open-Nirs-Datasets, Melamine, ...) store data as
# nested MATLAB structs. These tests exercise the struct-flattening path
# in loaders.py and the struct-detection path in sniffers.py.


def _save_struct_mat_v5(
    fp: Path,
    wn1: np.ndarray,
    wn2: np.ndarray,
    subsets: dict[str, dict[str, np.ndarray]],
) -> None:
    """Save a v5 .mat file mimicking the Open-Nirs-Datasets layout.

    Produces a top-level struct with fields ``wn1``, ``wn2``, and one
    nested struct per subset (e.g. R562). Each subset struct has
    ``X1``, ``X2``, ``Y`` fields.
    """
    from scipy.io import savemat

    savemat(str(fp), {"Dataset": {"wn1": wn1, "wn2": wn2, **subsets}})


def test_load_mat_struct_with_subset(tmp_path: Path) -> None:
    """Struct .mat with explicit subset selection loads X1+X2 and Y."""
    rng = np.random.default_rng(0)
    n_samples = 50
    X1 = rng.standard_normal((n_samples, 225))
    X2 = rng.standard_normal((n_samples, 121))
    Y = rng.standard_normal(n_samples)
    wn1 = np.linspace(900, 1700, 225)
    wn2 = np.linspace(1700, 2500, 121)

    subsets = {
        "R562": {"X1": X1, "X2": X2, "Y": Y},
        "R568": {"X1": X1 * 0.9, "X2": X2 * 0.9, "Y": Y * 1.1},
    }
    fp = tmp_path / "struct.mat"
    _save_struct_mat_v5(fp, wn1, wn2, subsets)

    loaded = load_mat(str(fp), subset="R562")
    assert loaded.X.shape == (n_samples, 225 + 121)
    np.testing.assert_allclose(loaded.X[:, :225], X1)
    np.testing.assert_allclose(loaded.X[:, 225:], X2)
    assert loaded.y is not None
    np.testing.assert_allclose(loaded.y, Y)
    assert loaded.wv is not None
    assert loaded.wv.shape == (225 + 121,)
    np.testing.assert_allclose(loaded.wv[:225], wn1)
    np.testing.assert_allclose(loaded.wv[225:], wn2)


def test_load_mat_struct_no_subset_multiple_raises(tmp_path: Path) -> None:
    """When multiple subsets exist and subset is None, a helpful error lists them."""
    rng = np.random.default_rng(1)
    n = 10
    subsets = {
        "R562": {"X1": rng.standard_normal((n, 5)), "Y": rng.standard_normal(n)},
        "R568": {"X1": rng.standard_normal((n, 5)), "Y": rng.standard_normal(n)},
    }
    fp = tmp_path / "multi.mat"
    _save_struct_mat_v5(fp, np.linspace(900, 1700, 5), np.array([]), subsets)

    with pytest.raises(ValueError) as excinfo:
        load_mat(str(fp))
    msg = str(excinfo.value)
    assert "R562" in msg
    assert "R568" in msg
    assert "subset" in msg.lower()


def test_load_mat_struct_invalid_subset_raises(tmp_path: Path) -> None:
    """Invalid subset name raises ValueError listing available subsets."""
    rng = np.random.default_rng(2)
    n = 8
    subsets = {
        "R562": {"X1": rng.standard_normal((n, 5)), "Y": rng.standard_normal(n)},
    }
    fp = tmp_path / "single.mat"
    _save_struct_mat_v5(fp, np.linspace(900, 1700, 5), np.array([]), subsets)

    with pytest.raises(ValueError) as excinfo:
        load_mat(str(fp), subset="R999")
    assert "R562" in str(excinfo.value)


def test_load_mat_struct_single_subset_auto_selected(tmp_path: Path) -> None:
    """When exactly one subset exists and subset is None, it's auto-selected."""
    rng = np.random.default_rng(3)
    n = 12
    X1 = rng.standard_normal((n, 5))
    Y = rng.standard_normal(n)
    wn1 = np.linspace(900, 1700, 5)
    subsets = {"only": {"X1": X1, "Y": Y}}
    fp = tmp_path / "auto.mat"
    _save_struct_mat_v5(fp, wn1, np.array([]), subsets)

    loaded = load_mat(str(fp))
    assert loaded.X.shape == (n, 5)
    np.testing.assert_allclose(loaded.X, X1)
    assert loaded.y is not None
    np.testing.assert_allclose(loaded.y, Y)


def test_load_mat_struct_flat_no_subset(tmp_path: Path) -> None:
    """Flat struct with X / Y / wn siblings (no nested sub-structs)."""
    from scipy.io import savemat

    rng = np.random.default_rng(4)
    n, p = 20, 15
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    wn = np.linspace(1100, 2500, p)
    fp = tmp_path / "flat.mat"
    savemat(str(fp), {"Data": {"X": X, "Y": Y, "wn": wn}})

    loaded = load_mat(str(fp))
    assert loaded.X.shape == (n, p)
    np.testing.assert_allclose(loaded.X, X)
    assert loaded.y is not None
    np.testing.assert_allclose(loaded.y, Y)
    assert loaded.wv is not None
    np.testing.assert_allclose(loaded.wv, wn)


def test_inspect_mat_struct_returns_subset_metadata(tmp_path: Path) -> None:
    """nir_inspect on a struct .mat returns is_struct + available_subsets."""
    from nir_core.io.sniffers import inspect_file

    rng = np.random.default_rng(5)
    n = 30
    subsets = {
        "R562": {
            "X1": rng.standard_normal((n, 225)),
            "X2": rng.standard_normal((n, 121)),
            "Y": rng.standard_normal(n),
        },
        "R568": {
            "X1": rng.standard_normal((n, 225)),
            "X2": rng.standard_normal((n, 121)),
            "Y": rng.standard_normal(n),
        },
    }
    fp = tmp_path / "inspect.mat"
    _save_struct_mat_v5(
        fp, np.linspace(900, 1700, 225), np.linspace(1700, 2500, 121), subsets
    )

    import json

    info = json.loads(inspect_file(str(fp)))
    assert info["is_struct"] is True
    assert set(info["available_subsets"]) == {"R562", "R568"}
    assert "X1" in info["x_fields"]
    assert "X2" in info["x_fields"]
    assert "Y" in info["y_fields"]
    assert "wn1" in info["wv_fields"]
    assert "wn2" in info["wv_fields"]
    # Shape should reflect concatenated X1 + X2 columns.
    assert info["shape"] == [n, 225 + 121]
    assert info["estimated_samples"] == n
    assert info["estimated_wavelengths"] == 225 + 121


def test_inspect_mat_struct_no_crash_on_object_dtype(tmp_path: Path) -> None:
    """Regression test: the original bug crashed with TypeError on object dtype."""
    from nir_core.io.sniffers import inspect_file

    rng = np.random.default_rng(6)
    n = 10
    subsets = {
        "R562": {"X1": rng.standard_normal((n, 5)), "Y": rng.standard_normal(n)},
    }
    fp = tmp_path / "bug.mat"
    _save_struct_mat_v5(fp, np.linspace(900, 1700, 5), np.array([]), subsets)

    # Should NOT raise TypeError: ufunc 'isfinite' not supported for input types.
    import json

    info = json.loads(inspect_file(str(fp)))
    assert info["is_struct"] is True
