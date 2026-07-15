"""Tests for :mod:`nir_core.io.loaders.load_mat`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nir_core.io.loaders import load_mat
from nir_core.models import SpectralData


def _h5py_available() -> bool:
    try:
        import h5py  # type: ignore  # noqa: F401

        # Force a real use to catch binary-incompat issues like the
        # numpy.dtype size mismatch we saw on this machine.
        import io
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
    np.testing.assert_allclose(loaded.y, src.y if src.y is not None else np.zeros(src.X.shape[0]))
    assert loaded.wv is not None
    np.testing.assert_allclose(loaded.wv, src.wv if src.wv is not None else np.zeros(src.X.shape[1]))


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
        "R562": {"X1": rng.standard_normal((n, 225)), "X2": rng.standard_normal((n, 121)), "Y": rng.standard_normal(n)},
        "R568": {"X1": rng.standard_normal((n, 225)), "X2": rng.standard_normal((n, 121)), "Y": rng.standard_normal(n)},
    }
    fp = tmp_path / "inspect.mat"
    _save_struct_mat_v5(fp, np.linspace(900, 1700, 225), np.linspace(1700, 2500, 121), subsets)

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
