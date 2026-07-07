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
