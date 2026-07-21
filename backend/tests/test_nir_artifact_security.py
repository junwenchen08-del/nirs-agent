"""Security contracts for NIR user-data and model artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deerflow.community.nir._common import (
    _load_npz_safely,
    _load_trusted_model_artifact,
    _write_trusted_model_artifact,
)


def test_safe_npz_loads_numeric_and_unicode_arrays(tmp_path: Path) -> None:
    path = tmp_path / "safe.npz"
    np.savez(path, X=np.ones((2, 3)), names=np.asarray(["蛋白", "水分"], dtype=str))

    loaded = _load_npz_safely(str(path))

    assert loaded["X"].shape == (2, 3)
    assert loaded["names"].tolist() == ["蛋白", "水分"]


def test_safe_npz_rejects_pickle_backed_object_arrays(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.npz"
    np.savez(path, X=np.ones((2, 3)), names=np.asarray(["protein"], dtype=object))

    with pytest.raises(ValueError, match="Unsafe legacy NPZ"):
        _load_npz_safely(str(path))


def test_model_loader_rejects_uploaded_pickle_even_with_manifest(tmp_path: Path) -> None:
    path = tmp_path / "uploaded.pkl"
    _write_trusted_model_artifact({"format": "nir_model_artifact"}, str(path))

    with pytest.raises(ValueError, match="Untrusted model path"):
        _load_trusted_model_artifact(str(path), "/mnt/user-data/uploads/uploaded.pkl")


def test_model_loader_detects_artifact_tampering(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    _write_trusted_model_artifact({"format": "nir_model_artifact", "value": 1}, str(path))
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="integrity check failed"):
        _load_trusted_model_artifact(str(path), "/mnt/user-data/outputs/model.pkl")


def test_model_loader_round_trips_verified_output(tmp_path: Path) -> None:
    path = tmp_path / "model.pkl"
    expected = {"format": "nir_model_artifact", "value": 7}
    _write_trusted_model_artifact(expected, str(path))

    loaded = _load_trusted_model_artifact(str(path), "/mnt/user-data/outputs/model.pkl")

    assert loaded == expected
