"""Dataset preview aggregates stay bounded and do not infer ambiguous roles."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from deerflow.community.nir.datasets.preview import build_preview


def test_csv_regression_counts_and_aggregate_spectrum(tmp_path: Path):
    path = tmp_path / "spectra.csv"
    path.write_text("moisture,1000,1100,1200\n1.5,1,2,3\n2.5,3,4,5\n3.5,5,6,7\n", encoding="utf-8")
    result = build_preview(path)
    assert result["analysis_type"] == "regression"
    assert (result["sample_count"], result["feature_count"]) == (3, 3)
    assert result["spectrum"][0] == {"x": 1000.0, "mean": 3.0, "low": 1.0, "high": 5.0}
    assert sum(bar["count"] for bar in result["targets"][0]["bars"]) == 3
    assert "X" not in result and "y" not in result
    json.dumps(result, allow_nan=False)


def test_chinese_classification_labels_are_not_spectral_columns(tmp_path: Path):
    path = tmp_path / "classes.csv"
    path.write_text("类别,1000,1100,1200\n小麦,1,2,3\n玉米,4,5,6\n小麦,7,8,9\n", encoding="utf-8")
    result = build_preview(path)
    assert result["status"] == "ready"
    assert result["analysis_type"] == "classification"
    assert result["feature_count"] == 3
    assert result["targets"][0]["bars"] == [{"label": "小麦", "count": 2}, {"label": "玉米", "count": 1}]


def test_mat_resolves_transpose_from_axis_and_target_evidence(tmp_path: Path):
    path = tmp_path / "spectra.mat"
    X = np.arange(21, dtype=float).reshape(3, 7)
    savemat(path, {"X": X.T, "y": np.array([1.5, 2.5, 3.5]), "wv": np.arange(7) + 1000})
    result = build_preview(path)
    assert result["status"] == "ready"
    assert (result["sample_count"], result["feature_count"]) == (3, 7)
    assert result["spectrum"][0]["mean"] == 7.0
    assert result["analysis_type"] == "regression"


def test_ambiguous_mat_does_not_pick_an_arbitrary_matrix(tmp_path: Path):
    path = tmp_path / "ambiguous.mat"
    savemat(path, {"A": np.ones((4, 10)), "B": np.ones((4, 10)), "y": np.arange(4) + 0.5, "wv": np.arange(10)})
    result = build_preview(path)
    assert result["status"] == "needs_mapping"
    assert result["spectrum"] == []
    assert str(tmp_path) not in json.dumps(result)


def test_numeric_class_labels_respect_confirmed_profile(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text("target,1000,1100,1200\n0,1,2,3\n1,2,3,4\n0,3,4,5\n", encoding="utf-8")
    result = build_preview(path, {"task_type": "classification", "mapping": {}})
    assert result["analysis_type"] == "classification"
    assert result["targets"][0]["kind"] == "classification"


def test_mixed_numeric_and_class_targets_do_not_choose_an_analysis_type(tmp_path: Path):
    path = tmp_path / "mixed.csv"
    path.write_text("class,moisture,1000,1100,1200\nA,1.5,1,2,3\nB,2.5,2,3,4\nA,3.5,3,4,5\n", encoding="utf-8")
    result = build_preview(path)
    assert result["status"] == "ready"
    assert result["analysis_type"] == "unknown"
    assert {target["kind"] for target in result["targets"]} == {"classification", "regression"}


def test_hdf5_mat_preview_supports_matlab_userblock_and_rejects_external_links(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "spectra.mat"
    with h5py.File(path, "w", userblock_size=512) as root:
        root["X"] = np.arange(21).reshape(3, 7).T
        root["y"] = np.array([1.5, 2.5, 3.5])
        root["wv"] = np.arange(7) + 1000
    with path.open("r+b") as stream:
        stream.write(b"MATLAB 7.3 MAT-file, HDF5 schema 1.00".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x02IM")
    result = build_preview(path)
    assert result["status"] == "ready"
    assert result["sample_count"] == 3
    with h5py.File(path, "a") as root:
        root["foreign"] = h5py.ExternalLink(str(tmp_path / "outside.h5"), "/X")
    rejected = build_preview(path)
    assert rejected["status"] == "needs_mapping"
    assert str(tmp_path) not in json.dumps(rejected)


def test_compressed_large_mat_is_rejected_before_array_loading(tmp_path: Path, monkeypatch):
    import deerflow.community.nir.datasets.preview as preview_module

    path = tmp_path / "compressed.mat"
    savemat(path, {"X": np.zeros((501, 10000))}, do_compression=True)

    def fail_load(*args, **kwargs):
        raise AssertionError("Large matrix must not be loaded")

    monkeypatch.setattr(preview_module, "load_mat", fail_load)
    result = build_preview(path)
    assert result["status"] == "needs_mapping"


def test_large_preview_is_bounded_and_nonfinite_values_are_json_safe(tmp_path: Path):
    path = tmp_path / "large.mat"
    X = np.ones((800, 500))
    X[:, 3] = np.nan
    savemat(path, {"X": X, "y": np.arange(800) + 0.5, "wv": np.arange(500)})
    result = build_preview(path)
    assert result["sample_count"] == 800
    assert result["preview_sample_count"] == 512
    assert len(result["spectrum"]) <= 256
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("suffix", ["csv", "txt"])
def test_spectra_only_files_remain_unconfirmed(tmp_path: Path, suffix: str):
    path = tmp_path / f"spectra.{suffix}"
    path.write_text("1,2,3\n2,3,4\n3,4,5\n", encoding="utf-8")
    result = build_preview(path)
    assert result["status"] == "ready"
    assert result["analysis_type"] == "unknown"
    assert result["targets"] == []
