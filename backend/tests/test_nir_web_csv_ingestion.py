"""Explicit single-target CSV headers survive governed web ingestion."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from nir_core.io.loaders import load_csv

from deerflow.community.nir import search_tools
from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.process import ProcessStore
from deerflow.community.nir.workflow import start_workflow


def source(tmp_path, target_column=0):
    rng = np.random.default_rng(510)
    X = rng.normal(size=(60, 8))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.02, size=60)
    wavelengths = np.arange(900, 1700, 100, dtype=float)
    matrix = np.insert(X, target_column, y, axis=1)
    headers = list(map(str, wavelengths))
    headers.insert(target_column, "protein")
    path = tmp_path / "spectra.csv"
    np.savetxt(path, matrix, delimiter=",", header=",".join(headers), comments="")
    return path, X, y, wavelengths


@pytest.mark.parametrize("target_column", [0, 3, 8])
@pytest.mark.parametrize("wavelength_row", [None, 0])
def test_named_single_target_with_numeric_axis_is_header(tmp_path, target_column, wavelength_row):
    path, X, y, wavelengths = source(tmp_path, target_column)
    result = load_csv(str(path), y_col=target_column, wv_row=wavelength_row)
    assert result.X.shape == (60, 8)
    np.testing.assert_allclose(result.X, X)
    np.testing.assert_allclose(result.y, y)
    np.testing.assert_array_equal(result.wv, wavelengths)


def test_numeric_first_observation_is_not_header(tmp_path):
    path, X, y, _ = source(tmp_path)
    np.savetxt(path, np.column_stack([y, X]), delimiter=",")
    result = load_csv(str(path), y_col=0)
    np.testing.assert_allclose(result.X, X)
    np.testing.assert_allclose(result.y, y)
    assert result.wv is None


@pytest.mark.parametrize("missing", ["", "NA", "n/a", "nan", "null", "none", "-"])
def test_missing_first_reference_stays_a_sample(tmp_path, missing):
    path = tmp_path / "missing.csv"
    path.write_text(f"{missing},0.1,0.2,0.3\n1,0.4,0.5,0.6\n2,0.7,0.8,0.9\n", encoding="utf-8")
    result = load_csv(str(path), y_col=0)
    assert result.X.shape == (3, 3)
    assert np.isnan(result.y[0])
    np.testing.assert_allclose(result.X[0], [0.1, 0.2, 0.3])
    assert result.wv is None


@pytest.mark.parametrize("target_column", [-1, 9])
def test_out_of_range_target_is_rejected(tmp_path, target_column):
    path, *_ = source(tmp_path)
    with pytest.raises(ValueError, match="y_col.*out of range"):
        load_csv(str(path), y_col=target_column)


def test_wrong_numeric_pin_does_not_remap_named_target(tmp_path):
    path, *_ = source(tmp_path)
    result = load_csv(str(path), y_col=1)
    assert result.X.shape == (61, 8)
    assert result.y[0] == 900
    assert np.isnan(result.X[0, 0])
    assert result.wv is None


def test_unpinned_legacy_layout_is_not_changed(tmp_path):
    path, *_ = source(tmp_path)
    result = load_csv(str(path), auto_layout=False)
    assert result.X.shape == (61, 9)
    assert result.y is None and result.wv is None


def test_mixed_header_reaches_sealed_prepare_without_test_statistics(tmp_path, monkeypatch):
    path, *_ = source(tmp_path)
    service = SearchSessionStore(ProcessStore(tmp_path / "private"))
    monkeypatch.setattr(search_tools, "_service", lambda _: service)
    monkeypatch.setattr(search_tools, "_resolve", lambda *args, **kwargs: str(path))
    workflow = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/spectra.csv",
        domain="default",
        analyte="protein",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        decision_strategy="llm_autonomous",
        autonomous_rounds=True,
        autonomous_delivery=True,
    )
    runtime = SimpleNamespace(state={"nir_workflow": workflow}, context={}, config={}, tool_call_id="prepare")
    result = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3, y_col=0)
    payload = json.loads(result.update["messages"][-1].content)
    assert payload.get("status") != "error", payload
    assert payload["partition_counts"] == {"calibration": 36, "validation": 12}
    assert payload["test_status"] == "sealed"
    assert "test" not in payload and "metrics" not in payload
    with service.process._connection() as db:
        session = service._session(db, payload["session_id"])
    assert session["stage"] == "searching" and session["final"] is None
    assert session["fits"] == session["experiments"] == 0
