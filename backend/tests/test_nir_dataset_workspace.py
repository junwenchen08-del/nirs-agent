"""Synthetic-only workbook editing, immutable versions and fixed split contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import test_nir_dataset_service as dataset_fixtures
from scipy.io import savemat
from test_nir_dataset_service import _upload

from deerflow.community.nir.datasets.service import DatasetError
from deerflow.community.nir.datasets.workspace import Workbook, WorkspaceError, fixed_regression_partitions, load_workspace_export, read_workbook, validate_settings, workbook_spectra, write_workspace_csv

library = dataset_fixtures.library


def table(path: Path, rows=80):
    path.write_text("sample_id,moisture,1000,1100,1200\n" + "".join(f"corn-{i},{i + 0.5},{i},{i + 1},{i + 2}\n" for i in range(rows)), encoding="utf-8")
    return read_workbook(path)


def test_exact_rows_and_spectra_and_range_preserve_values(tmp_path):
    book = table(tmp_path / "data.csv")
    assert book.names[59] == "corn-59"
    assert book.fields == ["moisture"]
    assert book.values[59] == [59.5]
    assert book.usages == [0] * 80
    spectra = workbook_spectra(book, [59, 4])
    assert spectra["curves"][0]["values"] == [59.0, 60.0, 61.0]
    assert len(workbook_spectra(book)["curves"]) == 24
    with pytest.raises(WorkspaceError):
        workbook_spectra(book, [80])
    with pytest.raises(WorkspaceError):
        workbook_spectra(book, [1, 1])
    roles = [1] * 60 + [2] * 10 + [3] * 10
    settings = validate_settings(book, roles, [1100, 1200])
    output = tmp_path / "selected.csv"
    write_workspace_csv(book, settings, output)
    data = load_workspace_export(str(output))
    partitions = fixed_regression_partitions(data)
    assert partitions["X_cal"].shape == (60, 2)
    assert partitions["y_val"].tolist() == [i + 0.5 for i in range(60, 70)]
    np.testing.assert_array_equal(partitions["X_test"], book.X[70:, 1:])
    np.testing.assert_array_equal(partitions["wv"], [1100, 1200])
    first = output.read_bytes()
    write_workspace_csv(book, settings, output)
    assert output.read_bytes() == first


def test_chinese_class_labels_and_original_roles_are_separate(tmp_path):
    path = tmp_path / "class.csv"
    path.write_text("类别,用途,1000,1100,1200\n小麦,训练集,1,2,3\n玉米,验证集,2,3,4\n小麦,测试集,3,4,5\n", encoding="utf-8")
    book = read_workbook(path)
    assert book.fields == ["类别"]
    assert book.values == [["小麦"], ["玉米"], ["小麦"]]
    assert book.usages == [1, 2, 3]
    assert book.X.shape == (3, 3)


def test_numeric_mat_orientation_and_spectra_only_index(tmp_path):
    path = tmp_path / "data.mat"
    X = np.arange(30).reshape(5, 6)
    savemat(path, {"X": X.T, "y": np.arange(5), "wv": np.arange(6) + 1000})
    book = read_workbook(path)
    np.testing.assert_array_equal(book.X, X)
    assert book.usages == [0] * 5
    plain = tmp_path / "spectra.txt"
    plain.write_text("1 2 3\n2 3 4\n", encoding="utf-8")
    book = read_workbook(plain)
    assert book.axis is None
    assert book.fields == []
    assert book.coordinates.tolist() == [1.0, 2.0, 3.0]
    export = tmp_path / "index.csv"
    write_workspace_csv(book, validate_settings(book, [1, 2], [1, 3]), export)
    assert load_workspace_export(str(export))["wv"] is None


def test_multi_target_headers_and_group_membership_are_preserved(tmp_path):
    path = tmp_path / "multi.csv"
    path.write_text("sample_id,moisture,protein,fat,1000,1100,1200\na,1,2,3,1,2,3\nb,2,3,4,2,3,4\n", encoding="utf-8")
    book = read_workbook(path)
    assert len(book.fields) == len(book.values[0]) == 3
    path = tmp_path / "groups.csv"
    path.write_text("sample_id,moisture,batch,1000,1100,1200\na,1,A,1,2,3\nb,2,A,2,3,4\nc,3,B,3,4,5\nd,4,C,4,5,6\n", encoding="utf-8")
    book = read_workbook(path)
    with pytest.raises(WorkspaceError, match="同一批次"):
        validate_settings(book, [1, 2, 2, 3], [1000, 1200])
    settings = validate_settings(book, [1, 1, 2, 3], [1000, 1200])
    export = tmp_path / "groups-export.csv"
    write_workspace_csv(book, settings, export)
    partitions = fixed_regression_partitions(load_workspace_export(str(export)))
    assert partitions["groups_cal"].tolist() == ["A", "A"]
    assert partitions["groups_val"].tolist() == ["B"]


@pytest.mark.parametrize("usages,band", [([True] * 3, [1000, 1200]), ([1, 1], [1000, 1200]), ([5] * 3, [1000, 1200]), ([1] * 3, [1200, 1000]), ([1] * 3, [900, 1200]), ([1] * 3, [1050, 1090]), ([1] * 3, [float("nan"), 1200])])
def test_invalid_edits_are_rejected(tmp_path, usages, band):
    book = table(tmp_path / "data.csv", 3)
    with pytest.raises(WorkspaceError):
        validate_settings(book, usages, band)


def test_ordinary_non_utf8_csv_is_not_intercepted(tmp_path):
    path = tmp_path / "普通.csv"
    path.write_bytes("名称,数据\n样本,3".encode("gbk"))
    assert load_workspace_export(str(path)) is None


def test_export_growth_limit_does_not_change_source_values(tmp_path, monkeypatch):
    from deerflow.community.nir.datasets import workspace as module

    source = tmp_path / "data.csv"
    book = table(source, 3)
    before = source.read_bytes()
    monkeypatch.setattr(module, "MAX_FILE_BYTES", 40)
    with pytest.raises(WorkspaceError, match="64 MB"):
        write_workspace_csv(book, validate_settings(book, [1, 2, 3], [1000, 1200]), tmp_path / "export.csv")
    assert source.read_bytes() == before


def test_unassigned_and_missing_partitions_never_randomized(tmp_path):
    book = table(tmp_path / "data.csv", 6)
    output = tmp_path / "export.csv"
    for usages in ([1, 1, 1, 2, 3, 0], [1] * 6, [4, 0, 0, 0, 0, 0], [4] * 6):
        write_workspace_csv(book, validate_settings(book, usages, [1000, 1200]), output)
        with pytest.raises(WorkspaceError):
            fixed_regression_partitions(load_workspace_export(str(output)))


def test_sealed_ingestion_uses_exact_saved_partitions_without_random_split(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.dataset_session import SearchSessionStore
    from deerflow.community.nir.process import ProcessStore
    from deerflow.community.nir.workflow import start_workflow

    rng = np.random.default_rng(41)
    X = rng.normal(size=(80, 10))
    y = X[:, 0] * 2 + X[:, 1]
    roles = [1] * 60 + [2] * 10 + [3] * 10
    book = Workbook(X, np.arange(10) + 1000, [f"sample-{i}" for i in range(80)], ["水分"], [[float(v)] for v in y], roles, {})
    path = tmp_path / "saved.csv"
    write_workspace_csv(book, validate_settings(book, roles, [1002, 1008]), path)
    service = SearchSessionStore(ProcessStore(tmp_path / "private"))
    monkeypatch.setattr(search_tools, "_service", lambda runtime: service)
    monkeypatch.setattr(search_tools, "_resolve", lambda *args, **kwargs: str(path))
    import nir_core.model.evaluation as evaluation

    monkeypatch.setattr(evaluation, "split_dataset", lambda *args, **kwargs: pytest.fail("Saved membership must not be randomized"))
    workflow = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/saved.csv", domain="default", analyte="moisture", unit="%", validation_goal="internal_holdout", search_protocol="sealed_holdout")
    runtime = SimpleNamespace(state={"nir_workflow": workflow}, tool_call_id="prepare", context={}, config={})
    result = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(result.update["messages"][-1].content)
    assert payload.get("status") != "error", payload
    assert payload["test_status"] == "sealed"
    npz = next((tmp_path / "private").rglob("search.npz"))
    with np.load(npz, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["X_cal"], X[:60, 2:9])
        np.testing.assert_array_equal(data["y_val"], y[60:70])
        assert "X_test" not in data


def test_classifier_honors_saved_roles_and_records_original_row_indices(tmp_path, monkeypatch):
    import nir_core.model.classification as core_classification

    from deerflow.community.nir import classification

    rng = np.random.default_rng(74)
    labels = np.asarray(["小麦" if i % 2 else "玉米" for i in range(80)])
    X = rng.normal(scale=0.2, size=(80, 10)) + (np.arange(80) % 2)[:, None] * 3
    roles = [4] + [1] * 59 + [2] * 10 + [3] * 10
    book = Workbook(X, np.arange(10) + 1000, [f"sample-{i}" for i in range(80)], ["品种"], [[v] for v in labels], roles, {})
    path = tmp_path / "saved.csv"
    write_workspace_csv(book, validate_settings(book, roles, [1000, 1009]), path)
    monkeypatch.setattr(classification, "_resolve", lambda runtime, path, **kwargs: str(tmp_path / path.rsplit("/", 1)[-1]))
    monkeypatch.setattr(core_classification, "stratified_three_way_split", lambda *args, **kwargs: pytest.fail("Saved membership must not be randomized"))
    result = classification.nir_train_classifier_tool.func(
        runtime=MagicMock(),
        file_path="/mnt/user-data/uploads/saved.csv",
        pipeline_steps="[]",
        methods='["logistic"]',
        model_output="/mnt/user-data/outputs/model.pkl",
        metrics_output="/mnt/user-data/outputs/metrics.json",
        report_output="/mnt/user-data/outputs/report.md",
    )
    payload = json.loads(result)
    assert payload["status"] == "ok", payload
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["split"]["strategy"] == "user_named_partitions"
    assert metrics["split"]["random_state"] is None
    assert metrics["partitions"]["calibration"]["sample_indices"] == list(range(1, 60))
    assert metrics["partitions"]["holdout"]["sample_indices"] == list(range(70, 80))
    assert metrics["validation_scope"] == "independent_holdout_not_external"


@pytest.mark.parametrize("roles", [[1] * 20 + [2] * 6 + [3] * 5 + [0], [1] * 32])
def test_classifier_rejects_incomplete_fixed_roles_before_training(tmp_path, monkeypatch, roles):
    import nir_core.model.classification as core_classification

    from deerflow.community.nir import classification

    X = np.random.default_rng(74).normal(size=(32, 8))
    book = Workbook(X, np.arange(8) + 1000, [f"s{i}" for i in range(32)], ["品种"], [["小麦" if i % 2 else "玉米"] for i in range(32)], roles, {})
    source = tmp_path / "saved.csv"
    write_workspace_csv(book, validate_settings(book, roles, [1000, 1007]), source)
    monkeypatch.setattr(classification, "_resolve", lambda *a, **k: str(source))
    monkeypatch.setattr(core_classification, "stratified_three_way_split", lambda *a, **k: pytest.fail("Explicit roles must not be randomized"))
    monkeypatch.setattr(core_classification, "select_classifier", lambda *a, **k: pytest.fail("Incomplete roles must not train"))
    result = json.loads(classification.nir_train_classifier_tool.func(runtime=MagicMock(), file_path="/mnt/user-data/uploads/saved.csv", methods='["logistic"]', pipeline_steps="[]"))
    assert result["status"] == "error" and "分配" in result["error"]


@pytest.mark.asyncio
async def test_owned_version_save_conflict_reload_and_verified_attachment(library):
    service, paths = library
    content = "sample_id,moisture,1000,1100,1200\n" + "".join(f"s{i},{i + 0.5},{i},{i + 1},{i + 2}\n" for i in range(10))
    virtual = _upload(paths, "alice", "thread1", "data.csv", content.encode())
    saved = await service.save_from_thread_upload("alice", "thread1", virtual, "光谱", save_confirmed=True)
    id = saved["id"]
    original = paths.user_nir_dataset_dir("alice", id) / "source.csv"
    with pytest.raises(DatasetError):
        await service.get_workspace("bob", id)
    initial = await service.get_workspace("alice", id)
    roles = [1] * 6 + [2] * 2 + [3] * 2
    first = await service.save_workspace("alice", id, source_sha256=saved["sha256"], profile_id=None, usages=roles, band_range=[1100, 1200])
    assert initial["version"] == 0 and first["version"] == 1
    with pytest.raises(DatasetError) as exc:
        await service.save_workspace("alice", id, source_sha256=saved["sha256"], profile_id=None, usages=roles, band_range=[1000, 1200])
    assert exc.value.code == "workspace_conflict"
    recovered = await service.get_workspace("alice", id)
    assert recovered["usages"] == roles and recovered["range"] == [1100.0, 1200.0]
    attached = await service.attach_to_thread("alice", id, "source-thread", profile_id=None, apply_workspace=True, idempotency_key="workspace:1")
    assert attached["profile_id"] == first["profile_id"]
    assert attached["workspace_version"] == 1
    destination = paths.sandbox_uploads_dir("source-thread", user_id="alice") / attached["virtual_path"].rsplit("/", 1)[-1]
    data = load_workspace_export(str(destination))
    assert data["X"].shape == (10, 2)
    assert data["usage"].tolist() == ["train"] * 6 + ["validation"] * 2 + ["test"] * 2
    assert destination.stat().st_size == attached["attachment_size_bytes"]
    reused = await service.attach_to_thread("alice", id, "source-thread", profile_id=None, apply_workspace=True, idempotency_key="workspace:1")
    assert reused["reused_existing"]
    second_roles = roles.copy()
    second_roles[5] = 2
    second = await service.save_workspace("alice", id, source_sha256=saved["sha256"], profile_id=first["profile_id"], usages=second_roles, band_range=[1000, 1200])
    assert second["version"] == 2
    assert load_workspace_export(str(destination))["usage"][5] == "train"
    assert original.read_text() == content
    assert len(await service.list_datasets("alice")) == 1
    json.dumps(recovered, allow_nan=False)
    original.write_text("tampered", encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        await service.get_workspace("alice", id)
    assert exc.value.code == "source_hash_mismatch"
