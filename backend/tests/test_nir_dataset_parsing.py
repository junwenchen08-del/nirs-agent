"""Synthetic visual parsing contracts, source integrity and exact reuse."""

import io
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import test_nir_dataset_service as dataset_fixtures
from scipy.io import savemat

from deerflow.community.nir.datasets.parsing import describe_file, resolve_selection
from deerflow.community.nir.datasets.service import DatasetError
from deerflow.community.nir.datasets.workspace import Workbook, fixed_regression_partitions, load_workspace_export, read_workbook, validate_settings, write_workspace_csv

library = dataset_fixtures.library


def nested_mat(path):
    savemat(
        path,
        {
            "Dataset": {
                "wn1": np.array([1000, 1100, 1200]),
                "wn2": np.array([1300, 1400]),
                "R562": {"X1": np.arange(18).reshape(6, 3), "X2": np.arange(12).reshape(6, 2), "Y": np.arange(6) + 0.5},
                "R568": {"X1": np.arange(12).reshape(4, 3), "X2": np.arange(8).reshape(4, 2), "Y": np.arange(4) + 10.5},
            }
        },
    )


def choose(schema, group="R562"):
    layout = next(item for item in schema["layouts"] if item["group"].endswith(group) and item["feature_count"] == 5)
    return {"layout_id": layout["id"], "target_id": layout["default_target"], "axis_id": layout["default_axis"]}


def test_nested_groups_and_bands_are_detected_without_picking_a_group(tmp_path):
    source = tmp_path / "data.mat"
    nested_mat(source)
    schema = describe_file(source)
    assert schema["suggested"] is None
    assert {item["sample_count"] for item in schema["layouts"]} == {4, 6}
    assert len(schema["layouts"]) == 6
    mapping = resolve_selection(source, choose(schema), "calibration")
    book = read_workbook(source, {"mapping": mapping, "task_type": "calibration"})
    assert book.X.shape == (6, 5)
    assert book.values[3] == [3.5]
    np.testing.assert_array_equal(book.axis, [1000, 1100, 1200, 1300, 1400])
    with pytest.raises(ValueError):
        resolve_selection(source, {"layout_id": "arbitrary-path"}, "calibration")


@pytest.mark.asyncio
async def test_parsing_preview_save_conflict_and_chat_use_keep_exact_group(library, tmp_path):
    service, paths = library
    source = tmp_path / "data.mat"
    nested_mat(source)
    content = source.read_bytes()
    dataset = await service.save_from_local_upload("alice", "nested.mat", io.BytesIO(content), len(content))
    schema = await service.get_parsing("alice", dataset["id"])
    selection = choose(schema)
    body = dict(source_sha256=dataset["sha256"], profile_id=None, selection=selection, task_type="calibration")
    preview = await service.preview_parsing("alice", dataset["id"], **body)
    assert preview["sample_count"] == 6 and preview["feature_count"] == 5
    assert await service.list_profiles("alice", dataset["id"]) == []
    saved = await service.save_parsing("alice", dataset["id"], **body)
    with pytest.raises(DatasetError) as exc:
        await service.save_parsing("alice", dataset["id"], **body)
    assert exc.value.code == "workspace_conflict"
    workspace = await service.get_workspace("alice", dataset["id"])
    assert workspace["sample_count"] == 6 and workspace["usages"] == [0] * 6
    attached = await service.attach_to_thread("alice", dataset["id"], "source-thread", profile_id=None, apply_workspace=True, idempotency_key="parsed")
    assert attached["profile_id"] == saved["id"]
    copy = paths.sandbox_uploads_dir("source-thread", user_id="alice") / attached["virtual_path"].rsplit("/", 1)[-1]
    data = load_workspace_export(str(copy))
    assert data["parsing_only"] is True
    np.testing.assert_array_equal(data["targets"][:, 0].astype(float), np.arange(6) + 0.5)
    assert fixed_regression_partitions(data)["y"].tolist() == (np.arange(6) + 0.5).tolist()
    assert data["X"].shape == (6, 5)
    assert (paths.user_nir_dataset_dir("alice", dataset["id"]) / "source.mat").read_bytes() == content
    with pytest.raises(DatasetError):
        await service.get_parsing("bob", dataset["id"])
    edited = await service.save_workspace("alice", dataset["id"], source_sha256=dataset["sha256"], profile_id=saved["id"], usages=[1, 1, 1, 2, 3, 3], band_range=[1100, 1400])
    retained = await service.save_parsing("alice", dataset["id"], **{**body, "profile_id": edited["profile_id"]})
    assert (await service.get_workspace("alice", dataset["id"]))["usages"] == [1, 1, 1, 2, 3, 3]
    assert (await service.get_workspace("alice", dataset["id"]))["range"] == [1100, 1400]
    await service.save_parsing("alice", dataset["id"], **{**body, "profile_id": retained["id"], "selection": choose(schema, "R568")})
    changed = await service.get_workspace("alice", dataset["id"])
    assert changed["sample_count"] == 4 and changed["usages"] == [0] * 4
    assert changed["range"] == [1000, 1400]
    assert load_workspace_export(str(copy))["X"].shape == (6, 5)


def test_table_manual_columns_and_chinese_names_need_no_json(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("样本编号,含量,A,B,C\n甲,1.5,1,2,3\n乙,2.5,2,3,4\n", encoding="utf-8")
    schema = describe_file(path)
    assert schema["format"] == "table"
    assert schema["columns"][0]["name"] == "样本编号"
    selection = {"spectral_start": 2, "spectral_end": 4, "target_columns": [1], "sample_column": 0}
    mapping = resolve_selection(path, selection, "calibration")
    book = read_workbook(path, {"mapping": mapping, "task_type": "calibration"})
    assert book.names == ["甲", "乙"]
    assert book.values == [[1.5], [2.5]]
    assert book.X.shape == (2, 3)
    with pytest.raises(ValueError):
        resolve_selection(path, {**selection, "target_columns": [2]}, "calibration")


def test_hdf5_selection_and_transpose_are_bounded_and_explicit(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "data.mat"
    with h5py.File(path, "w") as root:
        root["X"] = np.arange(24).reshape(3, 8)
        root["Y"] = np.arange(8)
        root["wv"] = np.array([1000, 1100, 1200])
    schema = describe_file(path)
    mapping = resolve_selection(path, schema["suggested"], "calibration")
    assert read_workbook(path, {"mapping": mapping}).X.shape == (8, 3)
    with h5py.File(path, "a") as root:
        root["foreign"] = h5py.ExternalLink(str(tmp_path / "outside.h5"), "/X")
    with pytest.raises(ValueError):
        describe_file(path)


def test_only_parsing_export_cannot_override_named_roles(tmp_path):
    book = Workbook(np.arange(12).reshape(4, 3), None, ["a", "b", "c", "d"], ["含量"], [[1], [2], [3], [4]], [0] * 4, {})
    path = tmp_path / "parsed.csv"
    write_workspace_csv(book, validate_settings(book, [0] * 4, [1, 3]), path, parsing_only=True)
    data = load_workspace_export(str(path))
    assert data["wv"] is None and fixed_regression_partitions(data)["X"].shape == (4, 3)
    path.write_text(path.read_text().replace("a,unassigned,1", "a,train,1"))
    with pytest.raises(ValueError, match="覆盖"):
        load_workspace_export(str(path))


def test_parsed_regression_keeps_existing_automatic_sealed_ingestion(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.dataset_session import SearchSessionStore
    from deerflow.community.nir.process import ProcessStore
    from deerflow.community.nir.workflow import start_workflow

    rng = np.random.default_rng(31)
    X = rng.normal(size=(80, 10))
    y = X[:, 0] * 2 + X[:, 1]
    book = Workbook(X, np.arange(10) + 1000, [f"sample-{i}" for i in range(80)], ["水分"], [[float(v)] for v in y], [0] * 80, {})
    path = tmp_path / "parsed.csv"
    write_workspace_csv(book, validate_settings(book, book.usages, [1002, 1008]), path, parsing_only=True)
    service = SearchSessionStore(ProcessStore(tmp_path / "private"))
    monkeypatch.setattr(search_tools, "_service", lambda runtime: service)
    monkeypatch.setattr(search_tools, "_resolve", lambda *args, **kwargs: str(path))
    workflow = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/parsed.csv", domain="default", analyte="moisture", unit="%", validation_goal="internal_holdout", search_protocol="sealed_holdout")
    runtime = SimpleNamespace(state={"nir_workflow": workflow}, tool_call_id="prepare", context={}, config={})
    result = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(result.update["messages"][-1].content)
    assert payload.get("status") != "error", payload
    assert payload["test_status"] == "sealed"
    with np.load(next((tmp_path / "private").rglob("search.npz")), allow_pickle=False) as data:
        assert data["X_cal"].shape[1] == 7
        assert "X_test" not in data


def test_parsed_classifier_keeps_automatic_split_when_no_roles_were_assigned(tmp_path, monkeypatch):
    from deerflow.community.nir import classification

    rng = np.random.default_rng(42)
    labels = ["小麦" if i % 2 else "玉米" for i in range(80)]
    X = rng.normal(scale=0.2, size=(80, 10)) + (np.arange(80) % 2)[:, None] * 3
    book = Workbook(X, np.arange(10) + 1000, [f"sample-{i}" for i in range(80)], ["品种"], [[v] for v in labels], [0] * 80, {})
    path = tmp_path / "parsed.csv"
    write_workspace_csv(book, validate_settings(book, book.usages, [1000, 1009]), path, parsing_only=True)
    monkeypatch.setattr(classification, "_resolve", lambda runtime, path, **kwargs: str(tmp_path / path.rsplit("/", 1)[-1]))
    payload = json.loads(
        classification.nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/parsed.csv",
            pipeline_steps="[]",
            methods='["logistic"]',
            model_output="/mnt/user-data/outputs/model.pkl",
            metrics_output="/mnt/user-data/outputs/metrics.json",
            report_output="/mnt/user-data/outputs/report.md",
        )
    )
    assert payload["status"] == "ok", payload
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["split"]["strategy"] != "user_named_partitions"
    assert metrics["split"]["random_state"] is not None


def test_compressed_nested_mat_is_rejected_before_allocating_arrays(tmp_path, monkeypatch):
    import scipy.io

    path = tmp_path / "nested-large.mat"
    savemat(path, {"Dataset": {"X": np.zeros((900, 10000))}}, do_compression=True)
    monkeypatch.setattr(scipy.io, "loadmat", lambda *args, **kwargs: pytest.fail("Oversized nested array must not be loaded"))
    with pytest.raises(ValueError, match="expanded"):
        describe_file(path)
