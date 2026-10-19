"""Exact, immutable library partitions from new trusted ingestion; synthetic data only."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import savemat
from sqlalchemy import select
from test_nir_dataset_service import library as library

from deerflow.community.nir import search_tools
from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.datasets.service import DatasetError
from deerflow.community.nir.datasets.workspace import (
    fixed_regression_partitions,
    load_workspace_export,
)
from deerflow.community.nir.process import ProcessStore
from deerflow.community.nir.workflow import start_workflow
from deerflow.persistence.nir_library.model import NIRDatasetProfileRow


async def prepare(library, monkeypatch, *, virtual=None, project="partition-project", autonomous=False):
    service, paths = library
    if virtual is None:
        source = paths.sandbox_uploads_dir("source-thread", user_id="alice") / "spectra.mat"
        source.parent.mkdir(parents=True, exist_ok=True)
        X = np.random.default_rng(714).normal(size=(60, 8))
        savemat(source, {"X": X, "y": X[:, 0] * 2 + X[:, 1], "wv": np.arange(1000, 1008)})
        virtual = "/mnt/user-data/uploads/spectra.mat"
        dataset = await service.save_from_thread_upload("alice", "source-thread", virtual, "光谱", save_confirmed=True)
    else:
        source = paths.sandbox_uploads_dir("source-thread", user_id="alice") / virtual.rsplit("/", 1)[-1]
        dataset = (await service.list_datasets("alice"))[0]
    store = SearchSessionStore(ProcessStore(paths.thread_dir("source-thread", user_id="alice") / "nir-process"))
    monkeypatch.setattr(search_tools, "_service", lambda _: store)
    monkeypatch.setattr(search_tools, "_resolve", lambda *args, **kwargs: str(source))
    from deerflow.community.nir import dataset_tools

    monkeypatch.setattr(dataset_tools, "_service", lambda: service)
    workflow = start_workflow(
        task_type="calibration",
        data_path=virtual,
        project_id=project,
        analyte="水分",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        **(
            {
                "decision_strategy": "llm_autonomous",
                "autonomous_rounds": True,
                "autonomous_delivery": True,
            }
            if autonomous
            else {}
        ),
    )
    runtime = SimpleNamespace(
        state={"nir_workflow": workflow},
        tool_call_id="prepare",
        context={"thread_id": "source-thread", "user_id": "alice"},
        config={},
    )
    monkeypatch.setattr(
        "deerflow.community.nir.dataset_partitions.resolve_runtime_user_id",
        lambda _: "alice",
    )
    result = await search_tools.nir_prepare_search_tool.coroutine(runtime=runtime, cv_folds=3)
    payload = json.loads(result.update["messages"][-1].content)
    assert payload.get("status") != "error", payload
    return service, paths, dataset, store, runtime, result


@pytest.mark.asyncio
async def test_automatic_partition_is_saved_once_without_replacing_current_profile(library, monkeypatch):
    service, paths, dataset, store, runtime, result = await prepare(library, monkeypatch)
    workflow = result.update["nir_workflow"]
    assert workflow["dataset_profile_id"] is None  # Original input lineage stays intact.
    profile_id = workflow["dataset_partition"]["profile_id"]
    profiles = await service.list_profiles("alice", dataset["id"])
    profile = next(p for p in profiles if p["id"] == profile_id)
    assert profile["profile_status"] == "split"
    assert profile["confirmed_at"] is None
    split = profile["mapping"]["partition"]
    assert split["counts"] == {"calibration": 36, "validation": 12, "test": 12}
    assert sorted(sum(split["indices"].values(), [])) == list(range(60))
    assert (await service.get_workspace("alice", dataset["id"]))["profile_id"] is None
    pinned = await service.get_workspace("alice", dataset["id"], profile_id=profile_id)
    assert pinned["read_only"] and pinned["partition"]["method"] == "random"
    assert pinned["usages"].count(3) == 12
    runtime.state["nir_workflow"] = workflow
    repeated = await search_tools.nir_prepare_search_tool.coroutine(runtime=runtime, cv_folds=3)
    assert repeated.update["nir_workflow"]["dataset_partition"]["profile_id"] == profile_id
    assert len(await service.list_profiles("alice", dataset["id"])) == len(profiles)
    # Neither sample membership nor Test statistics are sent to the model.
    assert "indices" not in result.update["messages"][-1].content
    assert "y_test" not in result.update["messages"][-1].content


@pytest.mark.asyncio
async def test_reusing_partition_preserves_actual_array_order_and_source_bytes(library, monkeypatch):
    service, paths, dataset, store, runtime, result = await prepare(library, monkeypatch)
    sid = result.update["nir_workflow"]["search_session_id"]
    profile_id = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    original = paths.user_nir_dataset_dir("alice", dataset["id"]) / "source.mat"
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.parent.iterdir() if p.is_file()}
    attached = await service.attach_to_thread(
        "alice",
        dataset["id"],
        "source-thread",
        profile_id=profile_id,
        idempotency_key="reuse-split",
        apply_workspace=True,
    )
    copy = paths.sandbox_uploads_dir("source-thread", user_id="alice") / attached["virtual_path"].rsplit("/", 1)[-1]
    partitions = fixed_regression_partitions(load_workspace_export(str(copy)))
    with np.load(store._dir(sid) / "search.npz", allow_pickle=False) as saved:
        for key in ("X_cal", "y_cal", "X_val", "y_val", "wv"):
            np.testing.assert_array_equal(partitions[key], saved[key])
    _, _, _, _, _, reused = await prepare(library, monkeypatch, virtual=attached["virtual_path"], project="reuse-project")
    assert reused.update["nir_workflow"]["dataset_partition"]["profile_id"] == profile_id
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in original.parent.iterdir() if p.is_file()} == before


@pytest.mark.asyncio
async def test_partition_is_owner_scoped_and_cannot_be_confirmed_or_forged(library, monkeypatch):
    service, _, dataset, _, _, result = await prepare(library, monkeypatch)
    pid = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    with pytest.raises(DatasetError):
        await service.get_workspace("bob", dataset["id"], profile_id=pid)
    with pytest.raises(DatasetError):
        await service.confirm_profile("alice", dataset["id"], pid)
    with pytest.raises(DatasetError):
        await service.create_profile(
            "alice",
            dataset["id"],
            task_type="calibration",
            schema_status="confirmed_mapping",
            mapping={"partition": {"indices": {}}},
        )
    async with service.session_factory() as db:
        row = await db.scalar(select(NIRDatasetProfileRow).where(NIRDatasetProfileRow.id == pid))
        row.mapping_json = {**row.mapping_json, "partition": {}}
        await db.commit()
    with pytest.raises(DatasetError):
        await service.attach_to_thread(
            "alice",
            dataset["id"],
            "source-thread",
            profile_id=pid,
            idempotency_key="tampered",
        )


@pytest.mark.asyncio
async def test_model_promotion_links_verified_partition_instead_of_original_profile(library, monkeypatch):
    from test_nir_model_service import _registered_package

    from deerflow.community.nir.models.service import ModelService

    service, paths, dataset, store, _, result = await prepare(library, monkeypatch, autonomous=True)
    workflow = result.update["nir_workflow"]
    sid = workflow["search_session_id"]
    with store.process._connection(write=True) as db:
        session = store._session(db, sid)
        session.update(
            stage="finalized",
            freeze={"experiment_id": "exp-synthetic", "model_hash": "f" * 64},
        )
        store._save(db, session)
    audit = {
        "session_id": sid,
        "freeze": session["freeze"],
        "task": {key: value for key, value in session["task_snapshot"].items() if key != "_workflow_snapshot"},
        "dataset_partition": session["library_partition"],
    }
    version, registered = _registered_package(
        paths,
        training_hash=session["search_hash"],
        extra_metrics={
            "autonomous_decision_audit": audit,
            "scientific_validation": session["scientific_validation"],
        },
    )
    # A restored registration workflow can still carry the original dataset
    # Profile. The finalized model/session receipt is authoritative.
    registered["dataset_id"] = dataset["id"]
    models = ModelService(service.session_factory, paths, service.policy)
    saved = await models.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=registered)
    assert saved["source_dataset_id"] == dataset["id"]
    assert saved["source_profile_id"] == workflow["dataset_partition"]["profile_id"]
    before = {p.name: p.read_bytes() for p in paths.user_nir_model_version_dir("alice", "tablet_assay", version).iterdir()}
    detail = await models.get_report("alice", "tablet_assay", version)
    assert detail["dataset"]["partition"]["counts"] == {
        "calibration": 36,
        "validation": 12,
        "test": 12,
    }
    assert detail["model"]["source_profile_id"] == detail["dataset"]["profile_id"]
    assert {p.name: p.read_bytes() for p in paths.user_nir_model_version_dir("alice", "tablet_assay", version).iterdir()} == before


@pytest.mark.asyncio
async def test_saved_parsing_and_selected_bands_are_pinned_without_superseding_them(library, monkeypatch):
    service, paths, dataset, _, _, _ = await prepare(library, monkeypatch)
    current = await service.save_workspace(
        "alice",
        dataset["id"],
        source_sha256=dataset["sha256"],
        profile_id=None,
        usages=[0] * 60,
        band_range=[1002, 1006],
    )
    attached = await service.attach_to_thread(
        "alice",
        dataset["id"],
        "source-thread",
        profile_id=current["profile_id"],
        idempotency_key="parsed-input",
        apply_workspace=True,
    )
    _, _, _, _, _, result = await prepare(library, monkeypatch, virtual=attached["virtual_path"], project="selected-bands")
    pid = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    workspace = await service.get_workspace("alice", dataset["id"], profile_id=pid)
    assert workspace["partition"]["parent_profile_id"] == current["profile_id"]
    assert workspace["range"] == [1002, 1006]
    assert (await service.get_workspace("alice", dataset["id"]))["profile_id"] == current["profile_id"]
    preview = await service.preview_dataset("alice", dataset["id"], profile_id=pid)
    assert preview["status"] == "ready" and preview["feature_count"] == 5


@pytest.mark.asyncio
async def test_snapshot_storage_failure_does_not_change_locked_partitions(library, monkeypatch):
    from deerflow.community.nir import dataset_partitions

    monkeypatch.setattr(
        dataset_partitions,
        "save_partition",
        lambda *args: (_ for _ in ()).throw(DatasetError("storage_error", "无法保存划分版本。")),
    )
    service, _, dataset, store, _, result = await prepare(library, monkeypatch)
    outcome = result.update["nir_workflow"]["dataset_partition"]
    assert outcome["status"] == "failed"
    assert (store._dir(result.update["nir_workflow"]["search_session_id"]) / "search.npz").is_file()
    assert not await service.list_profiles("alice", dataset["id"])


@pytest.mark.asyncio
async def test_csv_targets_are_preserved_when_splitting_row_identities(library, monkeypatch):
    from test_nir_dataset_service import _upload

    service, paths = library
    X = np.random.default_rng(441).normal(size=(60, 8))
    y = 10.125 + X[:, 0] * 1.7
    content = "水分," + ",".join(str(i + 1000) for i in range(8)) + "\n"
    content += "".join(",".join(str(value) for value in [target, *row]) + "\n" for row, target in zip(X, y, strict=True))
    virtual = _upload(paths, "alice", "source-thread", "data.csv", content.encode())
    dataset = await service.save_from_thread_upload("alice", "source-thread", virtual, "水分", save_confirmed=True)
    _, _, _, store, _, result = await prepare(library, monkeypatch, virtual=virtual)
    pid = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    assert pid is not None
    profile = next(p for p in await service.list_profiles("alice", dataset["id"]) if p["id"] == pid)
    chosen = profile["mapping"]["partition"]["indices"]["calibration"]
    with np.load(store._dir(result.update["nir_workflow"]["search_session_id"]) / "search.npz", allow_pickle=False) as cache:
        np.testing.assert_allclose(cache["y_cal"], y[chosen], rtol=1e-14)
        np.testing.assert_allclose(cache["X_cal"], X[chosen], rtol=1e-14)


@pytest.mark.asyncio
async def test_fixed_input_keeps_exclusions_and_current_profile(library, monkeypatch):
    service, _, dataset, _, _, _ = await prepare(library, monkeypatch)
    roles = [1] * 36 + [2] * 12 + [3] * 10 + [4] * 2
    original = await service.save_workspace("alice", dataset["id"], source_sha256=dataset["sha256"], profile_id=None, usages=roles, band_range=[1000, 1007])
    attached = await service.attach_to_thread("alice", dataset["id"], "source-thread", profile_id=original["profile_id"], idempotency_key="fixed-input", apply_workspace=True)
    _, _, _, _, _, result = await prepare(library, monkeypatch, virtual=attached["virtual_path"], project="fixed-input")
    pinned = await service.get_workspace("alice", dataset["id"], profile_id=result.update["nir_workflow"]["dataset_partition"]["profile_id"])
    assert pinned["partition"]["method"] == "fixed"
    assert pinned["usages"] == roles
    assert pinned["partition"]["counts"]["test"] == 10
    with pytest.raises(DatasetError):
        await service.save_workspace("alice", dataset["id"], source_sha256=dataset["sha256"], profile_id=pinned["profile_id"], usages=roles, band_range=[1000, 1007])


@pytest.mark.asyncio
async def test_library_reads_and_reuse_do_not_open_private_search_or_test(library, monkeypatch):
    service, _, dataset, _, _, result = await prepare(library, monkeypatch)
    pid = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Library reads must not open private NPZ caches"))
    assert (await service.get_workspace("alice", dataset["id"], profile_id=pid))["read_only"]
    assert (await service.preview_dataset("alice", dataset["id"], profile_id=pid))["status"] == "ready"
    await service.attach_to_thread("alice", dataset["id"], "source-thread", profile_id=pid, idempotency_key="read-only-reuse")


@pytest.mark.asyncio
async def test_retry_recovers_committed_version_after_missing_session_receipt(library, monkeypatch):
    from deerflow.community.nir.dataset_partitions import save_partition

    service, _, dataset, store, _, result = await prepare(library, monkeypatch)
    sid = result.update["nir_workflow"]["search_session_id"]
    with store.process._connection(write=True) as db:
        frozen = store._session(db, sid)
        receipt = frozen.pop("library_partition")
        store._save(db, frozen)
    assert await save_partition(service, "alice", "source-thread", sid) == receipt
    assert len(await service.list_profiles("alice", dataset["id"])) == 1


@pytest.mark.asyncio
async def test_parser_changes_cannot_silently_change_an_immutable_partition(library, monkeypatch):
    from deerflow.community.nir.datasets import workspace

    service, _, dataset, _, _, result = await prepare(library, monkeypatch)
    pid = result.update["nir_workflow"]["dataset_partition"]["profile_id"]
    loader = workspace.load_mat

    def changed_loader(*args, **kwargs):
        loaded = loader(*args, **kwargs)
        return loaded.model_copy(update={"X": loaded.X + 0.5})

    monkeypatch.setattr(workspace, "load_mat", changed_loader)
    with pytest.raises(DatasetError, match="固定划分不一致"):
        await service.get_workspace("alice", dataset["id"], profile_id=pid)
