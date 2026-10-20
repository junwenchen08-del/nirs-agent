"""Saved model data links use immutable source identity, never filename guesses."""

import hashlib
from unittest.mock import AsyncMock

import numpy as np
import pytest
from scipy.io import savemat
from sqlalchemy import select
from test_nir_model_service import _registered_package
from test_nir_model_service import model_library as model_library

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.datasets.service import DatasetService
from deerflow.community.nir.decision_budget import resolve_policy
from deerflow.community.nir.models.report import complete_dimensions
from deerflow.community.nir.models.source import recorded_source
from deerflow.community.nir.process import ProcessStore
from deerflow.community.nir.search_tools import _load_search_mat
from deerflow.persistence.nir_library.model import NIRDatasetRow, NIRModelVersionRow


def legacy_metrics(digest):
    return {
        "protocol": "automated_analysis_three_way_holdout",
        "training_data_hash": digest,
        "reproducibility": {"schema_version": 1, "protocol": "automated_analysis_three_way_holdout", "input_sha256": digest, "random_state": 42},
    }


async def legacy_dataset(service, paths, owner="alice", thread="source-thread", content=b"target,1000,1100\n1,2,3\n2,3,4\n"):
    source = paths.sandbox_uploads_dir(thread, user_id=owner) / "tablets.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(content)
    dataset = await DatasetService(service.session_factory, paths, service.policy).save_from_thread_upload(owner, thread, "/mnt/user-data/uploads/tablets.csv", "药片数据", save_confirmed=True)
    return dataset


@pytest.mark.asyncio
async def test_legacy_file_hash_recovers_group_without_inventing_profile_or_writing_history(model_library, monkeypatch):
    service, paths = model_library
    digest = hashlib.sha256(b"target,1000,1100\n1,2,3\n2,3,4\n").hexdigest()
    version, workflow = _registered_package(paths, training_hash=digest, extra_metrics=legacy_metrics(digest))
    saved = await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert saved["source_dataset_id"] is None
    folder = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    dataset = await legacy_dataset(service, paths)

    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy file identity must not inspect session arrays or reconstruct splits")

    monkeypatch.setattr("deerflow.community.nir.models.service.recorded_source", forbidden)
    monkeypatch.setattr("deerflow.community.nir.models.service.verify_historical_source", forbidden)
    detail = await service.get_report("alice", "tablet_assay", version)
    assert detail["dataset"]["id"] == dataset["id"]
    assert detail["dataset"]["name"] == "药片数据"
    assert detail["dataset"]["profile_id"] is None
    assert detail["model"]["source_dataset_id"] == dataset["id"]
    assert detail["model"]["source_profile_id"] is None
    async with service.session_factory() as db:
        row = await db.scalar(select(NIRModelVersionRow))
        assert row.source_dataset_id is None and row.source_profile_id is None
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["input_mismatch", "training_mismatch", "invalid_hash", "wrong_protocol", "wrong_manifest_protocol", "wrong_schema", "sealed_audit", "other_owner", "same_name", "corrupt_bytes", "corrupt_manifest"])
async def test_legacy_group_recovery_requires_exact_owned_verified_file(model_library, case):
    service, paths = model_library
    dataset = await legacy_dataset(service, paths, owner="bob" if case == "other_owner" else "alice", thread="bob-thread" if case == "other_owner" else "source-thread")
    metrics = legacy_metrics(dataset["sha256"])
    if case == "input_mismatch":
        metrics["reproducibility"]["input_sha256"] = "b" * 64
    elif case in {"training_mismatch", "same_name"}:
        metrics["training_data_hash"] = "b" * 64
        if case == "same_name":
            metrics["reproducibility"]["input_sha256"] = "b" * 64
    elif case == "invalid_hash":
        metrics["training_data_hash"] = metrics["reproducibility"]["input_sha256"] = "not-a-hash"
    elif case == "wrong_protocol":
        metrics["protocol"] = "sealed_holdout"
    elif case == "wrong_manifest_protocol":
        metrics["reproducibility"]["protocol"] = "unknown"
    elif case == "wrong_schema":
        metrics["reproducibility"]["schema_version"] = 2
    elif case == "sealed_audit":
        metrics["autonomous_decision_audit"] = {"session_id": "search_missing"}
    elif case in {"corrupt_bytes", "corrupt_manifest"}:
        directory = paths.user_nir_dataset_dir("alice", dataset["id"])
        if case == "corrupt_manifest":
            (directory / "manifest.json").write_text("{}")
        else:
            (directory / "source.csv").write_bytes(b"changed")
    async with service.session_factory() as db:
        assert await service._recover_dataset(db, "alice", "source-thread", metrics) is None


def source_fixture(paths, *, recorded_hash=False, source=None, data=None, project="project"):
    from nir_core.model.evaluation import split_dataset

    if source is None:
        source = paths.sandbox_uploads_dir("source-thread", user_id="alice") / "small_spectra.mat"
        source.parent.mkdir(parents=True, exist_ok=True)
        X = np.random.default_rng(71).normal(size=(60, 8))
        savemat(source, {"X": X, "y": X[:, 0] * 2 + X[:, 1], "wv": np.arange(1000, 1008)})
    virtual = "/mnt/user-data/uploads/" + source.name
    loaded = data or _load_search_mat(str(source))
    if "X_cal" not in loaded:
        cal, val, test = split_dataset(loaded["X"], loaded["y"], test_ratio=0.2, val_ratio=0.2, random_state=42)
        loaded = {"X_cal": cal[0], "y_cal": cal[1], "X_val": val[0], "y_val": val[1], "X_test": test[0], "y_test": test[1], **({"wv": loaded["wv"]} if "wv" in loaded else {})}
    task = {"_workflow_snapshot": {"data_path": virtual}}
    if recorded_hash:
        task["source_file"] = {"virtual_path": virtual, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "size_bytes": source.stat().st_size}
    search = SearchSessionStore(ProcessStore(paths.thread_dir("source-thread", user_id="alice") / "nir-process"))
    sid = search.prepare(project, loaded, cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development", {}), task_snapshot=task)["session_id"]
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        session.update(stage="finalized", freeze={"experiment_id": "exp-test", "model_hash": "f" * 64})
        search._save(db, session)
    metrics = {
        "training_data_hash": session["search_hash"],
        "autonomous_decision_audit": {"session_id": sid, "freeze": session["freeze"], "task": {k: v for k, v in task.items() if k != "_workflow_snapshot"}},
        "scientific_validation": session["scientific_validation"],
    }
    return source, virtual, session, metrics


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded_hash", [False, True])
async def test_direct_upload_model_is_automatically_linked_at_promotion(model_library, recorded_hash):
    service, paths = model_library
    source, virtual, session, metrics = source_fixture(paths, recorded_hash=recorded_hash)
    library = DatasetService(service.session_factory, paths, service.policy)
    dataset = await library.save_from_thread_upload("alice", "source-thread", virtual, source.name, save_confirmed=True)
    version, workflow = _registered_package(paths, training_hash=session["search_hash"], extra_metrics=metrics)
    saved = await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert saved["source_dataset_id"] == dataset["id"]
    assert workflow["dataset_id"] is None  # Historical scientific state is not rewritten.
    detail = await service.get_report("alice", "tablet_assay", version)
    assert detail["dataset"]["id"] == dataset["id"] and detail["report"]["feature_count"] == 8


@pytest.mark.asyncio
async def test_existing_model_recovers_display_link_without_database_or_package_writes(model_library):
    service, paths = model_library
    source, virtual, session, metrics = source_fixture(paths)
    version, workflow = _registered_package(paths, training_hash=session["search_hash"], extra_metrics=metrics)
    saved = await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert saved["source_dataset_id"] is None
    folder = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    dataset = await DatasetService(service.session_factory, paths, service.policy).save_from_thread_upload("alice", "source-thread", virtual, source.name, save_confirmed=True)
    detail = await service.get_report("alice", "tablet_assay", version)
    assert detail["dataset"]["id"] == dataset["id"]
    assert detail["model"]["source_dataset_id"] == detail["dataset"]["id"]
    assert detail["model"]["source_profile_id"] == detail["dataset"]["profile_id"]
    async with service.session_factory() as db:
        row = await db.scalar(select(NIRModelVersionRow))
        assert row.source_dataset_id is None
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


@pytest.mark.asyncio
async def test_changed_same_name_upload_cannot_be_linked_to_a_historical_model(model_library):
    service, paths = model_library
    source, virtual, session, metrics = source_fixture(paths)
    # A changed target changes the full frozen data binding, even with the same name and dimensions.
    data = _load_search_mat(str(source))
    savemat(source, {"X": data["X"], "y": data["y"] + 100, "wv": data["wv"]})
    await DatasetService(service.session_factory, paths, service.policy).save_from_thread_upload("alice", "source-thread", virtual, source.name, save_confirmed=True)
    async with service.session_factory() as db:
        assert await service._recover_dataset(db, "alice", "source-thread", metrics) is None


@pytest.mark.asyncio
async def test_corrupt_dataset_is_not_linked_or_quarantined_by_read(model_library):
    service, paths = model_library
    source, virtual, session, metrics = source_fixture(paths, recorded_hash=True)
    dataset = await DatasetService(service.session_factory, paths, service.policy).save_from_thread_upload("alice", "source-thread", virtual, source.name, save_confirmed=True)
    (paths.user_nir_dataset_dir("alice", dataset["id"]) / "manifest.json").write_text("{}")
    async with service.session_factory() as db:
        assert await service._recover_dataset(db, "alice", "source-thread", metrics) is None
        row = await db.get(NIRDatasetRow, dataset["id"])
        assert row.status == "ready"


@pytest.mark.asyncio
async def test_workspace_training_link_preserves_exact_saved_profile(model_library, monkeypatch):
    from deerflow.community.nir.datasets.workspace import fixed_regression_partitions, load_workspace_export

    service, paths = model_library
    source, virtual, _session, _metrics = source_fixture(paths)
    library = DatasetService(service.session_factory, paths, service.policy)
    dataset = await library.save_from_thread_upload("alice", "source-thread", virtual, source.name, save_confirmed=True)
    workspace = await library.save_workspace("alice", dataset["id"], source_sha256=dataset["sha256"], profile_id=None, usages=[1] * 36 + [2] * 12 + [3] * 12, band_range=[1000, 1007])
    attached = await library.attach_to_thread("alice", dataset["id"], "source-thread", profile_id=workspace["profile_id"], idempotency_key="training", apply_workspace=True)
    source = paths.sandbox_uploads_dir("source-thread", user_id="alice") / attached["virtual_path"].rsplit("/", 1)[-1]
    data = fixed_regression_partitions(load_workspace_export(str(source)))
    source, virtual, session, metrics = source_fixture(paths, source=source, data=data, recorded_hash=True, project="workspace")
    async with service.session_factory() as db:
        linked = await service._recover_dataset(db, "alice", "source-thread", metrics)
    assert linked["id"] == dataset["id"] and linked["profile_id"] == workspace["profile_id"]
    # Simulate the already-published model whose stored lineage was missing.
    resolver = service._recover_dataset
    monkeypatch.setattr(service, "_recover_dataset", AsyncMock(return_value=None))
    version, workflow = _registered_package(paths, training_hash=session["search_hash"], extra_metrics=metrics)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    monkeypatch.setattr(service, "_recover_dataset", resolver)
    detail = await service.get_report("alice", "tablet_assay", version)
    assert detail["dataset"]["profile_id"] == workspace["profile_id"]
    assert detail["model"]["source_dataset_id"] == detail["dataset"]["id"]
    assert detail["model"]["source_profile_id"] == detail["dataset"]["profile_id"]
    async with service.session_factory() as db:
        row = await db.scalar(select(NIRModelVersionRow))
        assert row.source_dataset_id is None and row.source_profile_id is None


def test_saved_dimension_metadata_fills_missing_counts_without_changing_recorded_values():
    report = {"feature_count": None, "model_feature_count": None, "charts": [{"scope": "calibration_before", "axis_total": 100}, {"scope": "calibration_after", "axis_total": 80}]}
    result = complete_dimensions(report, {})
    assert result["feature_count"] == 100 and result["model_feature_count"] == 80
    assert report["feature_count"] is None
    assert complete_dimensions({**report, "feature_count": 99}, {})["feature_count"] == 99


def test_recorded_file_identity_must_agree_with_saved_model_metrics(tmp_path):
    from deerflow.config.paths import Paths

    paths = Paths(tmp_path / "home")
    _source, _virtual, _session, metrics = source_fixture(paths, recorded_hash=True)
    assert recorded_source(metrics, paths, "alice", "source-thread") is not None
    metrics["autonomous_decision_audit"]["task"]["source_file"]["size_bytes"] += 1
    assert recorded_source(metrics, paths, "alice", "source-thread") is None
