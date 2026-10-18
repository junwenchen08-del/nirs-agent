"""Trusted ingestion snapshots backed by immutable library Profiles.

Full membership lives in private sessions and owner-only library mappings. Tool
messages carry only a receipt. Saving/reading a snapshot never fits a model or
opens search.npz/holdout.npz.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import PurePosixPath
from uuid import uuid4

import numpy as np
from sqlalchemy import func, select

from deerflow.persistence.nir_library.model import (
    NIRDatasetProfileRow,
    NIRDatasetRow,
    NIRDatasetUseRow,
)
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.runtime.user_context import resolve_runtime_user_id

from .process import ProcessStore, _digest, _id

logger = logging.getLogger(__name__)


class MissingPartitionManifest(ValueError):
    """Older ingestion did not record membership; never reconstruct it on read."""


def read_partition_session(paths, owner, thread, session_id):
    index = paths.thread_dir(thread, user_id=owner) / "nir-process" / "index.sqlite3"
    if any(p.is_symlink() for p in (index, *index.parents)) or not index.is_file():
        raise ValueError("Partition session unavailable")
    with sqlite3.connect(index.as_uri() + "?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM search_sessions WHERE id=?", (_id(session_id),)).fetchone()
    if row is None or len(row[0]) > 8 * 1024**2:
        raise ValueError("Partition session unavailable")
    session = json.loads(row[0])
    manifest = session.get("partition_manifest")
    if not manifest:
        raise MissingPartitionManifest("Partition membership was not recorded")
    if session.get("partition_manifest_sha256") != _digest(manifest):
        raise ValueError("Partition manifest failed verification")
    return session


def partition_summary(profile):
    mapping = profile.mapping_json
    from .datasets.service import _validated_mapping

    if _validated_mapping(mapping)[1] != profile.mapping_sha256:
        raise ValueError("Partition mapping failed verification")
    split = mapping.get("partition")
    if profile.profile_status != "split" or not isinstance(split, dict):
        return None
    return {
        "profile_id": profile.id,
        "version": profile.profile_version,
        **{
            key: split[key]
            for key in (
                "method",
                "counts",
                "parent_profile_id",
                "source_thread_id",
                "session_id",
            )
        },
    }


async def save_partition(service, owner, thread, session_id):
    from nir_core.utils.scientific_validation import _array_digest

    from .datasets.service import (
        DatasetError,
        _hash_regular_file,
        _owner_admission_lock,
        _source_path,
        _validated_mapping,
        _workspace_attachment,
    )
    from .datasets.workspace import read_workbook, validate_settings
    from .models.source import verified_library_source

    service._enabled()

    async def owned(db):
        if await db.scalar(select(ThreadMetaRow.thread_id).where(ThreadMetaRow.thread_id == thread, ThreadMetaRow.user_id == owner)) is None:
            raise DatasetError("thread_not_owned", "建模对话不存在。")

    async with service.session_factory() as db:
        await owned(db)
    frozen = await asyncio.to_thread(read_partition_session, service.paths, owner, thread, session_id)
    manifest = frozen["partition_manifest"]
    from .datasets.workspace import MAX_FEATURES, MAX_ROWS

    if sum(len(values) for values in manifest["indices"].values()) > MAX_ROWS or manifest["feature_count"] > MAX_FEATURES:
        raise DatasetError("partition_too_large", "数据规模超出数据集工作台支持范围，固定划分仍保留在本次建模记录中。")
    source_file = manifest["source_file"]
    if PurePosixPath(source_file["virtual_path"]).suffix.lower() not in {".csv", ".txt", ".mat"}:
        raise DatasetError("partition_source_unsupported", "此格式暂不支持数据集库的划分版本，固定划分仍保留在本次建模记录中。")
    source = _source_path(service.paths, owner, thread, source_file["virtual_path"])
    if any(p.is_symlink() for p in (source, *source.parents)) or not source.is_file() or source.stat().st_size != source_file["size_bytes"] or await asyncio.to_thread(_hash_regular_file, source) != source_file["sha256"]:
        raise DatasetError("source_hash_mismatch", "建模输入已变化，未保存新的划分版本。")
    root = service.paths.user_nir_datasets_dir(owner)
    async with _owner_admission_lock(root):
        async with service.session_factory() as db:
            await owned(db)
            uses = list(
                (
                    await db.scalars(
                        select(NIRDatasetUseRow).where(
                            NIRDatasetUseRow.owner_user_id == owner,
                            NIRDatasetUseRow.thread_id == thread,
                            NIRDatasetUseRow.attachment_filename == source.name,
                            NIRDatasetUseRow.status == "attached",
                        )
                    )
                ).all()
            )
            identities = {(use.dataset_id, use.profile_id, use.source_sha256) for use in uses}
            if len(identities) > 1:
                raise DatasetError("lineage_mismatch", "输入对应多个数据版本，无法确认划分来源。")
            parent = None
            if uses:
                use = uses[0]
                dataset = await db.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.id == use.dataset_id,
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.sha256 == use.source_sha256,
                        NIRDatasetRow.status == "ready",
                    )
                )
                if use.profile_id:
                    parent = await db.scalar(
                        select(NIRDatasetProfileRow).where(
                            NIRDatasetProfileRow.id == use.profile_id,
                            NIRDatasetProfileRow.owner_user_id == owner,
                            NIRDatasetProfileRow.dataset_id == use.dataset_id,
                        )
                    )
                    if parent is None or _validated_mapping(parent.mapping_json)[1] != parent.mapping_sha256:
                        raise DatasetError("profile_unavailable", "输入的解析版本无法核实。")
            else:
                dataset = await db.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.sha256 == source_file["sha256"],
                        NIRDatasetRow.status == "ready",
                    )
                )
            if dataset is None:
                raise DatasetError(
                    "dataset_not_found",
                    "原始数据尚未保存到数据集库，当前划分仍保留在建模记录中。",
                )
            if not await asyncio.to_thread(verified_library_source, service.paths, owner, dataset):
                raise DatasetError("source_hash_mismatch", "原始数据校验失败，未保存划分版本。")
            raw = service.paths.user_nir_dataset_dir(owner, dataset.id) / PurePosixPath(dataset.storage_relpath).name

            def verify_input():
                with _workspace_attachment(raw, parent, root, dataset.sha256, dataset.size_bytes) as (_, digest, size):
                    return digest == source_file["sha256"] and size == source_file["size_bytes"]

            if not await asyncio.to_thread(verify_input):
                raise DatasetError("source_hash_mismatch", "解析版本与建模输入不一致。")
            if parent and parent.profile_status == "split":
                if manifest["method"] != "fixed" or parent.mapping_json["partition"]["data_sha256"] != frozen["source_digest"]:
                    raise DatasetError("lineage_mismatch", "固定划分与建模输入不一致。")
                profile = parent
            else:
                mapping = dict(parent.mapping_json) if parent else {"model_input": manifest["input_mapping"]}
                book = await asyncio.to_thread(read_workbook, raw, {"mapping": mapping})
                settings = mapping.get("workspace") or validate_settings(
                    book,
                    book.usages,
                    [float(book.coordinates.min()), float(book.coordinates.max())],
                )
                selected = (book.coordinates >= settings["range"][0]) & (book.coordinates <= settings["range"][1])
                indices = manifest["indices"]
                flat = [i for part in ("calibration", "validation", "test") for i in indices[part]]
                if len(book.fields) != 1 or len(flat) != len(set(flat)) or any(type(i) is not int or not 0 <= i < len(book.X) for i in flat):
                    raise DatasetError("lineage_mismatch", "样本划分与原始数据不一致。")
                arrays = {}
                usages = [4] * len(book.X)
                for name, short, code in (
                    ("calibration", "cal", 1),
                    ("validation", "val", 2),
                    ("test", "test", 3),
                ):
                    chosen = indices[name]
                    arrays["X_" + short] = np.asarray(book.X[chosen][:, selected], dtype=float)
                    arrays["y_" + short] = np.asarray([book.values[i][0] for i in chosen], dtype=float)
                    if book.groups:
                        arrays["groups_" + short] = np.asarray([book.groups[i] for i in chosen], dtype=str)
                    for i in chosen:
                        usages[i] = code
                if book.axis is not None:
                    arrays["wv"] = book.axis[selected]
                if _array_digest(*(arrays[key] for key in sorted(arrays))) != frozen["source_digest"]:
                    raise DatasetError(
                        "lineage_mismatch",
                        "划分数据无法与本次建模核对，未生成替代划分。",
                    )
                settings = {
                    **validate_settings(book, usages, settings["range"]),
                    "source_sha256": dataset.sha256,
                    "sample_order": flat + [i for i, code in enumerate(usages) if code == 4],
                }
                split = {
                    "schema_version": 1,
                    "source_sha256": dataset.sha256,
                    "input_sha256": source_file["sha256"],
                    "parent_profile_id": parent.id if parent else None,
                    "source_thread_id": thread,
                    "session_id": session_id,
                    "binding": frozen["binding"],
                    "data_sha256": frozen["source_digest"],
                    "indices": indices,
                    "method": manifest["method"],
                    "parameters": manifest["parameters"],
                    "counts": {name: len(values) for name, values in indices.items()},
                    "cv_manifest_sha256": _digest(frozen["cv_manifest"]),
                }
                mapping, digest = _validated_mapping({**mapping, "workspace": settings, "partition": split})
                existing = list(
                    (
                        await db.scalars(
                            select(NIRDatasetProfileRow).where(
                                NIRDatasetProfileRow.dataset_id == dataset.id,
                                NIRDatasetProfileRow.owner_user_id == owner,
                                NIRDatasetProfileRow.profile_status == "split",
                            )
                        )
                    ).all()
                )
                profile = next(
                    (p for p in existing if p.mapping_json.get("partition", {}).get("session_id") == session_id and p.mapping_json.get("partition", {}).get("source_thread_id") == thread),
                    None,
                )
                if profile and profile.mapping_sha256 != digest:
                    raise DatasetError("lineage_mismatch", "已保存的划分版本与当前记录不一致。")
                if profile is None:
                    latest = await db.scalar(
                        select(func.max(NIRDatasetProfileRow.profile_version)).where(
                            NIRDatasetProfileRow.dataset_id == dataset.id,
                            NIRDatasetProfileRow.owner_user_id == owner,
                        )
                    )
                    profile = NIRDatasetProfileRow(
                        id=f"dsp_{uuid4().hex}",
                        owner_user_id=owner,
                        dataset_id=dataset.id,
                        profile_version=int(latest or 0) + 1,
                        profile_status="split",
                        task_type="calibration",
                        schema_status="confirmed_mapping",
                        mapping_json=mapping,
                        mapping_sha256=digest,
                    )
                    db.add(profile)
            if not await asyncio.to_thread(verified_library_source, service.paths, owner, dataset) or await asyncio.to_thread(_hash_regular_file, source) != source_file["sha256"]:
                raise DatasetError("source_hash_mismatch", "保存期间数据文件发生变化，请核查输入。")
            await db.commit()
            receipt = {
                "dataset_id": dataset.id,
                "profile_id": profile.id,
                "mapping_sha256": profile.mapping_sha256,
                "data_sha256": frozen["source_digest"],
            }

    # Commit the immutable library version first; a retry recovers its exact ID
    # if the process stops between the two stores. Do not change the binding.
    def bind_receipt():
        process = ProcessStore(service.paths.thread_dir(thread, user_id=owner) / "nir-process")
        with process._connection(write=True) as db:
            row = db.execute("SELECT payload FROM search_sessions WHERE id=?", (session_id,)).fetchone()
            session = json.loads(row[0])
            if session["binding"] != frozen["binding"] or session.get("library_partition", receipt) != receipt:
                raise ValueError("Partition receipt conflicts with immutable ingestion")
            session["library_partition"] = receipt
            db.execute(
                "UPDATE search_sessions SET payload=? WHERE id=?",
                (json.dumps(session, ensure_ascii=False, allow_nan=False), session_id),
            )

    await asyncio.to_thread(bind_receipt)
    return receipt


async def save_prepared_partition(runtime, result):
    """Persist new recorded ingestion, leaving scientific state intact on failure."""
    from .dataset_tools import _service, _thread_id
    from .datasets.service import DatasetError

    update = result.update if hasattr(result, "update") and isinstance(result.update, dict) else {}
    if update.get("messages") and json.loads(update["messages"][-1].content).get("status") == "error":
        return result
    workflow = update.get("nir_workflow") or (runtime.state or {}).get("nir_workflow") or {}
    sid = workflow.get("search_session_id")
    if not sid:
        return result
    try:
        service = _service()
        owner, thread = resolve_runtime_user_id(runtime), _thread_id(runtime)
        frozen = await asyncio.to_thread(read_partition_session, service.paths, owner, thread, sid)
        if not frozen.get("partition_manifest"):
            return result
        receipt = await save_partition(service, owner, thread, sid)
        outcome = {
            "status": "saved",
            "dataset_id": receipt["dataset_id"],
            "profile_id": receipt["profile_id"],
            "message": "已保存本次固定划分，可在数据集库查看和复用。",
        }
        # Keep the original upload/Profile/attachment identity intact. The
        # modeling partition is a separate result bound by this receipt.
    except DatasetError as exc:
        if exc.code == "feature_disabled":
            return result
        outcome = {"status": "failed", "error_code": exc.code, "message": str(exc)}
    except MissingPartitionManifest:
        # Legacy sessions without a membership manifest cannot be backfilled by
        # guessing or by reopening Test. Only new ingestions get snapshots.
        return result
    except ValueError:
        outcome = {
            "status": "failed",
            "error_code": "partition_unverified",
            "message": "划分记录校验失败，未生成替代版本。",
        }
    except Exception:
        logger.exception("Unable to persist dataset partition")
        outcome = {
            "status": "failed",
            "error_code": "storage_error",
            "message": "划分已保留在本次建模记录中，保存到数据集库失败。",
        }
    update["nir_workflow"] = {**workflow, "dataset_partition": outcome}
    if update.get("messages"):
        message = update["messages"][-1]
        payload = json.loads(message.content)
        message.content = json.dumps({**payload, "dataset_partition": outcome}, ensure_ascii=False)
    return result
