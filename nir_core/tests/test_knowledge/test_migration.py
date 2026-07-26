"""Tests for auditable in-place knowledge metadata migrations."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from nir_core.knowledge.governance import DocumentRecord, KnowledgeCatalog
from nir_core.knowledge.migration import (
    MetadataMigrationManifest,
    apply_metadata_migration,
    audit_catalog,
    backup_knowledge_store,
    load_migration_manifest,
    preflight_migration,
)


class _FakeMetadataStore:
    def __init__(self, doc_id: str) -> None:
        self.metadatas = {
            doc_id: [
                {
                    "doc_id": doc_id,
                    "review_status": "published",
                    "source_type": "document",
                },
                {
                    "doc_id": doc_id,
                    "review_status": "published",
                    "source_type": "document",
                },
            ]
        }
        self.update_calls: list[tuple[str, dict[str, Any]]] = []

    def get_document_metadatas(self, doc_id: str) -> list[dict[str, Any]]:
        return [dict(metadata) for metadata in self.metadatas.get(doc_id, [])]

    def update_document_metadata(self, doc_id: str, updates: dict[str, Any]) -> bool:
        if doc_id not in self.metadatas:
            return False
        self.update_calls.append((doc_id, dict(updates)))
        for metadata in self.metadatas[doc_id]:
            for key, value in updates.items():
                if value is None:
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
        return True


def _legacy_record() -> DocumentRecord:
    return DocumentRecord(
        doc_id="sha256:legacy",
        title="Paper_Alice",
        authors=["Alice"],
        year=2020,
        source_type="document",
        source="paper.pdf",
        language="en",
        quality_tier="C",
        review_status="published",
        content_sha256="a" * 64,
    )


def _manifest(*, expected_hash: str = "a" * 64) -> MetadataMigrationManifest:
    return MetadataMigrationManifest.model_validate(
        {
            "migration_id": "test-migration",
            "entries": [
                {
                    "doc_id": "sha256:legacy",
                    "expected_content_sha256": expected_hash,
                    "updates": {
                        "title": "Paper",
                        "authors": ["Alice", "Bob"],
                        "year": 2024,
                        "doi": "10.1000/migrated",
                        "source_type": "journal_article",
                    },
                    "target_review_status": "published",
                }
            ],
        }
    )


def test_migration_updates_catalog_and_all_chunks_without_changing_identity(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    original = catalog.register(_legacy_record()).record
    store = _FakeMetadataStore(original.doc_id)

    results = apply_metadata_migration(
        catalog=catalog,
        metadata_store=store,
        manifest=_manifest(),
    )

    migrated = catalog.get(original.doc_id)
    assert migrated is not None
    assert migrated.doc_id == original.doc_id
    assert migrated.content_sha256 == original.content_sha256
    assert migrated.version == original.version
    assert migrated.title == "Paper"
    assert migrated.authors == ["Alice", "Bob"]
    assert migrated.year == 2024
    assert migrated.doi == "10.1000/migrated"
    assert migrated.source_type == "journal_article"
    assert migrated.review_status == "published"
    assert results[0].readiness["ready"] is True
    assert results[0].chunk_count == 2
    assert all(
        metadata["review_status"] == "published"
        and metadata["source_type"] == "journal_article"
        and metadata["year"] == 2024
        for metadata in store.metadatas[original.doc_id]
    )
    assert store.update_calls[0][1] == {"review_status": "needs_review"}
    assert store.update_calls[-1][1] == {"review_status": "published"}


def test_migration_dry_run_and_hash_mismatch_do_not_mutate(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    original = catalog.register(_legacy_record()).record
    store = _FakeMetadataStore(original.doc_id)

    preview = apply_metadata_migration(
        catalog=catalog,
        metadata_store=store,
        manifest=_manifest(),
        dry_run=True,
    )

    assert preview[0].changed_fields == (
        "authors",
        "doi",
        "source_type",
        "title",
        "year",
    )
    assert catalog.get(original.doc_id) == original
    assert store.update_calls == []

    with pytest.raises(ValueError, match="Content hash mismatch"):
        preflight_migration(
            catalog=catalog,
            manifest=_manifest(expected_hash="b" * 64),
        )
    assert catalog.get(original.doc_id) == original


def test_migration_rejects_incomplete_publication_before_vector_changes(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    original = catalog.register(_legacy_record()).record
    store = _FakeMetadataStore(original.doc_id)
    manifest = MetadataMigrationManifest.model_validate(
        {
            "migration_id": "invalid",
            "entries": [
                {
                    "doc_id": original.doc_id,
                    "expected_content_sha256": original.content_sha256,
                    "updates": {"authors": []},
                    "target_review_status": "published",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="authors"):
        apply_metadata_migration(
            catalog=catalog,
            metadata_store=store,
            manifest=manifest,
        )
    assert store.update_calls == []


def test_audit_and_manifest_loader_report_legacy_readiness(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.register(_legacy_record())
    manifest_path = tmp_path / "migration.json"
    manifest_path.write_text(
        json.dumps(_manifest().model_dump()),
        encoding="utf-8",
    )

    loaded = load_migration_manifest(manifest_path)
    audit = audit_catalog(catalog)

    assert loaded.migration_id == "test-migration"
    assert audit[0]["readiness"]["ready"] is True
    assert audit[0]["readiness"]["doi_status"] == "missing"


def test_backup_copies_catalog_and_chroma_with_checksums(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.sqlite3"
    catalog = KnowledgeCatalog(catalog_path)
    catalog.register(_legacy_record())
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    (chroma_path / "chroma.sqlite3").write_bytes(b"vector metadata")
    destination = tmp_path / "backup"

    backup = backup_knowledge_store(
        catalog_path=catalog_path,
        chroma_path=chroma_path,
        destination=destination,
    )

    with sqlite3.connect(backup / "catalog.sqlite3") as connection:
        assert connection.execute("select count(*) from kb_documents").fetchone() == (
            1,
        )
    manifest = json.loads((backup / "backup-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "catalog.sqlite3",
        "chroma/chroma.sqlite3",
    }
    assert (backup / "chroma" / "chroma.sqlite3").read_bytes() == b"vector metadata"
