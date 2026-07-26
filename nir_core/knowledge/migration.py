"""Auditable, in-place metadata migrations for legacy knowledge documents."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from nir_core.knowledge.governance import (
    DocumentRecord,
    KnowledgeCatalog,
    ReviewStatus,
    publication_readiness,
    require_publication_ready,
)
from nir_core.knowledge.vectorstore import _sanitise_metadata

MIGRATABLE_METADATA_FIELDS = frozenset(
    {
        "title",
        "authors",
        "year",
        "doi",
        "source_type",
        "language",
        "domains",
        "quality_tier",
    }
)


class MetadataStore(Protocol):
    """Minimum vector-store contract required by metadata migrations."""

    def update_document_metadata(self, doc_id: str, updates: dict[str, Any]) -> bool:
        """Update metadata on all chunks belonging to ``doc_id``."""

    def get_document_metadatas(self, doc_id: str) -> list[dict[str, Any]]:
        """Return current chunk metadata for ``doc_id``."""


class MetadataMigrationEntry(BaseModel):
    """One hash-pinned document metadata correction."""

    doc_id: str = Field(min_length=1)
    expected_content_sha256: str
    updates: dict[str, Any]
    target_review_status: ReviewStatus = "published"
    evidence: list[str] = Field(default_factory=list)

    @field_validator("expected_content_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        normalized = value.casefold()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(
                "expected_content_sha256 must be a 64-character hexadecimal digest"
            )
        return normalized

    @field_validator("updates")
    @classmethod
    def _validate_updates(cls, value: dict[str, Any]) -> dict[str, Any]:
        unsupported = sorted(set(value) - MIGRATABLE_METADATA_FIELDS)
        if unsupported:
            raise ValueError(f"Unsupported metadata fields: {', '.join(unsupported)}")
        if not value:
            raise ValueError("updates must not be empty")
        return value


class MetadataMigrationManifest(BaseModel):
    """Versioned collection of metadata corrections."""

    migration_id: str = Field(min_length=1)
    description: str = ""
    entries: list[MetadataMigrationEntry] = Field(min_length=1)

    @field_validator("entries")
    @classmethod
    def _reject_duplicate_documents(
        cls, value: list[MetadataMigrationEntry]
    ) -> list[MetadataMigrationEntry]:
        doc_ids = [entry.doc_id for entry in value]
        duplicates = sorted(
            doc_id for doc_id in set(doc_ids) if doc_ids.count(doc_id) > 1
        )
        if duplicates:
            raise ValueError(
                f"migration contains duplicate doc_id values: {', '.join(duplicates)}"
            )
        return value


@dataclass(frozen=True)
class MigrationResult:
    """Outcome for one migrated document."""

    doc_id: str
    changed_fields: tuple[str, ...]
    review_status: ReviewStatus
    readiness: dict[str, Any]
    chunk_count: int


class ChromaMetadataStore:
    """Metadata-only Chroma adapter that does not load an embedding model."""

    def __init__(
        self,
        *,
        db_path: str | Path,
        collection_name: str,
    ) -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection = self._client.get_collection(name=collection_name)

    def get_document_metadatas(self, doc_id: str) -> list[dict[str, Any]]:
        data = self._collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        return [dict(metadata or {}) for metadata in data.get("metadatas", [])]

    def update_document_metadata(
        self,
        doc_id: str,
        updates: dict[str, Any],
    ) -> bool:
        data = self._collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        ids = list(data.get("ids", []))
        if not ids:
            return False
        metadatas = data.get("metadatas", [])
        updated = [
            _sanitise_metadata({**(metadata or {}), **updates})
            for metadata in metadatas
        ]
        self._collection.update(ids=ids, metadatas=updated)
        return True


def load_migration_manifest(path: str | Path) -> MetadataMigrationManifest:
    """Load and validate a JSON migration manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MetadataMigrationManifest.model_validate(payload)


def audit_catalog(catalog: KnowledgeCatalog) -> list[dict[str, Any]]:
    """Return publication-readiness diagnostics for every catalog record."""

    return [
        {
            "doc_id": record.doc_id,
            "title": record.title,
            "review_status": record.review_status,
            "source_type": record.source_type,
            "readiness": publication_readiness(record),
        }
        for record in catalog.list()
    ]


def _metadata_for_vectors(record: DocumentRecord) -> dict[str, Any]:
    return {
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "doi": record.doi,
        "source_type": record.source_type,
        "language": record.language,
        "domains": record.domains,
        "quality_tier": record.quality_tier,
    }


def preflight_migration(
    *,
    catalog: KnowledgeCatalog,
    manifest: MetadataMigrationManifest,
) -> list[tuple[MetadataMigrationEntry, DocumentRecord, DocumentRecord]]:
    """Validate every migration entry before any persistent state changes."""

    prepared: list[tuple[MetadataMigrationEntry, DocumentRecord, DocumentRecord]] = []
    for entry in manifest.entries:
        existing = catalog.get(entry.doc_id)
        if existing is None:
            raise ValueError(f"Document not found: {entry.doc_id}")
        if existing.content_sha256 != entry.expected_content_sha256:
            raise ValueError(
                "Content hash mismatch for "
                f"{entry.doc_id}: expected {entry.expected_content_sha256}, "
                f"found {existing.content_sha256}"
            )
        candidate = DocumentRecord.model_validate(
            {
                **existing.model_dump(),
                **entry.updates,
                "review_status": entry.target_review_status,
            }
        )
        if entry.target_review_status == "published":
            require_publication_ready(candidate)
        prepared.append((entry, existing, candidate))
    return prepared


def apply_metadata_migration(
    *,
    catalog: KnowledgeCatalog,
    metadata_store: MetadataStore,
    manifest: MetadataMigrationManifest,
    dry_run: bool = False,
) -> list[MigrationResult]:
    """Apply a hash-pinned migration without changing IDs or embeddings."""

    prepared = preflight_migration(catalog=catalog, manifest=manifest)
    results: list[MigrationResult] = []
    for entry, existing, candidate in prepared:
        chunk_metadatas = metadata_store.get_document_metadatas(entry.doc_id)
        if not chunk_metadatas:
            raise ValueError(f"Vector chunks not found: {entry.doc_id}")

        changed_fields = tuple(
            sorted(
                field
                for field in entry.updates
                if getattr(existing, field) != getattr(candidate, field)
            )
        )
        readiness = publication_readiness(candidate)
        if dry_run:
            results.append(
                MigrationResult(
                    doc_id=entry.doc_id,
                    changed_fields=changed_fields,
                    review_status=entry.target_review_status,
                    readiness=readiness,
                    chunk_count=len(chunk_metadatas),
                )
            )
            continue

        if existing.review_status == "published":
            if not metadata_store.update_document_metadata(
                entry.doc_id, {"review_status": "needs_review"}
            ):
                raise RuntimeError(
                    f"Failed to withdraw vector chunks for {entry.doc_id}"
                )
            catalog.set_review_status(entry.doc_id, "needs_review")

        updated = catalog.update_metadata(entry.doc_id, entry.updates)
        if updated is None:
            raise RuntimeError(f"Document disappeared during migration: {entry.doc_id}")
        if not metadata_store.update_document_metadata(
            entry.doc_id, _metadata_for_vectors(updated)
        ):
            raise RuntimeError(
                f"Failed to update vector metadata for {entry.doc_id}; "
                "document remains withdrawn"
            )

        if entry.target_review_status == "published":
            published = catalog.set_review_status(entry.doc_id, "published")
            if published is None:
                raise RuntimeError(f"Failed to publish catalog record: {entry.doc_id}")
            if not metadata_store.update_document_metadata(
                entry.doc_id, {"review_status": "published"}
            ):
                catalog.set_review_status(entry.doc_id, "needs_review")
                raise RuntimeError(
                    f"Failed to publish vector chunks for {entry.doc_id}; "
                    "document remains withdrawn"
                )
            updated = published
        elif entry.target_review_status != updated.review_status:
            catalog.set_review_status(entry.doc_id, entry.target_review_status)
            if not metadata_store.update_document_metadata(
                entry.doc_id,
                {"review_status": entry.target_review_status},
            ):
                raise RuntimeError(f"Failed to set vector status for {entry.doc_id}")
            updated = catalog.get(entry.doc_id) or updated

        results.append(
            MigrationResult(
                doc_id=entry.doc_id,
                changed_fields=changed_fields,
                review_status=updated.review_status,
                readiness=publication_readiness(updated),
                chunk_count=len(chunk_metadatas),
            )
        )
    return results


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backup_knowledge_store(
    *,
    catalog_path: str | Path,
    chroma_path: str | Path,
    destination: str | Path,
) -> Path:
    """Create a verified catalog/Chroma backup at a new destination."""

    source_catalog = Path(catalog_path).resolve()
    source_chroma = Path(chroma_path).resolve()
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"Backup destination already exists: {target}")
    if not source_catalog.is_file():
        raise FileNotFoundError(f"Catalog not found: {source_catalog}")
    if not source_chroma.is_dir():
        raise FileNotFoundError(f"Chroma directory not found: {source_chroma}")

    target.mkdir(parents=True)
    catalog_copy = target / source_catalog.name
    with (
        sqlite3.connect(source_catalog) as source_connection,
        sqlite3.connect(catalog_copy) as target_connection,
    ):
        source_connection.backup(target_connection)
    chroma_copy = target / source_chroma.name
    shutil.copytree(source_chroma, chroma_copy)

    copied_files = [catalog_copy, *sorted(chroma_copy.rglob("*"))]
    checksums = {
        path.relative_to(target).as_posix(): _file_sha256(path)
        for path in copied_files
        if path.is_file()
    }
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_catalog": str(source_catalog),
        "source_chroma": str(source_chroma),
        "files": checksums,
    }
    (target / "backup-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def migration_results_json(
    results: Sequence[MigrationResult],
) -> list[dict[str, Any]]:
    """Return JSON-safe migration results for CLI/reporting."""

    return [
        {
            "doc_id": result.doc_id,
            "changed_fields": list(result.changed_fields),
            "review_status": result.review_status,
            "readiness": result.readiness,
            "chunk_count": result.chunk_count,
        }
        for result in results
    ]
