"""Stable identity and lightweight SQLite governance for knowledge documents."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ReviewStatus = Literal["draft", "needs_review", "published", "retired"]
QualityTier = Literal["A", "B", "C", "D", "E"]
CatalogAction = Literal["created", "unchanged", "updated", "duplicate"]

_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_DOI_VALUE_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_DOI_SEARCH_RE = re.compile(r"\b10\.\d{4,9}/[^\s<>\"']+", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_DOI_EXPECTED_SOURCE_TYPES = {
    "article",
    "conference_paper",
    "document",
    "journal",
    "journal_article",
}


def content_sha256(content: bytes) -> str:
    """Return the lowercase SHA-256 hex digest for source file bytes."""
    return hashlib.sha256(content).hexdigest()


def normalize_doi(doi: str | None) -> str | None:
    """Normalize common DOI URL/prefix forms into a stable lowercase value."""
    if not doi:
        return None
    normalized = _DOI_PREFIX_RE.sub("", unicodedata.normalize("NFKC", doi).strip())
    normalized = normalized.strip().rstrip(".,;)]}").casefold()
    return normalized or None


def is_valid_doi(doi: str | None) -> bool:
    """Return whether a DOI has a valid registrant prefix and non-space suffix."""
    normalized = normalize_doi(doi)
    return bool(normalized and _DOI_VALUE_RE.fullmatch(normalized))


def require_valid_doi(doi: str | None) -> str | None:
    """Normalize an optional DOI and reject malformed values."""
    normalized = normalize_doi(doi)
    if normalized is not None and not is_valid_doi(normalized):
        raise ValueError("doi must match the canonical form 10.<registrant>/<suffix>")
    return normalized


def extract_doi(text: str) -> str | None:
    """Extract the first syntactically valid DOI from parsed document text."""
    for match in _DOI_SEARCH_RE.finditer(text or ""):
        candidate = normalize_doi(match.group(0))
        if is_valid_doi(candidate):
            return candidate
    return None


def _normalize_identity_text(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def stable_document_id(
    *,
    content: bytes,
    doi: str | None = None,
    title: str | None = None,
    authors: list[str] | tuple[str, ...] | None = None,
    year: int | None = None,
) -> str:
    """Build a stable ID using DOI, bibliography, then source content hash."""
    normalized_doi = require_valid_doi(doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"

    normalized_title = _normalize_identity_text(title or "")
    normalized_authors = [
        _normalize_identity_text(author) for author in (authors or []) if author.strip()
    ]
    if normalized_title and normalized_authors and year is not None:
        identity = json.dumps(
            [normalized_title, normalized_authors, int(year)],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"bib:{hashlib.sha256(identity).hexdigest()}"

    return f"sha256:{content_sha256(content)}"


def infer_language(text: str) -> str:
    """Return a small, deterministic language hint suitable for metadata."""
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", text))
    latin_count = len(re.findall(r"[A-Za-z]", text))
    if cjk_count and latin_count:
        return "zh-en"
    if cjk_count:
        return "zh"
    if latin_count:
        return "en"
    return "und"


class DocumentRecord(BaseModel):
    """Authoritative metadata record stored outside the vector collection."""

    doc_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1000, le=2100)
    doi: str | None = None
    source_type: str = "document"
    source: str = ""
    language: str = "und"
    domains: list[str] = Field(default_factory=list)
    quality_tier: QualityTier = "C"
    review_status: ReviewStatus = "draft"
    content_sha256: str
    version: int = Field(default=1, ge=1)
    created_at: str = ""
    updated_at: str = ""

    @field_validator("content_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        normalized = value.casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
        return normalized

    @field_validator("doi")
    @classmethod
    def _normalize_record_doi(cls, value: str | None) -> str | None:
        return require_valid_doi(value)


def publication_readiness(record: DocumentRecord) -> dict[str, Any]:
    """Return deterministic metadata requirements for publication."""

    missing: list[str] = []
    warnings: list[dict[str, str]] = []
    if not record.title.strip():
        missing.append("title")
    if not record.authors:
        missing.append("authors")
    if record.year is None:
        missing.append("year")
    if not record.source.strip():
        missing.append("source")
    expects_doi = record.source_type.casefold() in _DOI_EXPECTED_SOURCE_TYPES
    if expects_doi and not record.doi:
        warnings.append(
            {
                "code": "doi_missing",
                "message": (
                    "No DOI is recorded. Publication is allowed, but verify "
                    "that this source genuinely has no DOI."
                ),
            }
        )
    ready = not missing
    if not ready:
        message = "publication requires: " + ", ".join(missing)
    elif warnings:
        message = "ready for publication with warning: DOI is missing"
    else:
        message = "ready for publication"
    return {
        "ready": ready,
        "missing_fields": missing,
        "warnings": warnings,
        "doi_status": (
            "valid" if record.doi else ("missing" if expects_doi else "not_applicable")
        ),
        "message": message,
    }


def require_publication_ready(record: DocumentRecord) -> None:
    readiness = publication_readiness(record)
    if not readiness["ready"]:
        raise ValueError(str(readiness["message"]))


@dataclass(frozen=True)
class CatalogWriteResult:
    action: CatalogAction
    record: DocumentRecord


class KnowledgeCatalog:
    """Small SQLite document catalog for a single-node knowledge service."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS kb_documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors_json TEXT NOT NULL,
                    year INTEGER,
                    doi TEXT,
                    source_type TEXT NOT NULL,
                    source TEXT NOT NULL,
                    language TEXT NOT NULL,
                    domains_json TEXT NOT NULL,
                    quality_tier TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL UNIQUE,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS ix_kb_documents_review_status ON kb_documents(review_status)"
            )

    @staticmethod
    def _from_row(row: sqlite3.Row | None) -> DocumentRecord | None:
        if row is None:
            return None
        return DocumentRecord(
            doc_id=row["doc_id"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            doi=row["doi"],
            source_type=row["source_type"],
            source=row["source"],
            language=row["language"],
            domains=json.loads(row["domains_json"]),
            quality_tier=row["quality_tier"],
            review_status=row["review_status"],
            content_sha256=row["content_sha256"],
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM kb_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return self._from_row(row)

    def get_by_content_hash(self, digest: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM kb_documents WHERE content_sha256 = ?",
                (digest.casefold(),),
            ).fetchone()
        return self._from_row(row)

    def register(self, record: DocumentRecord) -> CatalogWriteResult:
        """Create, deduplicate, or version a document metadata record."""
        if record.review_status == "published":
            require_publication_ready(record)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            duplicate_row = connection.execute(
                "SELECT * FROM kb_documents WHERE content_sha256 = ?",
                (record.content_sha256,),
            ).fetchone()
            duplicate = self._from_row(duplicate_row)
            if duplicate is not None:
                action: CatalogAction = (
                    "unchanged" if duplicate.doc_id == record.doc_id else "duplicate"
                )
                return CatalogWriteResult(action=action, record=duplicate)

            existing_row = connection.execute(
                "SELECT * FROM kb_documents WHERE doc_id = ?", (record.doc_id,)
            ).fetchone()
            existing = self._from_row(existing_row)
            stored = record.model_copy(
                update={
                    "version": (existing.version + 1) if existing else 1,
                    "created_at": existing.created_at if existing else now,
                    "updated_at": now,
                }
            )
            connection.execute(
                """
                INSERT INTO kb_documents (
                    doc_id, title, authors_json, year, doi, source_type, source,
                    language, domains_json, quality_tier, review_status,
                    content_sha256, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    title=excluded.title,
                    authors_json=excluded.authors_json,
                    year=excluded.year,
                    doi=excluded.doi,
                    source_type=excluded.source_type,
                    source=excluded.source,
                    language=excluded.language,
                    domains_json=excluded.domains_json,
                    quality_tier=excluded.quality_tier,
                    review_status=excluded.review_status,
                    content_sha256=excluded.content_sha256,
                    version=excluded.version,
                    updated_at=excluded.updated_at
                """,
                (
                    stored.doc_id,
                    stored.title,
                    json.dumps(stored.authors, ensure_ascii=False),
                    stored.year,
                    stored.doi,
                    stored.source_type,
                    stored.source,
                    stored.language,
                    json.dumps(stored.domains, ensure_ascii=False),
                    stored.quality_tier,
                    stored.review_status,
                    stored.content_sha256,
                    stored.version,
                    stored.created_at,
                    stored.updated_at,
                ),
            )
        return CatalogWriteResult(
            action="updated" if existing else "created", record=stored
        )

    def list(
        self, *, review_status: ReviewStatus | None = None
    ) -> list[DocumentRecord]:
        query = "SELECT * FROM kb_documents"
        params: tuple[str, ...] = ()
        if review_status is not None:
            query += " WHERE review_status = ?"
            params = (review_status,)
        query += " ORDER BY doc_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [record for row in rows if (record := self._from_row(row)) is not None]

    def delete(self, doc_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM kb_documents WHERE doc_id = ?", (doc_id,)
            )
        return cursor.rowcount > 0

    def set_review_status(
        self, doc_id: str, review_status: ReviewStatus
    ) -> DocumentRecord | None:
        """Change governance state without creating a new content version."""
        current = self.get(doc_id)
        if current is None:
            return None
        if review_status == "published":
            require_publication_ready(current)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE kb_documents SET review_status = ?, updated_at = ? WHERE doc_id = ?",
                (review_status, now, doc_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get(doc_id)

    def update_metadata(
        self,
        doc_id: str,
        updates: Mapping[str, Any],
    ) -> DocumentRecord | None:
        """Update descriptive metadata without changing document identity/version."""
        editable_fields = {
            "title",
            "authors",
            "year",
            "doi",
            "language",
            "domains",
            "quality_tier",
        }
        unsupported = sorted(set(updates) - editable_fields)
        if unsupported:
            raise ValueError(f"Unsupported metadata fields: {', '.join(unsupported)}")

        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM kb_documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            existing = self._from_row(row)
            if existing is None:
                return None

            payload = existing.model_dump()
            payload.update(dict(updates))
            payload["updated_at"] = now
            updated = DocumentRecord.model_validate(payload)
            if updated.review_status == "published":
                require_publication_ready(updated)
            connection.execute(
                """
                UPDATE kb_documents SET
                    title = ?,
                    authors_json = ?,
                    year = ?,
                    doi = ?,
                    language = ?,
                    domains_json = ?,
                    quality_tier = ?,
                    updated_at = ?
                WHERE doc_id = ?
                """,
                (
                    updated.title,
                    json.dumps(updated.authors, ensure_ascii=False),
                    updated.year,
                    updated.doi,
                    updated.language,
                    json.dumps(updated.domains, ensure_ascii=False),
                    updated.quality_tier,
                    updated.updated_at,
                    doc_id,
                ),
            )
        return updated
