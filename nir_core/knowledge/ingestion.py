"""Governed, idempotent document ingestion shared by CLI and HTTP service."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from nir_core.knowledge.chunker import CHUNKER_VERSION, chunk_document
from nir_core.knowledge.entity_extractor import extract_entities
from nir_core.knowledge.governance import (
    CatalogAction,
    DocumentRecord,
    KnowledgeCatalog,
    QualityTier,
    ReviewStatus,
    content_sha256,
    infer_language,
    stable_document_id,
)
from nir_core.knowledge.parser import parse_document


class DocumentReplacer(Protocol):
    def replace_document(self, doc_id: str, chunks: list[Any]) -> int: ...


@dataclass(frozen=True)
class IngestionResult:
    action: CatalogAction
    record: DocumentRecord
    chunks_added: int


def _infer_source_type(filename: str) -> str:
    extension = Path(filename).suffix.casefold()
    return {
        ".pdf": "document",
        ".docx": "document",
        ".html": "web_document",
        ".htm": "web_document",
        ".csv": "dataset_documentation",
        ".md": "internal_document",
        ".markdown": "internal_document",
        ".txt": "internal_document",
    }.get(extension, "document")


def ingest_document_bytes(
    *,
    filename: str,
    content: bytes,
    retriever: DocumentReplacer,
    catalog: KnowledgeCatalog,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    doi: str | None = None,
    source_type: str | None = None,
    language: str | None = None,
    domains: list[str] | None = None,
    quality_tier: QualityTier = "C",
    review_status: ReviewStatus = "draft",
    chunk_strategy: str = "section",
    max_tokens: int = 512,
    overlap_percent: int = 10,
    index_version: str = "nir-papers-bge-m3-v1",
) -> IngestionResult:
    """Parse and replace a document only when its source bytes changed."""
    if not filename:
        raise ValueError("filename is required")
    if not content:
        raise ValueError("document content is empty")

    resolved_title = (title or Path(filename).stem).strip()
    resolved_authors = [author.strip() for author in (authors or []) if author.strip()]
    digest = content_sha256(content)
    doc_id = stable_document_id(
        content=content,
        doi=doi,
        title=resolved_title,
        authors=resolved_authors,
        year=year,
    )

    duplicate = catalog.get_by_content_hash(digest)
    if duplicate is not None:
        action: CatalogAction = (
            "unchanged" if duplicate.doc_id == doc_id else "duplicate"
        )
        return IngestionResult(action=action, record=duplicate, chunks_added=0)

    existing = catalog.get(doc_id)
    next_version = existing.version + 1 if existing else 1
    suffix = Path(filename).suffix or ".txt"
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            dir=catalog.path.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary_path = temporary.name
        markdown = parse_document(temporary_path)
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass

    if not markdown.strip():
        raise ValueError("document parser returned empty text")

    entities = extract_entities(markdown)
    resolved_domains = list(
        dict.fromkeys([*(domains or []), *entities.get("datasets", [])])
    )
    record = DocumentRecord(
        doc_id=doc_id,
        title=resolved_title,
        authors=resolved_authors,
        year=year,
        doi=doi,
        source_type=source_type or _infer_source_type(filename),
        source=filename,
        language=language or infer_language(markdown),
        domains=resolved_domains,
        quality_tier=quality_tier,
        review_status=review_status,
        content_sha256=digest,
        version=next_version,
    )

    chunks = chunk_document(
        markdown,
        source=filename,
        strategy=chunk_strategy,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        overlap_percent=overlap_percent,
        doc_id=doc_id,
        chunker_version=CHUNKER_VERSION,
    )
    if not chunks:
        raise ValueError("document produced no searchable chunks")

    document_metadata = {
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "doi": record.doi,
        "source_type": record.source_type,
        "language": record.language,
        "domains": record.domains,
        "quality_tier": record.quality_tier,
        "review_status": record.review_status,
        "content_sha256": record.content_sha256,
        "document_version": record.version,
        "index_version": index_version,
        "content_trust": "untrusted_evidence",
        **entities,
    }
    for chunk in chunks:
        chunk.metadata.update(document_metadata)

    chunks_added = retriever.replace_document(doc_id, chunks)
    catalog_result = catalog.register(record)
    return IngestionResult(
        action=catalog_result.action,
        record=catalog_result.record,
        chunks_added=chunks_added,
    )
