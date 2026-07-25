"""Tests for stable document identity and the lightweight KB catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from nir_core.knowledge.governance import (
    DocumentRecord,
    KnowledgeCatalog,
    content_sha256,
    extract_doi,
    is_valid_doi,
    publication_readiness,
    stable_document_id,
)
from nir_core.knowledge.ingestion import ingest_document_bytes


class _FakeRetriever:
    def __init__(self) -> None:
        self.replacements: list[tuple[str, list]] = []

    def replace_document(self, doc_id: str, chunks: list) -> int:
        self.replacements.append((doc_id, chunks))
        return len(chunks)


def test_stable_document_id_prefers_normalized_doi() -> None:
    first = stable_document_id(
        content=b"first version",
        doi="https://doi.org/10.1000/ABC.123",
        title="Ignored title",
    )
    second = stable_document_id(
        content=b"second version",
        doi="doi:10.1000/abc.123",
        title="Renamed file",
    )

    assert first == "doi:10.1000/abc.123"
    assert second == first


def test_doi_validation_and_extraction_reject_malformed_values() -> None:
    assert is_valid_doi("https://doi.org/10.1000/ABC.123") is True
    assert is_valid_doi("not-a-doi") is False
    assert (
        extract_doi("Available at https://doi.org/10.1016/j.chemo.2025.01.004.")
        == "10.1016/j.chemo.2025.01.004"
    )

    with pytest.raises(ValueError, match="canonical form"):
        stable_document_id(content=b"paper", doi="not-a-doi")


def test_stable_document_id_uses_normalized_bibliographic_identity() -> None:
    first = stable_document_id(
        content=b"v1",
        title="  Soil   Organic Carbon  ",
        authors=["Alice Zhang", "Bob Li"],
        year=2024,
    )
    second = stable_document_id(
        content=b"v2",
        title="soil organic carbon",
        authors=[" alice zhang ", "BOB LI"],
        year=2024,
    )

    assert first.startswith("bib:")
    assert second == first


def test_stable_document_id_falls_back_to_content_hash() -> None:
    raw = "近红外知识".encode()

    assert stable_document_id(content=raw) == f"sha256:{content_sha256(raw)}"


def _record(*, digest: str, title: str = "Paper") -> DocumentRecord:
    return DocumentRecord(
        doc_id="doi:10.1000/test",
        title=title,
        authors=["Alice"],
        year=2024,
        doi="10.1000/test",
        source_type="journal",
        source="paper.pdf",
        language="en",
        domains=["soil"],
        quality_tier="B",
        review_status="published",
        content_sha256=digest,
    )


def test_catalog_register_is_idempotent_and_versions_changes(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")

    created = catalog.register(_record(digest="a" * 64))
    unchanged = catalog.register(_record(digest="a" * 64))
    updated = catalog.register(_record(digest="b" * 64, title="Paper revised"))

    assert created.action == "created"
    assert created.record.version == 1
    assert unchanged.action == "unchanged"
    assert unchanged.record.version == 1
    assert updated.action == "updated"
    assert updated.record.version == 2
    assert catalog.get("doi:10.1000/test") == updated.record


def test_catalog_filters_review_status(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.register(_record(digest="a" * 64))
    catalog.register(
        DocumentRecord(
            doc_id="sha256:draft",
            title="Draft",
            authors=["Reviewer"],
            year=2024,
            source="draft.pdf",
            review_status="draft",
            content_sha256="b" * 64,
        )
    )

    assert [record.doc_id for record in catalog.list(review_status="published")] == [
        "doi:10.1000/test"
    ]

    updated = catalog.set_review_status("sha256:draft", "published")
    assert updated is not None
    assert updated.review_status == "published"
    assert updated.version == 1


def test_catalog_rejects_publication_with_incomplete_citation_metadata(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.register(
        DocumentRecord(
            doc_id="sha256:incomplete",
            title="Incomplete paper",
            source="paper.pdf",
            review_status="draft",
            content_sha256="c" * 64,
        )
    )

    with pytest.raises(ValueError, match="authors, year"):
        catalog.set_review_status("sha256:incomplete", "published")

    assert catalog.get("sha256:incomplete").review_status == "draft"


def test_publication_allows_missing_doi_with_explicit_warning() -> None:
    readiness = publication_readiness(
        DocumentRecord(
            doc_id="bib:paper",
            title="Paper without DOI",
            authors=["Researcher"],
            year=1992,
            source="legacy-paper.pdf",
            source_type="journal",
            content_sha256="d" * 64,
        )
    )

    assert readiness["ready"] is True
    assert readiness["doi_status"] == "missing"
    assert readiness["warnings"][0]["code"] == "doi_missing"


def test_catalog_updates_editable_metadata_without_new_content_version(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    created = catalog.register(_record(digest="a" * 64)).record

    updated = catalog.update_metadata(
        created.doc_id,
        {
            "title": "Updated title",
            "authors": ["Alice", "Bob"],
            "year": 2025,
            "doi": "https://doi.org/10.1000/UPDATED.",
            "language": "zh-en",
            "domains": ["soil", "wheat"],
            "quality_tier": "A",
        },
    )

    assert updated is not None
    assert updated.title == "Updated title"
    assert updated.authors == ["Alice", "Bob"]
    assert updated.year == 2025
    assert updated.doi == "10.1000/updated"
    assert updated.language == "zh-en"
    assert updated.domains == ["soil", "wheat"]
    assert updated.quality_tier == "A"
    assert updated.version == created.version
    assert updated.content_sha256 == created.content_sha256
    assert updated.review_status == created.review_status
    assert updated.updated_at >= created.updated_at


def test_catalog_metadata_update_rejects_non_editable_or_invalid_fields(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    catalog.register(_record(digest="a" * 64))

    with pytest.raises(ValueError, match="Unsupported metadata fields"):
        catalog.update_metadata(
            "doi:10.1000/test",
            {"content_sha256": "b" * 64},
        )

    with pytest.raises(ValueError, match="quality_tier"):
        catalog.update_metadata(
            "doi:10.1000/test",
            {"quality_tier": "Z"},
        )


def test_ingestion_is_idempotent_and_populates_governance_metadata(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    retriever = _FakeRetriever()
    raw = "# 方法\n\n采用 SNV 处理近红外光谱。".encode()

    created = ingest_document_bytes(
        filename="paper.md",
        content=raw,
        retriever=retriever,
        catalog=catalog,
        title="Soil study",
        authors=["Alice"],
        year=2024,
        doi="10.1000/nir",
        domains=["soil"],
        quality_tier="B",
        review_status="published",
    )
    unchanged = ingest_document_bytes(
        filename="renamed.md",
        content=raw,
        retriever=retriever,
        catalog=catalog,
        title="Soil study",
        authors=["Alice"],
        year=2024,
        doi="10.1000/nir",
        review_status="published",
    )

    assert created.action == "created"
    assert created.chunks_added == 1
    assert unchanged.action == "unchanged"
    assert unchanged.chunks_added == 0
    assert len(retriever.replacements) == 1
    chunk = retriever.replacements[0][1][0]
    assert chunk.metadata["review_status"] == "published"
    assert chunk.metadata["content_sha256"] == content_sha256(raw)
    assert chunk.metadata["document_version"] == 1


def test_ingestion_replaces_changed_version_with_same_doi(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    retriever = _FakeRetriever()
    common = {
        "filename": "paper.md",
        "retriever": retriever,
        "catalog": catalog,
        "title": "Study",
        "authors": ["Alice"],
        "year": 2024,
        "doi": "10.1000/versioned",
        "review_status": "published",
    }

    first = ingest_document_bytes(content=b"first version", **common)
    second = ingest_document_bytes(content=b"second version", **common)

    assert first.record.version == 1
    assert second.action == "updated"
    assert second.record.version == 2
    assert len(retriever.replacements) == 2
    assert retriever.replacements[-1][1][0].metadata["document_version"] == 2


def test_ingestion_extracts_doi_from_document_text(tmp_path: Path) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    retriever = _FakeRetriever()

    result = ingest_document_bytes(
        filename="paper.md",
        content=(
            b"# Paper\n\nDOI: 10.5555/NIR.AUTO.2026\n\nSNV preprocessing was evaluated."
        ),
        retriever=retriever,
        catalog=catalog,
        title="Auto DOI",
        authors=["Alice"],
        year=2026,
    )

    assert result.record.doi == "10.5555/nir.auto.2026"
    assert result.record.doc_id == "doi:10.5555/nir.auto.2026"
    assert retriever.replacements[0][1][0].metadata["doi"] == result.record.doi


def test_ingestion_rejects_invalid_supplied_doi_before_indexing(
    tmp_path: Path,
) -> None:
    catalog = KnowledgeCatalog(tmp_path / "catalog.sqlite3")
    retriever = _FakeRetriever()

    with pytest.raises(ValueError, match="canonical form"):
        ingest_document_bytes(
            filename="paper.md",
            content=b"paper content",
            retriever=retriever,
            catalog=catalog,
            doi="invalid identifier",
        )

    assert retriever.replacements == []
