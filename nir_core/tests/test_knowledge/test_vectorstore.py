"""Tests for the ChromaDB retriever.

These tests require the heavy ``chromadb`` and ``sentence-transformers``
dependencies. They are skipped automatically when those packages are not
installed or broken — run with ``pip install chromadb sentence-transformers``
to activate them.
"""

from __future__ import annotations

import os
from pathlib import Path

# Prevent transformers from importing tensorflow, which may be broken
# due to numpy/h5py binary incompatibility in some environments. This
# must be set BEFORE sentence_transformers is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# Force offline mode so SentenceTransformer doesn't try to reach
# huggingface.co (which is blocked in CN) — we use the local model copy.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest

# Skip the entire module if heavy deps are missing OR broken (e.g. the
# numpy/h5py binary incompatibility that breaks sentence_transformers in
# some environments). ``importorskip`` only catches ImportError, so we
# wrap in a broader try/except for ValueError etc.
for _mod in ("chromadb", "sentence_transformers"):
    try:
        __import__(_mod)
    except Exception as exc:  # noqa: BLE001 — catch binary-incompat errors too
        pytest.skip(f"{_mod} not available: {exc}", allow_module_level=True)

from nir_core.knowledge.base import Chunk  # noqa: E402
from nir_core.knowledge.config import get_config  # noqa: E402
from nir_core.knowledge.vectorstore import ChromaDBRetriever  # noqa: E402

# Use the local model path (avoids HuggingFace network dependency).
_LOCAL_MODEL = get_config().embedding_model
if not os.path.isdir(_LOCAL_MODEL):
    pytest.skip(
        f"local BGE-M3 model not found: {_LOCAL_MODEL}",
        allow_module_level=True,
    )


@pytest.fixture
def retriever(tmp_path: Path) -> ChromaDBRetriever:
    """A fresh ChromaDB retriever with a temporary DB path."""
    return ChromaDBRetriever(
        db_path=str(tmp_path / "chromadb"),
        embedding_model=_LOCAL_MODEL,
        collection_name="test_collection",
    )


def _make_chunks() -> list[Chunk]:
    """Build a small set of NIR-paper-like chunks."""
    return [
        Chunk(
            id="paper1_chunk_0",
            content="SNV (Standard Normal Variate) is a scatter correction method "
            "commonly applied to NIR soil spectra.",
            source="soil_paper.pdf",
            chunk_index=0,
            metadata={
                "doc_id": "paper1",
                "title": "Soil Analysis",
                "methods": ["snv"],
                "datasets": ["soil"],
                "review_status": "published",
            },
        ),
        Chunk(
            id="paper2_chunk_0",
            content="PLS regression achieved R2=0.92 on corn moisture prediction "
            "using NIR spectroscopy.",
            source="corn_paper.pdf",
            chunk_index=0,
            metadata={
                "doc_id": "paper2",
                "title": "Corn Moisture",
                "models": ["pls"],
                "metrics": ["r2"],
                "datasets": ["corn"],
                "review_status": "published",
            },
        ),
    ]


def test_add_and_search(retriever: ChromaDBRetriever) -> None:
    """Added chunks are retrievable by a semantic query."""
    chunks = _make_chunks()
    count = retriever.add_documents(chunks)
    assert count == 2

    results = retriever.search("SNV scatter correction soil", top_k=2)
    assert len(results) >= 1
    # The soil paper should rank higher for this query.
    assert "SNV" in results[0].chunk.content or "soil" in results[0].chunk.content


def test_search_empty_db(retriever: ChromaDBRetriever) -> None:
    """An empty collection returns no results."""
    results = retriever.search("anything", top_k=5)
    assert results == []


def test_list_documents(retriever: ChromaDBRetriever) -> None:
    """list_documents returns one entry per distinct doc_id."""
    retriever.add_documents(_make_chunks())
    docs = retriever.list_documents()
    assert len(docs) == 2
    doc_ids = [d["doc_id"] for d in docs]
    assert "paper1" in doc_ids
    assert "paper2" in doc_ids
    # Each doc should have chunk_count.
    for doc in docs:
        assert doc.get("chunk_count", 0) >= 1


def test_delete_document(retriever: ChromaDBRetriever) -> None:
    """delete_document removes all chunks of a doc."""
    retriever.add_documents(_make_chunks())
    success = retriever.delete_document("paper1")
    assert success is True
    docs = retriever.list_documents()
    assert all(d["doc_id"] != "paper1" for d in docs)
    assert any(d["doc_id"] == "paper2" for d in docs)


def test_get_related_entities_noop(retriever: ChromaDBRetriever) -> None:
    """get_related_entities returns [] in the ChromaDB-only implementation."""
    retriever.add_documents(_make_chunks())
    assert retriever.get_related_entities("snv") == []


def test_score_in_range(retriever: ChromaDBRetriever) -> None:
    """Scores are in [0, 1]."""
    retriever.add_documents(_make_chunks())
    results = retriever.search("NIR spectroscopy", top_k=2)
    for r in results:
        assert 0.0 <= r.score <= 1.0


def test_search_excludes_unpublished_documents_by_default(
    retriever: ChromaDBRetriever,
) -> None:
    chunks = _make_chunks()
    chunks[0].metadata["review_status"] = "draft"
    retriever.add_documents(chunks)

    published = retriever.search("SNV soil", top_k=2)
    all_statuses = retriever.search("SNV soil", top_k=2, published_only=False)

    assert all(
        result.chunk.metadata.get("review_status") == "published"
        for result in published
    )
    assert any(
        result.chunk.metadata.get("review_status") == "draft" for result in all_statuses
    )


def test_replace_document_removes_stale_chunks(retriever: ChromaDBRetriever) -> None:
    chunks = _make_chunks()
    chunks.append(
        Chunk(
            id="paper1_chunk_1",
            content="Old second chunk.",
            source="soil_paper.pdf",
            chunk_index=1,
            metadata={"doc_id": "paper1", "review_status": "published"},
        )
    )
    retriever.add_documents(chunks)

    replacement = _make_chunks()[0]
    replacement.content = "Updated soil paper."
    assert retriever.replace_document("paper1", [replacement]) == 1

    paper1 = [doc for doc in retriever.list_documents() if doc["doc_id"] == "paper1"]
    assert paper1[0]["chunk_count"] == 1


def test_update_document_metadata_can_publish_draft(
    retriever: ChromaDBRetriever,
) -> None:
    chunk = _make_chunks()[0]
    chunk.metadata["review_status"] = "draft"
    retriever.add_documents([chunk])

    assert retriever.search("SNV soil") == []
    assert (
        retriever.update_document_metadata("paper1", {"review_status": "published"})
        is True
    )
    assert retriever.search("SNV soil")
