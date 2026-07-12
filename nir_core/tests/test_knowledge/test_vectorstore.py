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

from nir_core.knowledge.base import Chunk
from nir_core.knowledge.vectorstore import ChromaDBRetriever

# Use the local model path (avoids HuggingFace network dependency).
_LOCAL_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "..", "knowledge", "all-MiniLM-L6-v2"
)
_LOCAL_MODEL = os.path.normpath(_LOCAL_MODEL)


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
