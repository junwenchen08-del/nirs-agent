"""Tests for the smart document chunker."""

from __future__ import annotations

from nir_core.knowledge.chunker import chunk_document


def test_chunk_naive_basic() -> None:
    """Naive strategy merges paragraphs up to max_tokens."""
    text = "\n\n".join(f"Paragraph {i}." for i in range(10))
    chunks = chunk_document(text, source="test.txt", strategy="naive", max_tokens=5)
    # Should produce multiple chunks.
    assert len(chunks) > 1
    # Each chunk should have proper metadata.
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.source == "test.txt"
        assert "doc_id" in chunk.metadata
        assert chunk.id.endswith(f"_chunk_{i}")


def test_chunk_naive_single_paragraph() -> None:
    """A short document fits in a single chunk."""
    text = "This is a short paragraph."
    chunks = chunk_document(text, source="short.md", strategy="naive", max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].content.strip() == "This is a short paragraph."


def test_chunk_naive_with_overlap() -> None:
    """Overlap keeps some paragraphs from the previous chunk."""
    paragraphs = [f"Paragraph {i} with some words here." for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks_no_overlap = chunk_document(
        text, source="t.txt", strategy="naive", max_tokens=20, overlap_percent=0
    )
    chunks_overlap = chunk_document(
        text, source="t.txt", strategy="naive", max_tokens=20, overlap_percent=50
    )
    # With overlap, the total number of chunks should be >= no-overlap.
    assert len(chunks_overlap) >= len(chunks_no_overlap)
    # And the overlap chunk should share content with the previous.
    if len(chunks_overlap) >= 2:
        last_para_prev = chunks_overlap[0].content.split("\n\n")[-1].strip()
        assert last_para_prev in chunks_overlap[1].content


def test_chunk_by_section() -> None:
    """Section strategy splits by Markdown headings."""
    text = (
        "# Introduction\n\nIntro content about NIR.\n\n"
        "# Methods\n\nWe used SNV and PLS.\n\n"
        "## Preprocessing\n\nSG smoothing applied.\n\n"
        "# Results\n\nR2 was 0.9."
    )
    chunks = chunk_document(text, source="paper.md", strategy="section")
    assert len(chunks) >= 3
    # Each chunk should start with a heading.
    contents = [c.content for c in chunks]
    assert any("Introduction" in c for c in contents)
    assert any("Methods" in c for c in contents)
    assert any("Preprocessing" in c for c in contents)
    assert any("Results" in c for c in contents)


def test_chunk_by_section_long_paragraph() -> None:
    """Long sections are recursively split with naive."""
    # A section with many words.
    long_body = "\n\n".join(f"Word word word {i}." for i in range(50))
    text = f"# Long Section\n\n{long_body}"
    chunks = chunk_document(text, source="t.md", strategy="section", max_tokens=20)
    # Should split into multiple chunks.
    assert len(chunks) > 1


def test_chunk_qa_format() -> None:
    """QA strategy extracts Q/A pairs."""
    text = (
        "Q: What is SNV?\n"
        "A: Standard Normal Variate scatter correction.\n\n"
        "Q: What is PLS?\n"
        "A: Partial Least Squares regression."
    )
    chunks = chunk_document(text, source="faq.md", strategy="qa")
    assert len(chunks) == 2
    assert "SNV" in chunks[0].content
    assert "PLS" in chunks[1].content


def test_chunk_qa_chinese_format() -> None:
    """QA strategy handles Chinese 问/答 format."""
    text = (
        "问：什么是 SNV？\n"
        "答：标准正态变量散射校正。\n\n"
        "问：什么是 PLS？\n"
        "答：偏最小二乘回归。"
    )
    chunks = chunk_document(text, source="faq_cn.md", strategy="qa")
    assert len(chunks) == 2


def test_chunk_qa_fallback() -> None:
    """QA strategy falls back to naive when no Q/A pairs found."""
    text = "Just a regular paragraph about NIR spectroscopy."
    chunks = chunk_document(text, source="t.md", strategy="qa")
    assert len(chunks) == 1
    assert "NIR" in chunks[0].content


def test_chunk_empty_document() -> None:
    """Empty input produces no chunks."""
    chunks = chunk_document("", source="empty.md", strategy="section")
    assert chunks == []


def test_chunk_doc_id_explicit() -> None:
    """Explicit doc_id is used in chunk IDs."""
    chunks = chunk_document(
        "content here", source="t.md", strategy="naive", doc_id="my_paper"
    )
    assert chunks[0].id == "my_paper_chunk_0"
    assert chunks[0].metadata["doc_id"] == "my_paper"


def test_chunk_unknown_strategy_raises() -> None:
    """Unknown strategy raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        chunk_document("text", source="t.md", strategy="invalid")  # type: ignore[arg-type]


def test_chunk_overlap_clamped() -> None:
    """Overlap > 50% is clamped to 50%."""
    text = "\n\n".join(f"P{i}." for i in range(20))
    # Should not crash.
    chunks = chunk_document(
        text, source="t.txt", strategy="naive", max_tokens=5, overlap_percent=200
    )
    assert len(chunks) > 0
