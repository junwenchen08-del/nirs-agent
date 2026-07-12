"""Smart document chunker.

Three strategies:
- ``naive``: merge paragraphs up to ``max_tokens`` with optional overlap.
- ``section``: split by Markdown headings (#, ##, ###); over-long sections
  fall back to naive.
- ``qa``: extract Q/A pairs (handles both ``Q:``/``A:`` and Chinese
  ``问：``/``答：``).

Token counts are approximated by whitespace word count — sufficient for
chunk-size control without pulling in a tokenizer dependency. NIR papers
are predominantly English, so this heuristic is adequate.
"""

from __future__ import annotations

import re
import uuid
from typing import Literal

from nir_core.knowledge.base import Chunk

ChunkStrategy = Literal["naive", "section", "qa"]


def chunk_document(
    markdown: str,
    source: str,
    strategy: ChunkStrategy = "section",
    max_tokens: int = 512,
    overlap_percent: int = 0,
    doc_id: str | None = None,
) -> list[Chunk]:
    """Chunk a Markdown document into :class:`Chunk` objects.

    Args:
        markdown: Document text.
        source: Source file name / path (stored on each chunk).
        strategy: ``"naive"``, ``"section"`` (default) or ``"qa"``.
        max_tokens: Target max word count per chunk.
        overlap_percent: Overlap between adjacent chunks (naive strategy
            only), as a percentage of chunk size (0-50).
        doc_id: Document ID. Auto-generated UUID4 if omitted.

    Returns:
        Non-empty list of chunks (empty input -> empty list).
    """
    doc_id = doc_id or str(uuid.uuid4())
    overlap_percent = max(0, min(overlap_percent, 50))

    if strategy == "naive":
        texts = _chunk_naive(markdown, max_tokens, overlap_percent)
    elif strategy == "section":
        texts = _chunk_by_section(markdown, max_tokens)
    elif strategy == "qa":
        texts = _chunk_qa(markdown)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    return [
        Chunk(
            id=f"{doc_id}_chunk_{i}",
            content=text,
            source=source,
            chunk_index=i,
            metadata={"doc_id": doc_id},
        )
        for i, text in enumerate(texts)
        if text.strip()
    ]


# ---------------------------------------------------------------------------
# naive
# ---------------------------------------------------------------------------


def _chunk_naive(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Merge paragraphs up to ``max_tokens`` words with optional overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para.split())
        # If adding this paragraph exceeds the budget and we already have
        # content, flush.
        if current and current_len + para_len > max_tokens:
            chunks.append("\n\n".join(current))
            if overlap > 0:
                keep = max(1, len(current) * overlap // 100)
                current = current[-keep:]
                current_len = sum(len(c.split()) for c in current)
            else:
                current = []
                current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# section
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


def _chunk_by_section(text: str, max_tokens: int) -> list[str]:
    """Split by Markdown headings; over-long sections recurse to naive."""
    # Split keeping the headings so we know where each section starts.
    parts = _HEADING_RE.split(text)

    chunks: list[str] = []
    current_header = ""
    current_content: list[str] = []

    def _flush() -> None:
        nonlocal current_header, current_content
        body = "\n".join(current_content).strip()
        if not body and not current_header:
            current_content = []
            return
        section = f"{current_header}\n\n{body}".strip() if current_header else body
        if section:
            if len(section.split()) > max_tokens:
                chunks.extend(_chunk_naive(section, max_tokens, 0))
            else:
                chunks.append(section)
        current_header = ""
        current_content = []

    for part in parts:
        if not part.strip():
            continue
        if _HEADING_RE.match(part):
            _flush()
            current_header = part.strip()
        else:
            current_content.append(part)

    _flush()
    return chunks


# ---------------------------------------------------------------------------
# qa
# ---------------------------------------------------------------------------

_QA_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Q[:：]|问[:：])\s*(.+?)\s*\n\s*(?:A[:：]|答[:：])\s*(.+?)"
    r"(?=\n\s*(?:Q[:：]|问[:：])|$)",
    re.DOTALL,
)


def _chunk_qa(text: str) -> list[str]:
    """Extract Q/A pairs as individual chunks; fall back to naive."""
    matches = _QA_PATTERN.findall(text)
    if not matches:
        return _chunk_naive(text, 512, 0)
    return [f"Q: {q.strip()}\nA: {a.strip()}" for q, a in matches]
