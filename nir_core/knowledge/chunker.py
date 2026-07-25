"""Smart document chunker.

Three strategies:
- ``naive``: merge paragraphs up to ``max_tokens`` with optional overlap.
- ``section``: split by Markdown headings (#, ##, ###); over-long sections
  fall back to naive.
- ``qa``: extract Q/A pairs (handles both ``Q:``/``A:`` and Chinese
  ``问：``/``答：``).

Token counts use a deterministic CJK-aware estimate without pulling in a
model tokenizer. Each CJK character and punctuation mark contributes to the
budget while contiguous Latin letters/numbers count as one approximate token.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from nir_core.knowledge.base import Chunk
from nir_core.knowledge.governance import stable_document_id

ChunkStrategy = Literal["naive", "section", "qa"]
CHUNKER_VERSION = "cjk-section-v2"

_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*|[^\s]")
_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->", re.IGNORECASE)


@dataclass(frozen=True)
class _ChunkDraft:
    content: str
    section_path: list[str]


def estimate_tokens(text: str) -> int:
    """Estimate mixed Chinese/English token length deterministically."""
    return len(_TOKEN_RE.findall(text))


def chunk_document(
    markdown: str,
    source: str,
    strategy: ChunkStrategy = "section",
    max_tokens: int = 512,
    overlap_percent: int = 0,
    doc_id: str | None = None,
    chunker_version: str = CHUNKER_VERSION,
) -> list[Chunk]:
    """Chunk a Markdown document into :class:`Chunk` objects.

    Args:
        markdown: Document text.
        source: Source file name / path (stored on each chunk).
        strategy: ``"naive"``, ``"section"`` (default) or ``"qa"``.
        max_tokens: Target max word count per chunk.
        overlap_percent: Overlap between adjacent chunks (naive strategy
            only), as a percentage of chunk size (0-50).
        doc_id: Stable document ID. Content-derived when omitted.
        chunker_version: Version recorded in chunk IDs and metadata.

    Returns:
        Non-empty list of chunks (empty input -> empty list).
    """
    doc_id = doc_id or stable_document_id(content=markdown.encode("utf-8"))
    overlap_percent = max(0, min(overlap_percent, 50))

    if strategy == "naive":
        drafts = [
            _ChunkDraft(text, [])
            for text in _chunk_naive(markdown, max_tokens, overlap_percent)
        ]
    elif strategy == "section":
        drafts = _chunk_by_section(markdown, max_tokens)
    elif strategy == "qa":
        drafts = [_ChunkDraft(text, []) for text in _chunk_qa(markdown, max_tokens)]
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")

    chunks: list[Chunk] = []
    for draft in drafts:
        cleaned_content = _PAGE_MARKER_RE.sub("", draft.content).strip()
        if not cleaned_content:
            continue
        index = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc_id}#{chunker_version}-{index:04d}",
                content=cleaned_content,
                source=source,
                chunk_index=index,
                metadata={
                    "doc_id": doc_id,
                    "chunker_version": chunker_version,
                    "section_path": draft.section_path,
                    **_page_metadata(draft.content),
                },
            )
        )
    for index, chunk in enumerate(chunks):
        chunk.metadata["previous_chunk_id"] = chunks[index - 1].id if index > 0 else ""
        chunk.metadata["next_chunk_id"] = (
            chunks[index + 1].id if index + 1 < len(chunks) else ""
        )
    return chunks


# ---------------------------------------------------------------------------
# naive
# ---------------------------------------------------------------------------


def _chunk_naive(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Merge paragraphs to a CJK-aware budget with optional overlap."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    paragraphs: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()):
        paragraphs.extend(_split_oversized(paragraph, max_tokens))
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = estimate_tokens(para)
        # If adding this paragraph exceeds the budget and we already have
        # content, flush.
        if current and current_len + para_len > max_tokens:
            chunks.append("\n\n".join(current))
            if overlap > 0:
                keep = max(1, len(current) * overlap // 100)
                current = current[-keep:]
                current_len = sum(estimate_tokens(c) for c in current)
            else:
                current = []
                current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;\.])\s*", text)
        if part.strip()
    ]
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_parts = _hard_split(sentence, max_tokens)
        for part in sentence_parts:
            part_len = estimate_tokens(part)
            if current and current_len + part_len > max_tokens:
                pieces.append("".join(current).strip())
                current = []
                current_len = 0
            current.append(part)
            current_len += part_len
    if current:
        pieces.append("".join(current).strip())
    return pieces


def _hard_split(text: str, max_tokens: int) -> list[str]:
    matches = list(_TOKEN_RE.finditer(text))
    if len(matches) <= max_tokens:
        return [text]
    pieces: list[str] = []
    start = 0
    for token_index in range(max_tokens, len(matches), max_tokens):
        end = matches[token_index].start()
        pieces.append(text[start:end].strip())
        start = end
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


# ---------------------------------------------------------------------------
# section
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)


def _chunk_by_section(text: str, max_tokens: int) -> list[_ChunkDraft]:
    """Split by Markdown headings while retaining the full heading path."""
    drafts: list[_ChunkDraft] = []
    section_path: list[str] = []
    current_path: list[str] = []
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_lines
        section = "\n".join(current_lines).strip()
        if section:
            drafts.extend(
                _ChunkDraft(content=part, section_path=list(current_path))
                for part in _chunk_naive(section, max_tokens, 0)
            )
        current_lines = []

    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            _flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            section_path = section_path[: level - 1]
            section_path.append(title)
            current_path = list(section_path)
            current_lines.append(line)
        else:
            current_lines.append(line)
    _flush()
    return drafts


# ---------------------------------------------------------------------------
# qa
# ---------------------------------------------------------------------------

_QA_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Q[:：]|问[:：])\s*(.+?)\s*\n\s*(?:A[:：]|答[:：])\s*(.+?)"
    r"(?=\n\s*(?:Q[:：]|问[:：])|$)",
    re.DOTALL,
)


def _chunk_qa(text: str, max_tokens: int) -> list[str]:
    """Extract Q/A pairs as individual chunks; fall back to naive."""
    matches = _QA_PATTERN.findall(text)
    if not matches:
        return _chunk_naive(text, max_tokens, 0)
    chunks: list[str] = []
    for question, answer in matches:
        chunks.extend(
            _chunk_naive(f"Q: {question.strip()}\nA: {answer.strip()}", max_tokens, 0)
        )
    return chunks


def _page_metadata(text: str) -> dict[str, int]:
    pages = [int(page) for page in _PAGE_MARKER_RE.findall(text)]
    if not pages:
        return {}
    return {"page_start": min(pages), "page_end": max(pages)}
