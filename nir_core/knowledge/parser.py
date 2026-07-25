"""Document parser: PDF / Word / HTML / CSV / TXT / Markdown -> Markdown text.

Adapted from yuxi-knowledge ``unified.py`` with MinIO / async dependencies
stripped. Heavy-lifting dependencies (``pypdf``, ``python-docx``, and
``markdownify``) are imported lazily so the module can be
imported even when those packages are not installed — callers only pay the
import cost for the format they actually parse.
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from itertools import islice
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".csv")

# Matches whitespace between two CJK characters (for PDF extraction cleanup).
# CJK Unified Ideographs + CJK punctuation.
_CJK_RANGE = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
_CJK_SPACE_CJK = re.compile(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_RANGE}])")


def _normalize_pdf_text(text: str) -> str:
    """Normalize PDF-extracted text for better entity matching and search.

    PDF extractors (especially pypdf) often produce:
    1. Full-width characters (ｍ → m, ２ → 2) — fixed via NFKC normalization.
    2. Spaces between every CJK character (近 红 外 → 近红外) — fixed by
       removing whitespace between adjacent CJK characters.
    """
    # NFKC: full-width ASCII → half-width, compatibility decomposition.
    text = unicodedata.normalize("NFKC", text)
    # Remove inter-CJK whitespace (common pypdf artifact for Chinese text).
    text = _CJK_SPACE_CJK.sub("", text)
    return text


def _format_pdf_pages(pages: list[str]) -> str:
    """Join extracted pages with machine-readable, one-based page markers."""
    return "\n\n".join(
        f"<!-- page: {page_number} -->\n\n{page_text}"
        for page_number, page_text in enumerate(pages, 1)
    )


def parse_document(file_path: str | Path, params: dict[str, Any] | None = None) -> str:
    """Parse any supported document into Markdown text.

    Args:
        file_path: Path to the document.
        params: Optional parser hints (currently unused; reserved for OCR
            / table-extraction flags).

    Returns:
        Markdown text.

    Raises:
        FileNotFoundError: File does not exist.
        ValueError: Unsupported file extension.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        return _parse_pdf(path, params or {})
    if ext == ".docx":
        return _parse_docx(path)
    if ext in (".html", ".htm"):
        return _parse_html(path)
    if ext == ".csv":
        return _parse_csv(path)

    raise ValueError(
        f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    )


def _parse_pdf(path: Path, params: dict[str, Any]) -> str:
    """PDF -> Markdown via PyPDFLoader (langchain-community).

    Falls back to ``pypdf`` directly if langchain is unavailable or its
    import chain breaks (e.g. tensorflow/numpy binary incompatibility).
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
        docs = loader.load()
        text = _format_pdf_pages([document.page_content for document in docs])
        if text.strip():
            return _normalize_pdf_text(text)
        logger.warning("PyPDFLoader returned empty content for %s; trying pypdf", path)
    except Exception as exc:  # noqa: BLE001 — catch ValueError from tf/h5py too
        logger.info(
            "PyPDFLoader unavailable (%s); using pypdf directly", type(exc).__name__
        )

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _normalize_pdf_text(_format_pdf_pages(pages))


def _parse_docx(path: Path) -> str:
    """DOCX -> Markdown preserving headings."""
    import docx

    doc = docx.Document(str(path))
    lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if "heading 1" in style:
            lines.append(f"# {text}")
        elif "heading 2" in style:
            lines.append(f"## {text}")
        elif "heading 3" in style:
            lines.append(f"### {text}")
        elif "heading 4" in style or "heading 5" in style or "heading 6" in style:
            lines.append(f"#### {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _parse_html(path: Path) -> str:
    """HTML -> Markdown."""
    from markdownify import markdownify as md_convert

    content = path.read_text(encoding="utf-8", errors="replace")
    return md_convert(content, heading_style="ATX")


def _parse_csv(path: Path) -> str:
    """CSV -> Markdown table (first 50 rows to avoid huge chunks)."""

    def markdown_cell(value: str) -> str:
        return (
            value.replace("|", r"\|")
            .replace("\r\n", "<br>")
            .replace("\n", "<br>")
            .replace("\r", "<br>")
        )

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return ""
        rows = list(islice(reader, 50))

    width = max(len(header), *(len(row) for row in rows)) if rows else len(header)

    def normalized(row: list[str]) -> list[str]:
        return [
            markdown_cell(value)
            for value in [*row, *([""] * (width - len(row)))][:width]
        ]

    lines = [
        f"| {' | '.join(normalized(header))} |",
        f"| {' | '.join(['---'] * width)} |",
    ]
    lines.extend(f"| {' | '.join(normalized(row))} |" for row in rows)
    return "\n".join(lines)
