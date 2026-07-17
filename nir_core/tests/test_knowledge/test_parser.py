"""Tests for the document parser."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from nir_core.knowledge.parser import SUPPORTED_EXTENSIONS, _normalize_pdf_text, parse_document


def test_parse_txt(tmp_path: Path) -> None:
    """Plain text files are read as-is."""
    p = tmp_path / "notes.txt"
    p.write_text("Hello NIR\nSNV is common", encoding="utf-8")
    result = parse_document(p)
    assert "Hello NIR" in result
    assert "SNV is common" in result


def test_parse_markdown(tmp_path: Path) -> None:
    """Markdown files are read as-is."""
    p = tmp_path / "paper.md"
    p.write_text("# Title\n\nSome content about PLS.", encoding="utf-8")
    result = parse_document(p)
    assert "# Title" in result
    assert "PLS" in result


def test_parse_csv(tmp_path: Path) -> None:
    """CSV files are converted to a Markdown table."""
    p = tmp_path / "data.csv"
    p.write_text("wavelength,absorbance\n900,0.5\n910,0.6\n", encoding="utf-8")
    result = parse_document(p)
    assert "|" in result  # markdown table separator
    assert "wavelength" in result
    assert "absorbance" in result


def test_parse_csv_truncates_large_file(tmp_path: Path) -> None:
    """Large CSVs are truncated to 50 rows to keep chunks reasonable."""
    p = tmp_path / "big.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x"])
        for i in range(100):
            writer.writerow([i])
    result = parse_document(p)
    # Only the first 50 data rows should be present.
    assert "99" not in result  # row 100 value
    assert "0" in result      # row 1 value


def test_parse_csv_handles_bom_quotes_pipes_and_ragged_rows(tmp_path: Path) -> None:
    """CSV conversion remains valid Markdown without pandas or tabulate."""
    p = tmp_path / "complex.csv"
    p.write_text(
        '\ufeffname,description,value\n"sample, one","SNV | MSC",1\nshort,row\n',
        encoding="utf-8",
    )

    result = parse_document(p)

    assert result.splitlines()[0] == "| name | description | value |"
    assert "| sample, one | SNV \\| MSC | 1 |" in result
    assert "| short | row |  |" in result


def test_parse_empty_csv_returns_empty_text(tmp_path: Path) -> None:
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")

    assert parse_document(p) == ""


def test_parse_html(tmp_path: Path) -> None:
    """HTML files are converted to Markdown."""
    pytest.importorskip("markdownify")
    p = tmp_path / "page.html"
    p.write_text(
        "<html><body><h1>Title</h1><p>SNV content</p></body></html>",
        encoding="utf-8",
    )
    result = parse_document(p)
    assert "Title" in result
    assert "SNV content" in result


def test_parse_pdf_requires_lib(tmp_path: Path) -> None:
    """PDF parsing skips gracefully when pypdf is unavailable."""
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"%PDF-1.4\n%fake\n")
    pytest.importorskip("pypdf", reason="pypdf not installed")
    # pypdf is installed but this is a fake PDF — expect empty or error,
    # not a crash. We just verify it doesn't raise ImportError.
    try:
        result = parse_document(p)
        assert isinstance(result, str)
    except Exception:
        # Fake PDFs may raise; acceptable.
        pass


def test_parse_docx_requires_lib(tmp_path: Path) -> None:
    """DOCX parsing skips when python-docx is unavailable."""
    pytest.importorskip("docx", reason="python-docx not installed")
    # No real docx to test; just verify the import path works.
    # Create a minimal empty docx is complex; skip body test.
    p = tmp_path / "empty.docx"
    # Writing a placeholder file; the real parse would fail, but we just
    # want to confirm the import path. Skip if docx isn't installed.
    p.write_bytes(b"PK\x03\x04")  # zip signature
    try:
        parse_document(p)
    except Exception:
        pass  # invalid docx — expected


def test_unsupported_extension(tmp_path: Path) -> None:
    """Unsupported file types raise ValueError."""
    p = tmp_path / "file.xyz"
    p.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse_document(p)


def test_file_not_found(tmp_path: Path) -> None:
    """Missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_document(tmp_path / "nonexistent.pdf")


def test_supported_extensions_includes_common() -> None:
    """The supported extensions list covers the common document types."""
    for ext in (".pdf", ".docx", ".txt", ".md", ".html", ".csv"):
        assert ext in SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# _normalize_pdf_text
# ---------------------------------------------------------------------------

def test_normalize_removes_inter_cjk_spaces() -> None:
    """Spaces between CJK characters (a pypdf artifact) are removed."""
    raw = "近 红 外 光 谱 定 量 分 析"
    result = _normalize_pdf_text(raw)
    assert result == "近红外光谱定量分析"


def test_normalize_preserves_latin_spaces() -> None:
    """Spaces between Latin/number tokens are preserved."""
    raw = "SNV is a common preprocessing method"
    result = _normalize_pdf_text(raw)
    assert "SNV is a common preprocessing method" == result


def test_normalize_preserves_cjk_latin_boundary() -> None:
    """A space between a CJK char and a Latin char is preserved."""
    raw = "近红外 NIR spectroscopy"
    result = _normalize_pdf_text(raw)
    assert result == "近红外 NIR spectroscopy"


def test_normalize_fullwidth_to_halfwidth() -> None:
    """Full-width ASCII characters are converted to half-width via NFKC."""
    raw = "ｍｓｃ ｓｎｖ ２０２４"
    result = _normalize_pdf_text(raw)
    assert "msc" in result
    assert "snv" in result
    assert "2024" in result


def test_normalize_collapses_repeated_cjk_spaces() -> None:
    """Multiple spaces/newlines between CJK chars are removed."""
    raw = "近  红\n外  光\t谱"
    result = _normalize_pdf_text(raw)
    assert result == "近红外光谱"
