"""Converted upload previews must not disclose mixed spectral/Test rows."""

import pytest

from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware, _extract_outline_for_file


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".npz", ".mat", ".txt"])
def test_spectral_upload_and_its_converted_markdown_have_no_preview(tmp_path, suffix):
    source = tmp_path / ("spectra" + suffix)
    source.write_bytes(b"input")
    converted = tmp_path / "spectra.md"
    converted.write_text("# Test targets\n99999 88888 77777", encoding="utf-8")
    assert _extract_outline_for_file(source) == ([], [])
    assert _extract_outline_for_file(converted) == ([], [])


def test_ordinary_document_outline_remains_available(tmp_path):
    source = tmp_path / "notes.pdf"
    source.write_bytes(b"input")
    (tmp_path / "notes.md").write_text("# Methods\nReference notes", encoding="utf-8")
    outline, preview = _extract_outline_for_file(source)
    assert outline and outline[0]["title"] == "Methods"
    assert not preview


def test_sealed_upload_message_directs_to_trusted_ingestion_only():
    text = UploadsMiddleware()._create_files_message([{"filename": "spectra.csv", "size": 100, "path": "/mnt/user-data/uploads/spectra.csv", "extension": ".csv"}], [], sealed=True)
    assert "nir_prepare_search" in text
    assert "nir_load_data" not in text
    assert "nir_inspect" not in text
