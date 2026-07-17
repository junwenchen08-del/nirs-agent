from __future__ import annotations

import base64
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.routers import knowledge


def _make_client(monkeypatch, handler) -> TestClient:
    app = FastAPI()
    app.include_router(knowledge.router)
    monkeypatch.setattr(knowledge, "_kb_request", handler)
    return TestClient(app)


def test_batch_upload_forwards_each_valid_file_with_metadata(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_kb_request(method, path, body=None, timeout=60.0):
        assert method == "POST"
        assert path == "/documents"
        assert timeout == 120.0
        assert body is not None
        calls.append(body)
        raw = base64.b64decode(body["content_b64"])
        return {
            "doc_id": f"doc-{body['filename']}",
            "chunks_added": len(raw),
        }

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.post(
        "/api/knowledge/documents/batch",
        data={"title": "Batch title", "year": "2024"},
        files=[
            ("files", ("paper.txt", b"alpha", "text/plain")),
            ("files", ("notes.md", b"beta", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["succeeded"] == 2
    assert payload["failed"] == 0
    assert payload["total_chunks_added"] == 9
    assert [item["filename"] for item in payload["results"]] == ["paper.txt", "notes.md"]
    assert all(item["success"] for item in payload["results"])

    assert [call["filename"] for call in calls] == ["paper.txt", "notes.md"]
    assert all(call["title"] == "Batch title" for call in calls)
    assert all(call["year"] == 2024 for call in calls)
    assert [base64.b64decode(call["content_b64"]) for call in calls] == [b"alpha", b"beta"]


def test_batch_upload_reports_partial_failures_without_forwarding_invalid_files(monkeypatch) -> None:
    forwarded_filenames: list[str] = []

    def fake_kb_request(method, path, body=None, timeout=60.0):
        assert body is not None
        forwarded_filenames.append(body["filename"])
        if body["filename"] == "server.md":
            return {"error": "parser failed"}
        return {"doc_id": "doc-ok", "chunks_added": 2}

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.post(
        "/api/knowledge/documents/batch",
        files=[
            ("files", ("ok.txt", b"ok", "text/plain")),
            ("files", ("bad.exe", b"x", "application/octet-stream")),
            ("files", ("empty.md", b"", "text/markdown")),
            ("files", ("server.md", b"boom", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 4
    assert payload["succeeded"] == 1
    assert payload["failed"] == 3
    assert payload["total_chunks_added"] == 2
    assert forwarded_filenames == ["ok.txt", "server.md"]

    results = {item["filename"]: item for item in payload["results"]}
    assert results["ok.txt"]["success"] is True
    assert results["bad.exe"]["success"] is False
    assert "Unsupported file type" in results["bad.exe"]["error"]
    assert results["empty.md"]["success"] is False
    assert results["empty.md"]["error"] == "Empty file"
    assert results["server.md"]["success"] is False
    assert results["server.md"]["error"] == "parser failed"


def test_batch_upload_rejects_too_many_files_before_forwarding(monkeypatch) -> None:
    forwarded = False

    def fake_kb_request(method, path, body=None, timeout=60.0):
        nonlocal forwarded
        forwarded = True
        return {"doc_id": "unexpected", "chunks_added": 1}

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.post(
        "/api/knowledge/documents/batch",
        files=[("files", (f"doc-{idx}.txt", b"x", "text/plain")) for idx in range(knowledge._MAX_BATCH_FILES + 1)],
    )

    assert response.status_code == 400
    assert "Too many files" in response.json()["detail"]
    assert forwarded is False
