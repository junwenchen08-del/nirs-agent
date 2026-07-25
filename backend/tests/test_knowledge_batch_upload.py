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
        assert timeout == knowledge._UPLOAD_TIMEOUT_SECONDS
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
    assert all(call["review_status"] == "draft" for call in calls)
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


def test_search_exposes_retrieval_abstention_diagnostics(monkeypatch) -> None:
    def fake_kb_request(method, path, body=None, timeout=60.0):
        assert method == "POST"
        assert path == "/search"
        assert body == {"query": "unrelated query", "top_k": 5}
        return {
            "results": [],
            "count": 0,
            "query": "unrelated query",
            "retrieval": {
                "abstained": True,
                "reason": "below_weak_score",
                "candidate_count": 20,
                "result_count": 0,
                "top_score": 0.51,
                "runner_up_document_score": 0.49,
                "document_margin": 0.02,
            },
        }

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.post(
        "/api/knowledge/search",
        json={"query": "unrelated query", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert payload["retrieval"]["abstained"] is True
    assert payload["retrieval"]["reason"] == "below_weak_score"
    assert payload["retrieval"]["top_score"] == 0.51


def test_set_document_status_forwards_publication_transition(monkeypatch) -> None:
    def fake_kb_request(method, path, body=None, timeout=60.0):
        assert method == "PATCH"
        assert path == "/documents/doi%3A10.1000%2Fpaper/status"
        assert body == {"review_status": "published"}
        return {
            "doc_id": "doi:10.1000/paper",
            "review_status": "published",
        }

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.patch(
        "/api/knowledge/documents/doi%3A10.1000%2Fpaper/status",
        json={"review_status": "published"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": "doi:10.1000/paper",
        "review_status": "published",
    }


def test_update_document_metadata_forwards_only_supplied_fields(monkeypatch) -> None:
    def fake_kb_request(method, path, body=None, timeout=60.0):
        assert method == "PATCH"
        assert path == "/documents/doi%3A10.1000%2Fpaper/metadata"
        assert body == {
            "title": "Updated paper",
            "authors": ["Alice", "Bob"],
            "year": 2025,
            "doi": "10.1000/paper",
            "language": "en",
            "domains": ["meat"],
            "quality_tier": "A",
        }
        return {
            "doc_id": "doi:10.1000/paper",
            **body,
            "source": "paper.pdf",
            "source_type": "document",
            "review_status": "published",
            "content_sha256": "a" * 64,
            "version": 1,
        }

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.patch(
        "/api/knowledge/documents/doi%3A10.1000%2Fpaper/metadata",
        json={
            "title": "Updated paper",
            "authors": ["Alice", "Bob"],
            "year": 2025,
            "doi": "10.1000/paper",
            "language": "en",
            "domains": ["meat"],
            "quality_tier": "A",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_id"] == "doi:10.1000/paper"
    assert payload["title"] == "Updated paper"
    assert payload["authors"] == ["Alice", "Bob"]
    assert payload["quality_tier"] == "A"
    assert payload["review_status"] == "published"
    assert "chunk_count" not in payload


def test_update_document_metadata_rejects_empty_request(monkeypatch) -> None:
    forwarded = False

    def fake_kb_request(method, path, body=None, timeout=60.0):
        nonlocal forwarded
        forwarded = True
        return {}

    client = _make_client(monkeypatch, fake_kb_request)
    response = client.patch(
        "/api/knowledge/documents/doc-1/metadata",
        json={},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No metadata fields supplied"
    assert forwarded is False
