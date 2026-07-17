"""Knowledge base management router.

Exposes CRUD endpoints for the NIR knowledge base. All operations are
proxied to the lightweight HTTP search server running on the host
(``_search_server.py`` on port 8089) because the Docker container cannot
import chromadb / sentence-transformers directly.

Endpoints:
    GET    /api/knowledge/documents           List all documents
    POST   /api/knowledge/documents           Upload a document (multipart)
    POST   /api/knowledge/documents/batch     Upload multiple documents
    DELETE /api/knowledge/documents/{doc_id}   Delete a document
    GET    /api/knowledge/stats               Knowledge base statistics
    POST   /api/knowledge/search              Test semantic search
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Supported document extensions (must match parser.SUPPORTED_EXTENSIONS)
_SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB (reduces memory spike from base64 encoding)


def _kb_base_url() -> str:
    """Resolve the knowledge base HTTP server URL.

    Priority:
    1. ``NIR_KNOWLEDGE_URL`` env var
    2. ``http://host.docker.internal:8089`` (Docker -> host)
    3. ``http://localhost:8089`` (local dev)
    """
    url = os.environ.get("NIR_KNOWLEDGE_URL", "").rstrip("/")
    if url:
        return url
    return "http://host.docker.internal:8089"


def _kb_headers() -> dict[str, str]:
    """Build headers expected by the host knowledge server."""
    headers = {"Content-Type": "application/json; charset=utf-8"}
    token = os.environ.get("NIR_KNOWLEDGE_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _kb_request(method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    """Send a JSON request to the knowledge base server and return the response.

    Raises ``HTTPException`` with a friendly message on any failure.
    """
    url = _kb_base_url() + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers=_kb_headers(),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Server responded with an HTTP error status
        try:
            err_body = json.loads(exc.read().decode("utf-8"))
            err_msg = err_body.get("error", str(err_body))
        except (ValueError, TypeError):
            err_msg = f"HTTP {exc.code} {exc.reason}"
        raise HTTPException(status_code=exc.code if exc.code >= 400 else 500, detail=err_msg) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=(f"Knowledge base server unreachable at {url}: {exc.reason}. Start it with: python nir_core/knowledge/_search_server.py"),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


async def _kb_request_async(method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    """Run the blocking KB HTTP request outside the event loop."""
    return await asyncio.to_thread(_kb_request, method, path, body, timeout)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class KnowledgeDocument(BaseModel):
    """A single document in the knowledge base."""

    doc_id: str
    title: str = ""
    source: str = ""
    year: int | None = None
    chunk_count: int = 0


class KnowledgeDocumentsResponse(BaseModel):
    """Response for GET /documents."""

    documents: list[KnowledgeDocument]
    count: int


class KnowledgeStatsResponse(BaseModel):
    """Response for GET /stats."""

    documents: int
    total_chunks: int
    db_path: str
    embedding_model: str
    collection_name: str


class KnowledgeSearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResult(BaseModel):
    """A single search result."""

    content: str
    source: str
    score: float
    entities: dict
    related_entities: list[str]


class KnowledgeSearchResponse(BaseModel):
    """Response for POST /search."""

    results: list[KnowledgeSearchResult]
    count: int
    query: str


class KnowledgeUploadResponse(BaseModel):
    """Response for POST /documents."""

    doc_id: str
    chunks_added: int
    title: str = ""
    methods: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class KnowledgeDeleteResponse(BaseModel):
    """Response for DELETE /documents/{doc_id}."""

    deleted: bool
    doc_id: str


class KnowledgeBatchUploadItem(BaseModel):
    """Result for a single file in a batch upload."""

    filename: str
    success: bool
    doc_id: str = ""
    chunks_added: int = 0
    error: str = ""


class KnowledgeBatchUploadResponse(BaseModel):
    """Response for POST /documents/batch."""

    results: list[KnowledgeBatchUploadItem]
    total: int
    succeeded: int
    failed: int
    total_chunks_added: int


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------


async def _build_document_body(
    file: UploadFile,
    *,
    title: str | None = None,
    year: int | None = None,
) -> tuple[str, dict[str, object]]:
    """Validate an uploaded file and build the JSON body for the KB server."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(_SUPPORTED_EXTS))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(raw)} bytes (max {_MAX_FILE_SIZE})",
        )

    body: dict[str, object] = {
        "filename": file.filename,
        "content_b64": base64.b64encode(raw).decode("ascii"),
    }
    if title:
        body["title"] = title
    if year is not None:
        body["year"] = year
    return file.filename, body


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=KnowledgeDocumentsResponse)
async def list_documents() -> KnowledgeDocumentsResponse:
    """List all documents in the knowledge base."""
    data = await _kb_request_async("GET", "/documents")
    docs = [KnowledgeDocument(**d) for d in data.get("documents", [])]
    return KnowledgeDocumentsResponse(documents=docs, count=data.get("count", len(docs)))


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_stats() -> KnowledgeStatsResponse:
    """Return knowledge base statistics."""
    data = await _kb_request_async("GET", "/stats")
    return KnowledgeStatsResponse(
        documents=data.get("documents", 0),
        total_chunks=data.get("total_chunks", 0),
        db_path=data.get("db_path", ""),
        embedding_model=data.get("embedding_model", ""),
        collection_name=data.get("collection_name", ""),
    )


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search(req: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """Test semantic search against the knowledge base."""
    data = await _kb_request_async("POST", "/search", body={"query": req.query, "top_k": req.top_k})
    return KnowledgeSearchResponse(
        results=[KnowledgeSearchResult(**r) for r in data.get("results", [])],
        count=data.get("count", 0),
        query=data.get("query", req.query),
    )


@router.post("/documents", response_model=KnowledgeUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    year: int | None = Form(default=None),
) -> KnowledgeUploadResponse:
    """Upload a document to the knowledge base.

    The file is base64-encoded and forwarded to the knowledge base HTTP
    server, which runs the parser → chunker → entity_extractor → vectorstore
    pipeline.
    """
    _, body = await _build_document_body(file, title=title, year=year)

    data = await _kb_request_async("POST", "/documents", body=body, timeout=120.0)

    if data.get("error"):
        raise HTTPException(status_code=500, detail=data["error"])

    return KnowledgeUploadResponse(
        doc_id=data.get("doc_id", ""),
        chunks_added=data.get("chunks_added", 0),
        title=data.get("title", ""),
        methods=data.get("methods", []),
        models=data.get("models", []),
        datasets=data.get("datasets", []),
        metrics=data.get("metrics", []),
    )


# Max files per batch upload (prevents excessive memory usage)
_MAX_BATCH_FILES = 20


@router.post("/documents/batch", response_model=KnowledgeBatchUploadResponse)
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    title: str | None = Form(default=None),
    year: int | None = Form(default=None),
) -> KnowledgeBatchUploadResponse:
    """Upload multiple documents to the knowledge base in one request.

    Each file is processed sequentially through the same validation and
    knowledge-base ingestion path as the single-file endpoint.
    Returns per-file results so the frontend can show partial successes.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > _MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files: {len(files)} (max {_MAX_BATCH_FILES})",
        )

    results: list[KnowledgeBatchUploadItem] = []
    total_chunks = 0
    succeeded = 0

    for file in files:
        filename = file.filename or "unknown"
        try:
            filename, body = await _build_document_body(file, title=title, year=year)

            data = await _kb_request_async("POST", "/documents", body=body, timeout=120.0)

            if data.get("error"):
                results.append(
                    KnowledgeBatchUploadItem(
                        filename=filename,
                        success=False,
                        error=data["error"],
                    )
                )
                continue

            chunks_added = data.get("chunks_added", 0)
            total_chunks += chunks_added
            succeeded += 1
            results.append(
                KnowledgeBatchUploadItem(
                    filename=filename,
                    success=True,
                    doc_id=data.get("doc_id", ""),
                    chunks_added=chunks_added,
                )
            )
        except HTTPException as exc:
            results.append(
                KnowledgeBatchUploadItem(
                    filename=filename,
                    success=False,
                    error=str(exc.detail),
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                KnowledgeBatchUploadItem(
                    filename=filename,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return KnowledgeBatchUploadResponse(
        results=results,
        total=len(files),
        succeeded=succeeded,
        failed=len(files) - succeeded,
        total_chunks_added=total_chunks,
    )


@router.delete("/documents/{doc_id}", response_model=KnowledgeDeleteResponse)
async def delete_document(doc_id: str) -> KnowledgeDeleteResponse:
    """Delete a document and all its chunks from the knowledge base."""
    # URL-encode the doc_id to handle any special characters
    import urllib.parse

    encoded = urllib.parse.quote(doc_id, safe="")
    data = await _kb_request_async("DELETE", f"/documents/{encoded}")

    if data.get("error"):
        raise HTTPException(status_code=500, detail=data["error"])

    return KnowledgeDeleteResponse(
        deleted=data.get("deleted", False),
        doc_id=data.get("doc_id", doc_id),
    )
