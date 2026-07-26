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
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Supported document extensions (must match parser.SUPPORTED_EXTENSIONS)
_SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".csv"}
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB (reduces memory spike from base64 encoding)
_UPLOAD_TIMEOUT_SECONDS = 600.0  # BGE-M3 CPU embedding can exceed two minutes for large PDFs.


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
    authors: list[str] = Field(default_factory=list)
    doi: str | None = None
    source_type: str = "document"
    language: str = "und"
    domains: list[str] = Field(default_factory=list)
    quality_tier: str = "C"
    review_status: str = "draft"
    content_sha256: str = ""
    version: int = 1
    publication_readiness: dict[str, object] | None = None


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
    index_version: str = ""


class KnowledgeSearchRequest(BaseModel):
    """Request body for POST /search."""

    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    purpose: str = Field(default="answer", pattern="^(answer|decision)$")


class KnowledgeSearchResult(BaseModel):
    """A single search result."""

    content: str
    source: str
    score: float
    entities: dict
    related_entities: list[str]
    evidence_id: str = ""
    chunk_id: str = ""
    doc_id: str = ""
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    quality_tier: str = ""
    review_status: str = ""
    content_trust: str = "untrusted_evidence"


class KnowledgeRetrievalDiagnostics(BaseModel):
    """Answerability and diversity diagnostics from the retriever."""

    abstained: bool = False
    reason: str = ""
    candidate_count: int = 0
    result_count: int = 0
    top_score: float | None = None
    runner_up_document_score: float | None = None
    document_margin: float | None = None


class KnowledgeSearchResponse(BaseModel):
    """Response for POST /search."""

    results: list[KnowledgeSearchResult]
    count: int
    query: str
    retrieval: KnowledgeRetrievalDiagnostics | None = None
    evidence_assessment: dict[str, object] | None = None


class KnowledgeUploadResponse(BaseModel):
    """Response for POST /documents."""

    doc_id: str
    chunks_added: int
    title: str = ""
    methods: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    action: str = "created"
    version: int = 1
    review_status: str = "draft"
    content_sha256: str = ""


class KnowledgeStatusRequest(BaseModel):
    review_status: str = Field(pattern="^(draft|needs_review|published|retired)$")


class KnowledgeStatusResponse(BaseModel):
    doc_id: str
    review_status: str


class KnowledgeMetadataRequest(BaseModel):
    title: str | None = Field(default=None, max_length=500)
    authors: list[str] | None = None
    year: int | None = Field(default=None, ge=1000, le=2100)
    doi: str | None = Field(default=None, max_length=300)
    source_type: str | None = Field(default=None, min_length=1, max_length=100)
    language: str | None = Field(default=None, min_length=1, max_length=50)
    domains: list[str] | None = None
    quality_tier: str | None = Field(default=None, pattern="^[A-E]$")

    @field_validator("doi")
    @classmethod
    def _validate_doi(cls, value: str | None) -> str | None:
        from nir_core.knowledge.governance import require_valid_doi

        return require_valid_doi(value)


class KnowledgeMetadataResponse(BaseModel):
    doc_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source_type: str = "document"
    language: str = "und"
    domains: list[str] = Field(default_factory=list)
    quality_tier: str = "C"
    review_status: str = "draft"


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
    authors: list[str] | None = None,
    doi: str | None = None,
    source_type: str | None = None,
    domains: list[str] | None = None,
    quality_tier: str = "C",
    review_status: str = "draft",
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
    if authors:
        body["authors"] = authors
    if doi:
        from nir_core.knowledge.governance import require_valid_doi

        try:
            body["doi"] = require_valid_doi(doi)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if source_type:
        body["source_type"] = source_type
    if domains:
        body["domains"] = domains
    body["quality_tier"] = quality_tier
    body["review_status"] = review_status
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
        index_version=data.get("index_version", ""),
    )


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search(req: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """Test semantic search against the knowledge base."""
    data = await _kb_request_async(
        "POST",
        "/search",
        body={
            "query": req.query,
            "top_k": req.top_k,
            "purpose": req.purpose,
        },
    )
    return KnowledgeSearchResponse(
        results=[KnowledgeSearchResult(**r) for r in data.get("results", [])],
        count=data.get("count", 0),
        query=data.get("query", req.query),
        retrieval=data.get("retrieval"),
        evidence_assessment=data.get("evidence_assessment"),
    )


@router.post("/documents", response_model=KnowledgeUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    year: int | None = Form(default=None),
    authors: list[str] | None = Form(default=None),
    doi: str | None = Form(default=None),
    source_type: str | None = Form(default=None),
    domains: list[str] | None = Form(default=None),
    quality_tier: str = Form(default="C"),
    review_status: str = Form(default="draft"),
) -> KnowledgeUploadResponse:
    """Upload a document to the knowledge base.

    The file is base64-encoded and forwarded to the knowledge base HTTP
    server, which runs the parser → chunker → entity_extractor → vectorstore
    pipeline.
    """
    _, body = await _build_document_body(
        file,
        title=title,
        year=year,
        authors=authors,
        doi=doi,
        source_type=source_type,
        domains=domains,
        quality_tier=quality_tier,
        review_status=review_status,
    )

    data = await _kb_request_async(
        "POST",
        "/documents",
        body=body,
        timeout=_UPLOAD_TIMEOUT_SECONDS,
    )

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
        action=data.get("action", "created"),
        version=data.get("version", 1),
        review_status=data.get("review_status", review_status),
        content_sha256=data.get("content_sha256", ""),
    )


# Max files per batch upload (prevents excessive memory usage)
_MAX_BATCH_FILES = 20


@router.post("/documents/batch", response_model=KnowledgeBatchUploadResponse)
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
    title: str | None = Form(default=None),
    year: int | None = Form(default=None),
    quality_tier: str = Form(default="C"),
    review_status: str = Form(default="draft"),
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
            filename, body = await _build_document_body(
                file,
                title=title,
                year=year,
                quality_tier=quality_tier,
                review_status=review_status,
            )

            data = await _kb_request_async(
                "POST",
                "/documents",
                body=body,
                timeout=_UPLOAD_TIMEOUT_SECONDS,
            )

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


@router.patch(
    "/documents/{doc_id:path}/status",
    response_model=KnowledgeStatusResponse,
)
async def set_document_status(doc_id: str, request: KnowledgeStatusRequest) -> KnowledgeStatusResponse:
    """Publish, retire, or return a document to review without re-indexing it."""
    import urllib.parse

    encoded = urllib.parse.quote(doc_id, safe="")
    data = await _kb_request_async(
        "PATCH",
        f"/documents/{encoded}/status",
        body={"review_status": request.review_status},
    )
    if data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])
    return KnowledgeStatusResponse(
        doc_id=data.get("doc_id", doc_id),
        review_status=data.get("review_status", request.review_status),
    )


@router.patch(
    "/documents/{doc_id:path}/metadata",
    response_model=KnowledgeMetadataResponse,
)
async def update_document_metadata(
    doc_id: str,
    request: KnowledgeMetadataRequest,
) -> KnowledgeMetadataResponse:
    """Update descriptive metadata in the catalog and existing vector chunks."""
    import urllib.parse

    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No metadata fields supplied")

    encoded = urllib.parse.quote(doc_id, safe="")
    data = await _kb_request_async(
        "PATCH",
        f"/documents/{encoded}/metadata",
        body=updates,
    )
    if data.get("error"):
        raise HTTPException(status_code=400, detail=data["error"])
    return KnowledgeMetadataResponse(**data)


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
