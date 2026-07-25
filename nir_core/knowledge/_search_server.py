"""Lightweight HTTP server for knowledge base search and management.

Runs on the host machine using Anaconda Python (which has chromadb +
sentence-transformers installed). The Docker container's
``nir_search_knowledge`` tool and the backend ``knowledge`` router both call
this server via ``http://host.docker.internal:8089``.

Usage:
    python _search_server.py [--port 8089] [--host 0.0.0.0]

Endpoints:
    GET  /health                  -> {"status": "ok"}
    GET  /documents                -> {"documents": [...], "count": N}
    GET  /stats                    -> {"documents": N, "total_chunks": M, ...}
    POST /search                   -> {"results": [...], "count": N}
         Body: {"query": "...", "top_k": 5}
    POST /documents                -> {"doc_id": "...", "chunks_added": N}
         Body: {"filename": "paper.pdf", "content_b64": "...",
                "title": "...", "year": 2023}
    PATCH /documents/{doc_id}/metadata -> updated descriptive metadata
    DELETE /documents/{doc_id}     -> {"deleted": true, "doc_id": "..."}
"""

import base64
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote, urlparse

# Set env vars BEFORE any heavy imports
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nir_core.knowledge.config import get_config
from nir_core.knowledge.governance import DocumentRecord, KnowledgeCatalog
from nir_core.knowledge.ingestion import ingest_document_bytes
from nir_core.knowledge.vectorstore import ChromaDBRetriever

# --- Initialize retriever (loaded once at startup) ---
_cfg = get_config()
_retriever = ChromaDBRetriever(
    db_path=_cfg.chroma_path,
    embedding_model=_cfg.embedding_model,
    collection_name=_cfg.collection_name,
)
_catalog = KnowledgeCatalog(_cfg.catalog_path)

# 20 MB upload cap (matches backend upload limit)
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024

# Max JSON body size for POST endpoints (60 MB — allows base64 overhead)
_MAX_BODY_BYTES = 60 * 1024 * 1024

# Simple token authentication: if NIR_KNOWLEDGE_TOKEN env var is set,
# all requests must include "Authorization: Bearer <token>" header.
_AUTH_TOKEN = os.environ.get("NIR_KNOWLEDGE_TOKEN", "")

print(f"[knowledge-server] ChromaDB: {_cfg.chroma_path}")
print(f"[knowledge-server] Model: {_cfg.embedding_model}")
print(f"[knowledge-server] Index: {_cfg.index_version}")
print("[knowledge-server] Retriever initialized")
if _AUTH_TOKEN:
    print("[knowledge-server] Auth: enabled (token from NIR_KNOWLEDGE_TOKEN)")
else:
    print("[knowledge-server] Auth: disabled (set NIR_KNOWLEDGE_TOKEN to enable)")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _do_search(query: str, top_k: int = 5) -> dict:
    """Search the knowledge base and return JSON-serializable results."""
    decision = _retriever.search_with_diagnostics(query, top_k=top_k)
    results = list(decision.results)
    retrieval = decision.diagnostics()
    if not results:
        return {
            "results": [],
            "count": 0,
            "query": query,
            "retrieval": retrieval,
            "error": None,
        }

    output = []
    for r in results:
        meta = r.chunk.metadata or {}

        def _load(key: str) -> list:
            val = meta.get(key, "")
            if isinstance(val, list):
                return val
            if isinstance(val, str) and val:
                try:
                    decoded = json.loads(val)
                    return decoded if isinstance(decoded, list) else [val]
                except (ValueError, TypeError):
                    return [val]
            return []

        output.append(
            {
                "evidence_id": r.chunk.id,
                "chunk_id": r.chunk.id,
                "doc_id": meta.get("doc_id", ""),
                "content": r.chunk.content[:1000],
                "source": r.chunk.source,
                "score": round(r.score, 4),
                "title": meta.get("title", ""),
                "authors": _load("authors"),
                "year": meta.get("year"),
                "doi": meta.get("doi", ""),
                "section_path": _load("section_path"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "quality_tier": meta.get("quality_tier", ""),
                "review_status": meta.get("review_status", ""),
                "content_trust": meta.get("content_trust", "untrusted_evidence"),
                "entities": {
                    "methods": _load("methods"),
                    "models": _load("models"),
                    "datasets": _load("datasets"),
                    "metrics": _load("metrics"),
                },
                "related_entities": r.related_entities,
            }
        )

    return {
        "results": output,
        "count": len(output),
        "query": query,
        "retrieval": retrieval,
        "error": None,
    }


def _do_list_documents() -> dict:
    """List all documents in the knowledge base."""
    chunk_counts = {
        document.get("doc_id", ""): document.get("chunk_count", 0)
        for document in _retriever.list_documents()
    }
    out = [
        {
            **record.model_dump(),
            "chunk_count": chunk_counts.get(record.doc_id, 0),
        }
        for record in _catalog.list()
    ]
    return {"documents": out, "count": len(out), "error": None}


def _do_stats() -> dict:
    """Return knowledge base statistics."""
    vector_docs = _retriever.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in vector_docs)
    return {
        "documents": len(_catalog.list()),
        "total_chunks": total_chunks,
        "db_path": _cfg.chroma_path,
        "embedding_model": _cfg.embedding_model,
        "collection_name": _cfg.collection_name,
        "index_version": _cfg.index_version,
        "error": None,
    }


def _do_add_document(
    filename: str,
    content_b64: str,
    title: str | None = None,
    year: int | None = None,
    authors: list[str] | None = None,
    doi: str | None = None,
    source_type: str | None = None,
    domains: list[str] | None = None,
    quality_tier: str = "C",
    review_status: str = "draft",
) -> dict:
    """Parse, chunk, extract entities, and add a document to the KB.

    ``content_b64`` is base64-encoded file bytes (so we can ship binary
    PDFs over JSON without multipart parsing headaches).
    """
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Invalid base64 content: {exc}",
            "doc_id": None,
            "chunks_added": 0,
        }

    if len(raw) > _MAX_UPLOAD_BYTES:
        return {
            "error": f"File too large: {len(raw)} bytes (max {_MAX_UPLOAD_BYTES})",
            "doc_id": None,
            "chunks_added": 0,
        }

    try:
        result = ingest_document_bytes(
            filename=filename,
            content=raw,
            retriever=_retriever,
            catalog=_catalog,
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            source_type=source_type,
            domains=domains,
            quality_tier=quality_tier,
            review_status=review_status,
            chunk_strategy=_cfg.chunk_strategy,
            max_tokens=_cfg.chunk_max_tokens,
            overlap_percent=_cfg.chunk_overlap,
            index_version=_cfg.index_version,
        )
        return {
            **result.record.model_dump(),
            "action": result.action,
            "chunks_added": result.chunks_added,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "doc_id": None,
            "chunks_added": 0,
        }


def _do_delete_document(doc_id: str) -> dict:
    """Delete all chunks belonging to ``doc_id``."""
    vector_deleted = _retriever.delete_document(doc_id)
    catalog_deleted = _catalog.delete(doc_id)
    success = vector_deleted or catalog_deleted
    return {
        "deleted": success,
        "doc_id": doc_id,
        "error": None if success else "delete failed",
    }


def _do_set_review_status(doc_id: str, review_status: str) -> dict:
    if review_status not in {"draft", "needs_review", "published", "retired"}:
        return {"error": f"Invalid review_status: {review_status}"}
    record = _catalog.get(doc_id)
    if record is None:
        return {"error": "Document not found", "doc_id": doc_id}
    if not _retriever.update_document_metadata(
        doc_id, {"review_status": review_status}
    ):
        return {"error": "Vector chunks not found", "doc_id": doc_id}
    updated = _catalog.set_review_status(doc_id, review_status)
    return {"error": None, **updated.model_dump()}


_EDITABLE_METADATA_FIELDS = {
    "title",
    "authors",
    "year",
    "doi",
    "language",
    "domains",
    "quality_tier",
}


def _metadata_for_vectors(record: DocumentRecord) -> dict:
    return {
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "doi": record.doi,
        "language": record.language,
        "domains": record.domains,
        "quality_tier": record.quality_tier,
    }


def _do_update_document_metadata(doc_id: str, updates: dict) -> dict:
    """Update catalog and chunk metadata without re-embedding content."""
    unsupported = sorted(set(updates) - _EDITABLE_METADATA_FIELDS)
    if unsupported:
        return {
            "error": f"Unsupported metadata fields: {', '.join(unsupported)}",
            "doc_id": doc_id,
        }
    if not updates:
        return {"error": "No metadata fields supplied", "doc_id": doc_id}

    existing = _catalog.get(doc_id)
    if existing is None:
        return {"error": "Document not found", "doc_id": doc_id}

    try:
        candidate = DocumentRecord.model_validate({**existing.model_dump(), **updates})
    except ValueError as exc:
        return {"error": str(exc), "doc_id": doc_id}

    if not _retriever.update_document_metadata(
        doc_id, _metadata_for_vectors(candidate)
    ):
        return {"error": "Vector chunks not found", "doc_id": doc_id}

    try:
        updated = _catalog.update_metadata(doc_id, updates)
    except ValueError as exc:
        _retriever.update_document_metadata(doc_id, _metadata_for_vectors(existing))
        return {"error": str(exc), "doc_id": doc_id}
    if updated is None:
        _retriever.update_document_metadata(doc_id, _metadata_for_vectors(existing))
        return {"error": "Document not found", "doc_id": doc_id}
    return {"error": None, **updated.model_dump()}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        """Return True if auth passes (or is disabled)."""
        if not _AUTH_TOKEN:
            return True
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:] == _AUTH_TOKEN
        return False

    def _read_json_body(self) -> dict | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        if content_length > _MAX_BODY_BYTES:
            return None  # caller will see None and return 400
        body = self.rfile.read(content_length)
        try:
            return json.loads(body)
        except (ValueError, TypeError):
            return None

    def do_GET(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._send_json(200, {"status": "ok", "service": "nir-knowledge"})
            return
        if path == "/documents":
            self._send_json(200, _do_list_documents())
            return
        if path == "/stats":
            self._send_json(200, _do_stats())
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/search":
            req = self._read_json_body()
            if req is None:
                self._send_json(400, {"error": "Invalid JSON body or body too large"})
                return
            query = req.get("query", "")
            top_k = int(req.get("top_k", 5))
            if not query:
                self._send_json(400, {"error": "Missing 'query' field"})
                return
            try:
                self._send_json(200, _do_search(query, top_k))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "results": [],
                        "count": 0,
                        "query": query,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            return

        if path == "/documents":
            req = self._read_json_body()
            if req is None:
                self._send_json(400, {"error": "Invalid JSON body or body too large"})
                return
            filename = req.get("filename", "")
            content_b64 = req.get("content_b64", "")
            if not filename or not content_b64:
                self._send_json(
                    400, {"error": "Missing 'filename' or 'content_b64' field"}
                )
                return
            title = req.get("title")
            year = req.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            try:
                self._send_json(
                    200,
                    _do_add_document(
                        filename,
                        content_b64,
                        title,
                        year,
                        authors=req.get("authors"),
                        doi=req.get("doi"),
                        source_type=req.get("source_type"),
                        domains=req.get("domains"),
                        quality_tier=req.get("quality_tier", "C"),
                        review_status=req.get("review_status", "draft"),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "doc_id": None,
                        "chunks_added": 0,
                    },
                )
            return

        self._send_json(404, {"error": "Not found"})

    def do_PATCH(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        path = urlparse(self.path).path
        if path.startswith("/documents/") and path.endswith("/status"):
            encoded_doc_id = path[len("/documents/") : -len("/status")].rstrip("/")
            doc_id = unquote(encoded_doc_id)
            req = self._read_json_body()
            if not doc_id or req is None or not req.get("review_status"):
                self._send_json(400, {"error": "Missing doc_id or review_status"})
                return
            result = _do_set_review_status(doc_id, req["review_status"])
            self._send_json(400 if result.get("error") else 200, result)
            return
        if path.startswith("/documents/") and path.endswith("/metadata"):
            encoded_doc_id = path[len("/documents/") : -len("/metadata")].rstrip("/")
            doc_id = unquote(encoded_doc_id)
            req = self._read_json_body()
            if not doc_id or req is None:
                self._send_json(400, {"error": "Missing doc_id or metadata body"})
                return
            result = _do_update_document_metadata(doc_id, req)
            self._send_json(400 if result.get("error") else 200, result)
            return
        self._send_json(404, {"error": "Not found"})

    def do_DELETE(self) -> None:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized"})
            return
        parsed = urlparse(self.path)
        path = parsed.path

        # /documents/{doc_id}
        if path.startswith("/documents/"):
            doc_id = unquote(path[len("/documents/") :])
            if not doc_id:
                self._send_json(400, {"error": "Missing doc_id"})
                return
            try:
                self._send_json(200, _do_delete_document(doc_id))
            except Exception as exc:  # noqa: BLE001
                self._send_json(
                    500,
                    {
                        "deleted": False,
                        "doc_id": doc_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.command}] {self.path} - {args[0] if args else ''}")


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8089
    host = "0.0.0.0"  # bind all interfaces so Docker can reach it
    server = HTTPServer((host, port), _Handler)
    print(f"[knowledge-server] Listening on http://{host}:{port}")
    print("[knowledge-server] Endpoints: GET /health, GET /documents, GET /stats,")
    print(
        "[knowledge-server]             POST /search, POST /documents, DELETE /documents/{doc_id}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[knowledge-server] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
