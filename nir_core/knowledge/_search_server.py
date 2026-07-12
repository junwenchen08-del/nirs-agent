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
    DELETE /documents/{doc_id}     -> {"deleted": true, "doc_id": "..."}
"""

import base64
import json
import os
import sys
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, unquote

# Set env vars BEFORE any heavy imports
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nir_core.knowledge.chunker import chunk_document
from nir_core.knowledge.config import get_config
from nir_core.knowledge.entity_extractor import extract_entities
from nir_core.knowledge.parser import parse_document
from nir_core.knowledge.vectorstore import ChromaDBRetriever

# --- Initialize retriever (loaded once at startup) ---
_cfg = get_config()
_model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
_retriever = ChromaDBRetriever(
    db_path=_cfg.chroma_path,
    embedding_model=_model_path,
    collection_name=_cfg.collection_name,
)

# 50 MB upload cap (matches backend upload limit)
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

print(f"[knowledge-server] ChromaDB: {_cfg.chroma_path}")
print(f"[knowledge-server] Model: {_model_path}")
print(f"[knowledge-server] Retriever initialized")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _do_search(query: str, top_k: int = 5) -> dict:
    """Search the knowledge base and return JSON-serializable results."""
    results = _retriever.search(query, top_k=top_k)
    if not results:
        return {"results": [], "count": 0, "query": query, "error": None}

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

        output.append({
            "content": r.chunk.content[:1000],
            "source": r.chunk.source,
            "score": round(r.score, 4),
            "entities": {
                "methods": _load("methods"),
                "models": _load("models"),
                "datasets": _load("datasets"),
                "metrics": _load("metrics"),
            },
            "related_entities": r.related_entities,
        })

    return {"results": output, "count": len(output), "query": query, "error": None}


def _do_list_documents() -> dict:
    """List all documents in the knowledge base."""
    docs = _retriever.list_documents()
    # Normalise keys for JSON output
    out = []
    for d in docs:
        out.append({
            "doc_id": d.get("doc_id", ""),
            "title": d.get("title", "") or d.get("doc_id", ""),
            "source": d.get("source", ""),
            "year": d.get("year"),
            "chunk_count": d.get("chunk_count", 0),
        })
    return {"documents": out, "count": len(out), "error": None}


def _do_stats() -> dict:
    """Return knowledge base statistics."""
    docs = _retriever.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    return {
        "documents": len(docs),
        "total_chunks": total_chunks,
        "db_path": _cfg.chroma_path,
        "embedding_model": _cfg.embedding_model,
        "collection_name": _cfg.collection_name,
        "error": None,
    }


def _do_add_document(
    filename: str,
    content_b64: str,
    title: str | None = None,
    year: int | None = None,
) -> dict:
    """Parse, chunk, extract entities, and add a document to the KB.

    ``content_b64`` is base64-encoded file bytes (so we can ship binary
    PDFs over JSON without multipart parsing headaches).
    """
    try:
        raw = base64.b64decode(content_b64, validate=True)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Invalid base64 content: {exc}", "doc_id": None, "chunks_added": 0}

    if len(raw) > _MAX_UPLOAD_BYTES:
        return {
            "error": f"File too large: {len(raw)} bytes (max {_MAX_UPLOAD_BYTES})",
            "doc_id": None,
            "chunks_added": 0,
        }

    doc_id = Path(filename).stem
    suffix = Path(filename).suffix.lower()

    # Write to temp file so parser can sniff the format by extension.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        markdown = parse_document(tmp_path)
        chunks = chunk_document(
            markdown,
            source=filename,
            strategy=_cfg.chunk_strategy,
            max_tokens=_cfg.chunk_max_tokens,
            doc_id=doc_id,
        )
        entities = extract_entities(markdown)
        for chunk in chunks:
            chunk.metadata.update({
                "title": title or doc_id,
                "year": year,
                **entities,
            })
        count = _retriever.add_documents(chunks)
        return {
            "doc_id": doc_id,
            "chunks_added": count,
            "title": title or doc_id,
            "methods": entities.get("methods", []),
            "models": entities.get("models", []),
            "datasets": entities.get("datasets", []),
            "metrics": entities.get("metrics", []),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "doc_id": doc_id,
            "chunks_added": 0,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _do_delete_document(doc_id: str) -> dict:
    """Delete all chunks belonging to ``doc_id``."""
    success = _retriever.delete_document(doc_id)
    return {"deleted": success, "doc_id": doc_id, "error": None if success else "delete failed"}


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

    def _read_json_body(self) -> dict | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return None
        body = self.rfile.read(content_length)
        try:
            return json.loads(body)
        except (ValueError, TypeError):
            return None

    def do_GET(self) -> None:
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
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/search":
            req = self._read_json_body()
            if req is None:
                self._send_json(400, {"error": "Invalid JSON body"})
                return
            query = req.get("query", "")
            top_k = int(req.get("top_k", 5))
            if not query:
                self._send_json(400, {"error": "Missing 'query' field"})
                return
            try:
                self._send_json(200, _do_search(query, top_k))
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {
                    "results": [], "count": 0, "query": query,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            return

        if path == "/documents":
            req = self._read_json_body()
            if req is None:
                self._send_json(400, {"error": "Invalid JSON body"})
                return
            filename = req.get("filename", "")
            content_b64 = req.get("content_b64", "")
            if not filename or not content_b64:
                self._send_json(400, {"error": "Missing 'filename' or 'content_b64' field"})
                return
            title = req.get("title")
            year = req.get("year")
            if isinstance(year, str) and year.isdigit():
                year = int(year)
            try:
                self._send_json(200, _do_add_document(filename, content_b64, title, year))
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {
                    "error": f"{type(exc).__name__}: {exc}",
                    "doc_id": None, "chunks_added": 0,
                })
            return

        self._send_json(404, {"error": "Not found"})

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # /documents/{doc_id}
        if path.startswith("/documents/"):
            doc_id = unquote(path[len("/documents/"):])
            if not doc_id:
                self._send_json(400, {"error": "Missing doc_id"})
                return
            try:
                self._send_json(200, _do_delete_document(doc_id))
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {
                    "deleted": False, "doc_id": doc_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            return

        self._send_json(404, {"error": "Not found"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.command}] {self.path} - {args[0] if args else ''}")


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8089
    host = "0.0.0.0"  # bind all interfaces so Docker can reach it
    server = HTTPServer((host, port), _Handler)
    print(f"[knowledge-server] Listening on http://{host}:{port}")
    print(f"[knowledge-server] Endpoints: GET /health, GET /documents, GET /stats,")
    print(f"[knowledge-server]             POST /search, POST /documents, DELETE /documents/{{doc_id}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[knowledge-server] Shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()
