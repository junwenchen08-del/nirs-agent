"""ChromaDB-backed retriever implementation.

Design notes:
- Local persistent storage (``.chromadb/``), zero external services.
- HNSW cosine index — 10K chunks query in <10ms.
- ``sentence-transformers`` local embedding (all-MiniLM-L6-v2, 384 dim).
- Metadata filtering via ChromaDB ``where`` clauses.
- ``get_related_entities`` always returns ``[]`` today — the hook exists so
  the future ``GraphEnhancedRetriever`` can populate it without touching
  the tool layer.
"""

from __future__ import annotations

import logging
import os
from typing import Any

# Prevent transformers from importing tensorflow, which may be broken
# due to numpy/h5py binary incompatibility. Must be set BEFORE
# sentence_transformers is imported anywhere.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
# If the embedding_model is a local directory, force offline mode to
# prevent HuggingFace network calls (which fail in CN without a mirror).
if os.path.isdir(
    os.path.join(os.path.dirname(__file__), "all-MiniLM-L6-v2")
):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from nir_core.knowledge.base import Chunk, SearchResult

logger = logging.getLogger(__name__)


class ChromaDBRetriever:
    """ChromaDB vector retriever implementing ``KnowledgeRetriever``."""

    def __init__(
        self,
        db_path: str = "nir_core/knowledge/.chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "nir_papers",
    ) -> None:
        # Lazy imports so importing this module doesn't drag in the heavy
        # chromadb / sentence-transformers stack unless actually used.
        import chromadb
        from sentence_transformers import SentenceTransformer

        os.makedirs(db_path, exist_ok=True)
        self._model = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=db_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Semantic search. Returns up to ``top_k`` results sorted by score."""
        if self._collection.count() == 0:
            return []

        query_emb = self._model.encode([query])
        # sentence-transformers may return numpy array; coerce to list.
        emb_list = (
            query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        )
        if isinstance(emb_list[0], (list, tuple)):
            emb_list = emb_list[0]

        results = self._collection.query(
            query_embeddings=[emb_list],
            n_results=top_k,
            where=where,
        )

        out: list[SearchResult] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            distance = distances[i] if i < len(distances) else 1.0
            # cosine distance ∈ [0, 2]; similarity = 1 - distance (clamped)
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            meta = metadatas[i] if i < len(metadatas) else {}
            out.append(
                SearchResult(
                    chunk=Chunk(
                        id=ids[i],
                        content=documents[i] if i < len(documents) else "",
                        source=meta.get("source", ""),
                        chunk_index=meta.get("chunk_index", 0),
                        metadata=dict(meta),
                    ),
                    score=score,
                    related_entities=[],
                )
            )
        return out

    # ------------------------------------------------------------------
    # graph hook (no-op today)
    # ------------------------------------------------------------------

    def get_related_entities(
        self,
        entity: str,
        relation: str | None = None,
    ) -> list[str]:
        """No-op. Returns ``[]``. Future ``GraphEnhancedRetriever`` fills this."""
        return []

    # ------------------------------------------------------------------
    # mutations
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list[Chunk]) -> int:
        """Embed and add chunks. Returns number added."""
        if not chunks:
            return 0

        # Batch encode for efficiency.
        contents = [c.content for c in chunks]
        embeddings = self._model.encode(contents)
        embeddings_list = (
            embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
        )

        ids = [c.id for c in chunks]
        metadatas = [
            {**c.metadata, "source": c.source, "chunk_index": c.chunk_index}
            for c in chunks
        ]

        # ChromaDB requires metadata values to be primitives (str/int/float/
        # bool/None). Lists / dicts must be JSON-serialised.
        clean_metadatas = [_sanitise_metadata(m) for m in metadatas]

        self._collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=contents,
            metadatas=clean_metadatas,
        )
        return len(chunks)

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks whose ``doc_id`` matches."""
        try:
            self._collection.delete(where={"doc_id": doc_id})
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("delete_document failed for %s: %s", doc_id, exc)
            return False

    def list_documents(self) -> list[dict[str, Any]]:
        """Return one entry per distinct ``doc_id`` in the collection."""
        if self._collection.count() == 0:
            return []

        all_data = self._collection.get(include=["metadatas"])
        docs: dict[str, dict[str, Any]] = {}
        for meta in all_data.get("metadatas", []):
            doc_id = meta.get("doc_id", "")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                    "year": meta.get("year"),
                    "chunk_count": 1,
                }
            else:
                docs[doc_id]["chunk_count"] += 1
        return list(docs.values())


def _sanitise_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerce metadata values to ChromaDB-compatible primitives.

    ``None`` values are dropped (ChromaDB does not accept null metadata).
    Lists / dicts are JSON-serialised to strings; primitives passed through.
    """
    import json

    clean: dict[str, Any] = {}
    for k, v in meta.items():
        if v is None:
            continue  # ChromaDB rejects None metadata values
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            clean[k] = json.dumps(list(v), ensure_ascii=False)
        elif isinstance(v, dict):
            clean[k] = json.dumps(v, ensure_ascii=False)
        else:
            clean[k] = str(v)
    return clean
