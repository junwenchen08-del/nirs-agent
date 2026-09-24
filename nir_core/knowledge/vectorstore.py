"""ChromaDB-backed retriever implementation.

Design notes:
- Local persistent storage (``.chromadb-bge-m3/``), zero external services.
- HNSW cosine index — 10K chunks query in <10ms.
- ``sentence-transformers`` local BGE-M3 embedding (1024 dimensions).
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
from nir_core.knowledge.base import Chunk, SearchResult
from nir_core.knowledge.reranker import KnowledgeReranker
from nir_core.knowledge.retrieval_policy import (
    RetrievalDecision,
    RetrievalPolicy,
    apply_retrieval_policy,
)

logger = logging.getLogger(__name__)


def _select_rerank_candidates(
    candidates: list[SearchResult],
    *,
    limit: int,
    max_chunks_per_document: int | None,
) -> list[SearchResult]:
    """Keep the dense order while exposing more documents to the reranker."""
    selected: list[SearchResult] = []
    document_counts: dict[str, int] = {}
    for candidate in candidates:
        metadata = candidate.chunk.metadata or {}
        document_key = str(
            metadata.get("doc_id") or candidate.chunk.source or candidate.chunk.id
        )
        count = document_counts.get(document_key, 0)
        if max_chunks_per_document is not None and count >= max_chunks_per_document:
            continue
        selected.append(candidate)
        document_counts[document_key] = count + 1
        if len(selected) >= limit:
            break
    return selected


class ChromaDBRetriever:
    """ChromaDB vector retriever implementing ``KnowledgeRetriever``."""

    def __init__(
        self,
        db_path: str = "nir_core/knowledge/.chromadb-bge-m3",
        embedding_model: str = "BAAI/bge-m3",
        collection_name: str = "nir_papers_bge_m3",
        retrieval_policy: RetrievalPolicy | None = None,
        reranker: KnowledgeReranker | None = None,
        rerank_max_candidates: int = 12,
    ) -> None:
        if rerank_max_candidates < 1:
            raise ValueError("rerank_max_candidates must be at least 1")
        # Local embedding directories must not trigger HuggingFace lookups.
        if os.path.isdir(embedding_model):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
        self._retrieval_policy = retrieval_policy or RetrievalPolicy()
        self._reranker = reranker
        self._rerank_max_candidates = rerank_max_candidates

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        published_only: bool = True,
    ) -> list[SearchResult]:
        """Semantic search with answerability and document-diversity policy."""
        decision = self.search_with_diagnostics(
            query,
            top_k=top_k,
            where=where,
            published_only=published_only,
        )
        return list(decision.results)

    def search_with_diagnostics(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        published_only: bool = True,
    ) -> RetrievalDecision:
        """Search and return both selected evidence and policy diagnostics."""
        if self._collection.count() == 0:
            return apply_retrieval_policy(
                [],
                top_k=top_k,
                policy=self._retrieval_policy,
            )

        effective_where = where
        if published_only:
            publication_filter: dict[str, Any] = {"review_status": "published"}
            effective_where = (
                {"$and": [publication_filter, where]} if where else publication_filter
            )

        query_emb = self._model.encode([query])
        # sentence-transformers may return numpy array; coerce to list.
        emb_list = (
            query_emb.tolist() if hasattr(query_emb, "tolist") else list(query_emb)
        )
        if isinstance(emb_list[0], (list, tuple)):
            emb_list = emb_list[0]

        candidate_count = min(
            self._collection.count(),
            self._retrieval_policy.candidate_count(top_k),
        )
        results = self._collection.query(
            query_embeddings=[emb_list],
            n_results=candidate_count,
            where=effective_where,
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
        ranked_candidates = out
        ranking_strategy = "dense"
        rerank_error = None
        if self._reranker is not None and out:
            try:
                rerank_count = min(
                    len(out),
                    max(top_k, self._rerank_max_candidates),
                )
                rerank_pool = _select_rerank_candidates(
                    out,
                    limit=rerank_count,
                    max_chunks_per_document=(
                        self._retrieval_policy.max_chunks_per_document
                        if self._retrieval_policy.diversity_enabled
                        else None
                    ),
                )
                ranked_candidates = self._reranker.rerank(
                    query,
                    rerank_pool,
                )
                ranking_strategy = "cross_encoder"
            except Exception as exc:
                rerank_error = type(exc).__name__
                logger.exception(
                    "Knowledge reranker %s failed; using dense order",
                    self._reranker.model_name,
                )

        return apply_retrieval_policy(
            ranked_candidates,
            top_k=top_k,
            policy=self._retrieval_policy,
            ranking_strategy=ranking_strategy,
            rerank_error=rerank_error,
        )

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

    def replace_document(self, doc_id: str, chunks: list[Chunk]) -> int:
        """Delete stale chunks for ``doc_id`` and upsert its current version."""
        if not chunks:
            return 0
        if any(chunk.metadata.get("doc_id") != doc_id for chunk in chunks):
            raise ValueError("all replacement chunks must belong to doc_id")

        self._collection.delete(where={"doc_id": doc_id})
        contents = [chunk.content for chunk in chunks]
        embeddings = self._model.encode(contents)
        embeddings_list = (
            embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
        )
        metadatas = [
            _sanitise_metadata(
                {
                    **chunk.metadata,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
            )
            for chunk in chunks
        ]
        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            embeddings=embeddings_list,
            documents=contents,
            metadatas=metadatas,
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

    def update_document_metadata(self, doc_id: str, updates: dict[str, Any]) -> bool:
        """Apply primitive-safe metadata updates to all chunks of a document."""
        data = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        ids = data.get("ids", [])
        if not ids:
            return False
        metadatas = data.get("metadatas", [])
        updated = [
            _sanitise_metadata({**(metadata or {}), **updates})
            for metadata in metadatas
        ]
        self._collection.update(ids=ids, metadatas=updated)
        return True

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
                    "review_status": meta.get("review_status", ""),
                    "quality_tier": meta.get("quality_tier", ""),
                    "version": meta.get("document_version", 1),
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
