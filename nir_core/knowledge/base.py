"""Core data models and retriever protocol for the NIR knowledge base.

The ``KnowledgeRetriever`` Protocol is the single contract that the agent
tool layer depends on. The current implementation is
:class:`nir_core.knowledge.vectorstore.ChromaDBRetriever`; a future
``GraphEnhancedRetriever`` (ChromaDB + Neo4j) can be swapped in via
``config.py`` without touching the tool layer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantically meaningful block of a document.

    Attributes:
        id: Stable identifier (``{doc_id}_chunk_{i}``).
        content: The chunk text (Markdown).
        source: Source file name or path.
        chunk_index: 0-based index within the parent document.
        metadata: Arbitrary metadata (title, authors, year, entities, ...).
    """

    id: str
    content: str
    source: str
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single retrieval result.

    Attributes:
        chunk: The matched document chunk.
        score: Similarity score in ``[0, 1]`` (higher = better).
        related_entities: Entities linked to this result. Empty in the
            ChromaDB-only implementation; populated by the future
            ``GraphEnhancedRetriever`` via Neo4j.
    """

    chunk: Chunk
    score: float
    related_entities: list[str] = Field(default_factory=list)


@runtime_checkable
class KnowledgeRetriever(Protocol):
    """Abstract retrieval interface.

    The agent tool ``nir_search_knowledge`` depends only on this Protocol.
    Swapping the backend (e.g. to ``GraphEnhancedRetriever``) requires no
    changes to ``tools.py``.
    """

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Semantic search over the knowledge base.

        Args:
            query: Natural-language query.
            top_k: Maximum number of results to return.
            where: Optional ChromaDB-style metadata filter, e.g.
                ``{"year": {"$gte": 2020}}``.
        """
        ...

    def get_related_entities(
        self,
        entity: str,
        relation: str | None = None,
    ) -> list[str]:
        """Return entities related to ``entity``.

        The ChromaDB-only implementation always returns ``[]``. The future
        graph backend will query Neo4j for related methods/models/datasets.
        """
        ...

    def add_documents(self, chunks: list[Chunk]) -> int:
        """Add document chunks to the knowledge base.

        Returns:
            Number of chunks added.
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks belonging to ``doc_id``.

        Returns:
            ``True`` if the delete request was accepted.
        """
        ...

    def list_documents(self) -> list[dict[str, Any]]:
        """List all documents in the knowledge base."""
        ...
