"""Knowledge base configuration and retriever factory.

The factory ``get_retriever`` is the single point where the backend is
selected. Today it returns a ``ChromaDBRetriever``; the day Neo4j is
added, only the ``graph_backend`` field needs to flip and the import
inside this function will switch — no other file changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class KnowledgeConfig:
    """Configuration for the NIR knowledge base.

    Vector backend: always ChromaDB today.
    Graph backend: ``None`` today; set to ``"neo4j"`` (plus ``neo4j_uri``)
        to activate the future ``GraphEnhancedRetriever``.
    """

    # Vector backend (only "chromadb" is supported today)
    vector_backend: str = "chromadb"

    # Graph backend (None today; "neo4j" in the future)
    graph_backend: str | None = None

    # ChromaDB
    chroma_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), ".chromadb"
        )
    )
    collection_name: str = "nir_papers"

    # Embedding model — local path avoids HuggingFace network issues.
    # If a HuggingFace model ID is used instead, set HF_ENDPOINT for mirrors.
    embedding_model: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "all-MiniLM-L6-v2"
        )
    )
    embedding_dim: int = 384

    # Chunking defaults (overridable per-call in chunker)
    chunk_strategy: str = "section"
    chunk_max_tokens: int = 512
    chunk_overlap: int = 0

    # Neo4j (reserved; unused today)
    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None


_default_config: KnowledgeConfig | None = None


def get_config() -> KnowledgeConfig:
    """Return the process-wide default config (singleton)."""
    global _default_config
    if _default_config is None:
        _default_config = KnowledgeConfig()
    return _default_config


def set_config(config: KnowledgeConfig) -> None:
    """Override the default config (mainly for tests)."""
    global _default_config
    _default_config = config


def get_retriever(config: KnowledgeConfig | None = None):
    """Return a retriever instance honouring the ``KnowledgeRetriever`` Protocol.

    Today: always ``ChromaDBRetriever``.
    Future: if ``config.graph_backend == "neo4j"``, return
    ``GraphEnhancedRetriever`` wrapping the ChromaDB retriever.
    """
    cfg = config or get_config()

    from nir_core.knowledge.vectorstore import ChromaDBRetriever

    chroma = ChromaDBRetriever(
        db_path=cfg.chroma_path,
        embedding_model=cfg.embedding_model,
        collection_name=cfg.collection_name,
    )

    if cfg.graph_backend == "neo4j" and cfg.neo4j_uri:
        # Future extension — see IMPLEMENTATION_PLAN.md §7.
        # from nir_core.knowledge.graph_retriever import GraphEnhancedRetriever
        # from nir_core.knowledge.neo4j_client import Neo4jClient
        # neo4j = Neo4jClient(cfg.neo4j_uri, cfg.neo4j_user, cfg.neo4j_password)
        # return GraphEnhancedRetriever(chroma, neo4j)
        raise NotImplementedError(
            "Graph backend is reserved but not yet implemented. "
            "See IMPLEMENTATION_PLAN.md §7."
        )

    return chroma
