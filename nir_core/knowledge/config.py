"""Knowledge base configuration and retriever factory.

The factory ``get_retriever`` is the single point where the backend is
selected. Today it returns a ``ChromaDBRetriever``; the day Neo4j is
added, only the ``graph_backend`` field needs to flip and the import
inside this function will switch — no other file changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_KNOWLEDGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _KNOWLEDGE_DIR.parent.parent


def _load_repo_dotenv() -> None:
    """Load local knowledge overrides without making dotenv mandatory."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_REPO_ROOT / ".env", override=False)


_load_repo_dotenv()


def _env_path(name: str, default: Path) -> str:
    return os.path.expandvars(os.path.expanduser(os.environ.get(name, str(default))))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


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
        default_factory=lambda: _env_path(
            "NIR_KNOWLEDGE_CHROMA_PATH", _KNOWLEDGE_DIR / ".chromadb-bge-m3"
        )
    )
    collection_name: str = field(
        default_factory=lambda: os.environ.get(
            "NIR_KNOWLEDGE_COLLECTION_NAME", "nir_papers_bge_m3"
        )
    )
    catalog_path: str = field(
        default_factory=lambda: _env_path(
            "NIR_KNOWLEDGE_CATALOG_PATH",
            _KNOWLEDGE_DIR / ".knowledge_catalog.bge-m3.sqlite3",
        )
    )
    index_version: str = field(
        default_factory=lambda: os.environ.get(
            "NIR_KNOWLEDGE_INDEX_VERSION", "nir-papers-bge-m3-v1"
        )
    )

    # Embedding model — local path avoids HuggingFace network issues.
    # If a HuggingFace model ID is used instead, set HF_ENDPOINT for mirrors.
    embedding_model: str = field(
        default_factory=lambda: _env_path(
            "NIR_KNOWLEDGE_EMBEDDING_MODEL",
            Path("BAAI/bge-m3"),
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: int(
            os.environ.get("NIR_KNOWLEDGE_EMBEDDING_DIM", "1024")
        )
    )

    # Retrieval policy. Thresholds are calibrated against the versioned
    # BGE-M3 evaluation set and do not require rebuilding the vector index.
    retrieval_candidate_multiplier: int = field(
        default_factory=lambda: int(
            os.environ.get("NIR_KNOWLEDGE_RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
        )
    )
    retrieval_max_candidates: int = field(
        default_factory=lambda: int(
            os.environ.get("NIR_KNOWLEDGE_RETRIEVAL_MAX_CANDIDATES", "80")
        )
    )
    retrieval_max_chunks_per_document: int = field(
        default_factory=lambda: int(
            os.environ.get(
                "NIR_KNOWLEDGE_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT",
                "2",
            )
        )
    )
    retrieval_strong_score_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "NIR_KNOWLEDGE_RETRIEVAL_STRONG_SCORE_THRESHOLD",
                "0.62",
            )
        )
    )
    retrieval_weak_score_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "NIR_KNOWLEDGE_RETRIEVAL_WEAK_SCORE_THRESHOLD",
                "0.58",
            )
        )
    )
    retrieval_min_document_margin: float = field(
        default_factory=lambda: float(
            os.environ.get(
                "NIR_KNOWLEDGE_RETRIEVAL_MIN_DOCUMENT_MARGIN",
                "0.02",
            )
        )
    )
    retrieval_answerability_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "NIR_KNOWLEDGE_RETRIEVAL_ANSWERABILITY_ENABLED",
            True,
        )
    )
    retrieval_diversity_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "NIR_KNOWLEDGE_RETRIEVAL_DIVERSITY_ENABLED",
            True,
        )
    )

    # Chunking defaults (overridable per-call in chunker)
    chunk_strategy: str = "section"
    chunk_max_tokens: int = 512
    chunk_overlap: int = 10

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
    from nir_core.knowledge.retrieval_policy import RetrievalPolicy

    chroma = ChromaDBRetriever(
        db_path=cfg.chroma_path,
        embedding_model=cfg.embedding_model,
        collection_name=cfg.collection_name,
        retrieval_policy=RetrievalPolicy(
            candidate_multiplier=cfg.retrieval_candidate_multiplier,
            max_candidates=cfg.retrieval_max_candidates,
            max_chunks_per_document=cfg.retrieval_max_chunks_per_document,
            strong_score_threshold=cfg.retrieval_strong_score_threshold,
            weak_score_threshold=cfg.retrieval_weak_score_threshold,
            min_document_margin=cfg.retrieval_min_document_margin,
            answerability_enabled=cfg.retrieval_answerability_enabled,
            diversity_enabled=cfg.retrieval_diversity_enabled,
        ),
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
