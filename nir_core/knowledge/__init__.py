"""NIR knowledge base RAG system.

Provides document parsing, smart chunking, entity extraction, and vector
retrieval for the NIR Agent.

Quick start:
    from nir_core.knowledge.config import get_retriever
    retriever = get_retriever()
    results = retriever.search("SNV scatter correction soil")
"""

from nir_core.knowledge.base import Chunk, SearchResult
from nir_core.knowledge.evidence import assess_evidence

__all__ = ["Chunk", "SearchResult", "assess_evidence"]
