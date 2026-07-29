from pathlib import Path

from nir_core.knowledge.config import KnowledgeConfig


def test_knowledge_config_reads_local_embedding_index_overrides(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "bge-m3"
    chroma_path = tmp_path / "chroma-bge-m3"
    catalog_path = tmp_path / "catalog-bge-m3.sqlite3"
    monkeypatch.setenv("NIR_KNOWLEDGE_EMBEDDING_MODEL", str(model_path))
    monkeypatch.setenv("NIR_KNOWLEDGE_EMBEDDING_DIM", "1024")
    monkeypatch.setenv("NIR_KNOWLEDGE_CHROMA_PATH", str(chroma_path))
    monkeypatch.setenv("NIR_KNOWLEDGE_CATALOG_PATH", str(catalog_path))
    monkeypatch.setenv("NIR_KNOWLEDGE_COLLECTION_NAME", "nir_papers_bge_m3")
    monkeypatch.setenv("NIR_KNOWLEDGE_INDEX_VERSION", "nir-papers-bge-m3-v1")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_CANDIDATE_MULTIPLIER", "6")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_MAX_CANDIDATES", "60")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT", "3")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_STRONG_SCORE_THRESHOLD", "0.65")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_WEAK_SCORE_THRESHOLD", "0.57")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_MIN_DOCUMENT_MARGIN", "0.03")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_ANSWERABILITY_ENABLED", "true")
    monkeypatch.setenv("NIR_KNOWLEDGE_RETRIEVAL_DIVERSITY_ENABLED", "false")
    reranker_path = tmp_path / "bge-reranker-v2-m3"
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_ENABLED", "true")
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_MODEL", str(reranker_path))
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_DEVICE", "cpu")
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_BATCH_SIZE", "7")
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_MAX_LENGTH", "768")
    monkeypatch.setenv("NIR_KNOWLEDGE_RERANK_MAX_CANDIDATES", "11")

    config = KnowledgeConfig()

    assert config.embedding_model == str(model_path)
    assert config.embedding_dim == 1024
    assert config.chroma_path == str(chroma_path)
    assert config.catalog_path == str(catalog_path)
    assert config.collection_name == "nir_papers_bge_m3"
    assert config.index_version == "nir-papers-bge-m3-v1"
    assert config.retrieval_candidate_multiplier == 6
    assert config.retrieval_max_candidates == 60
    assert config.retrieval_max_chunks_per_document == 3
    assert config.retrieval_strong_score_threshold == 0.65
    assert config.retrieval_weak_score_threshold == 0.57
    assert config.retrieval_min_document_margin == 0.03
    assert config.retrieval_answerability_enabled is True
    assert config.retrieval_diversity_enabled is False
    assert config.rerank_enabled is True
    assert config.rerank_model == str(reranker_path)
    assert config.rerank_device == "cpu"
    assert config.rerank_batch_size == 7
    assert config.rerank_max_length == 768
    assert config.rerank_max_candidates == 11
