from __future__ import annotations

import pytest

from nir_core.knowledge.base import Chunk, SearchResult
from nir_core.knowledge.reranker import build_passage, rerank_results
from nir_core.knowledge.retrieval_policy import RetrievalPolicy
from nir_core.knowledge.vectorstore import ChromaDBRetriever


def _result(
    doc_id: str,
    score: float,
    *,
    title: str = "",
    section_path: object = None,
) -> SearchResult:
    metadata = {"doc_id": doc_id, "title": title}
    if section_path is not None:
        metadata["section_path"] = section_path
    return SearchResult(
        chunk=Chunk(
            id=f"{doc_id}-0",
            content=f"{doc_id} body",
            source=f"{doc_id}.pdf",
            metadata=metadata,
        ),
        score=score,
    )


def test_build_passage_places_title_and_section_before_chunk_text() -> None:
    result = _result(
        "paper",
        0.7,
        title="近红外非线性建模",
        section_path='["方法", "Isomap-PLS"]',
    )

    passage = build_passage(result)

    assert passage.splitlines() == [
        "近红外非线性建模",
        "方法 > Isomap-PLS",
        "paper body",
    ]


def test_rerank_results_preserves_dense_score_and_orders_by_cross_encoder() -> None:
    candidates = [_result("dense-first", 0.81), _result("semantic-first", 0.64)]

    reranked = rerank_results(candidates, [0.12, 0.94])

    assert [result.chunk.metadata["doc_id"] for result in reranked] == [
        "semantic-first",
        "dense-first",
    ]
    assert reranked[0].score == pytest.approx(0.94)
    assert reranked[0].chunk.metadata["dense_score"] == pytest.approx(0.64)
    assert reranked[0].chunk.metadata["rerank_score"] == pytest.approx(0.94)
    assert candidates[0].chunk.metadata.get("dense_score") is None


def test_rerank_results_rejects_score_count_mismatch() -> None:
    with pytest.raises(ValueError, match="score count"):
        rerank_results([_result("a", 0.7)], [])


def test_vector_retriever_fails_open_when_reranker_raises() -> None:
    class FakeModel:
        def encode(self, queries):
            return [[0.1, 0.2]]

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            return {
                "ids": [["a-0", "b-0"]],
                "documents": [["paper a", "paper b"]],
                "metadatas": [
                    [
                        {
                            "doc_id": "a",
                            "source": "a.pdf",
                            "review_status": "published",
                        },
                        {
                            "doc_id": "b",
                            "source": "b.pdf",
                            "review_status": "published",
                        },
                    ]
                ],
                "distances": [[0.2, 0.3]],
            }

    class FailingReranker:
        model_name = "broken-reranker"

        def rerank(self, query, candidates):
            raise RuntimeError("model unavailable")

    retriever = object.__new__(ChromaDBRetriever)
    retriever._model = FakeModel()
    retriever._collection = FakeCollection()
    retriever._retrieval_policy = RetrievalPolicy()
    retriever._reranker = FailingReranker()
    retriever._rerank_max_candidates = 12

    decision = retriever.search_with_diagnostics("query", top_k=2)

    assert [result.chunk.metadata["doc_id"] for result in decision.results] == [
        "a",
        "b",
    ]
    assert decision.ranking_strategy == "dense"
    assert decision.rerank_error == "RuntimeError"
    assert decision.top_score == pytest.approx(0.8)


def test_vector_retriever_caps_cross_encoder_candidate_pool() -> None:
    class FakeModel:
        def encode(self, queries):
            return [[0.1, 0.2]]

    class FakeCollection:
        def count(self):
            return 20

        def query(self, **kwargs):
            return {
                "ids": [[f"doc-{index}" for index in range(20)]],
                "documents": [[f"paper {index}" for index in range(20)]],
                "metadatas": [
                    [
                        {
                            "doc_id": ("dominant" if index < 8 else f"doc-{index}"),
                            "source": f"doc-{index}.pdf",
                            "review_status": "published",
                        }
                        for index in range(20)
                    ]
                ],
                "distances": [[0.1 + index * 0.01 for index in range(20)]],
            }

    class RecordingReranker:
        model_name = "recording-reranker"

        def __init__(self):
            self.candidate_count = 0
            self.document_ids = []

        def rerank(self, query, candidates):
            self.candidate_count = len(candidates)
            self.document_ids = [
                candidate.chunk.metadata["doc_id"] for candidate in candidates
            ]
            scores = [(index + 1) / len(candidates) for index in range(len(candidates))]
            return rerank_results(candidates, scores)

    reranker = RecordingReranker()
    retriever = object.__new__(ChromaDBRetriever)
    retriever._model = FakeModel()
    retriever._collection = FakeCollection()
    retriever._retrieval_policy = RetrievalPolicy()
    retriever._reranker = reranker
    retriever._rerank_max_candidates = 12

    decision = retriever.search_with_diagnostics("query", top_k=5)

    assert reranker.candidate_count == 12
    assert reranker.document_ids.count("dominant") == 2
    assert "doc-17" in reranker.document_ids
    assert decision.candidate_count == 12
    assert decision.ranking_strategy == "cross_encoder"
