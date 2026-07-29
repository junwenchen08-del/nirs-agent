from __future__ import annotations

import pytest

from nir_core.knowledge.base import Chunk, SearchResult
from nir_core.knowledge.retrieval_policy import (
    RetrievalPolicy,
    apply_retrieval_policy,
)


def _result(doc_id: str, score: float, index: int) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=f"{doc_id}-{index}",
            content=f"content {index}",
            source=f"{doc_id}.pdf",
            metadata={"doc_id": doc_id},
        ),
        score=score,
    )


def test_policy_overfetches_then_limits_chunks_per_document() -> None:
    policy = RetrievalPolicy(max_chunks_per_document=2)
    candidates = [
        _result("a", 0.90, 0),
        _result("a", 0.89, 1),
        _result("a", 0.88, 2),
        _result("b", 0.80, 0),
        _result("c", 0.70, 0),
    ]

    decision = apply_retrieval_policy(candidates, top_k=4, policy=policy)

    assert [result.chunk.metadata["doc_id"] for result in decision.results] == [
        "a",
        "a",
        "b",
        "c",
    ]
    assert decision.abstained is False
    assert policy.candidate_count(5) == 20


def test_policy_rejects_low_score_candidates() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.55, 0), _result("b", 0.54, 0)],
        top_k=5,
        policy=RetrievalPolicy(),
    )

    assert decision.results == ()
    assert decision.abstained is True
    assert decision.reason == "below_weak_score"


def test_policy_rejects_ambiguous_mid_score_candidates() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.618, 0), _result("b", 0.616, 0)],
        top_k=5,
        policy=RetrievalPolicy(),
    )

    assert decision.results == ()
    assert decision.abstained is True
    assert decision.reason == "ambiguous_across_documents"
    assert decision.document_margin == pytest.approx(0.002)


def test_policy_accepts_mid_score_with_clear_document_margin() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.603, 0), _result("b", 0.57, 0)],
        top_k=5,
        policy=RetrievalPolicy(),
    )

    assert decision.abstained is False
    assert decision.reason == "document_margin"
    assert [result.chunk.metadata["doc_id"] for result in decision.results] == [
        "a",
        "b",
    ]


def test_policy_accepts_strong_score_without_margin() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.65, 0), _result("b", 0.649, 0)],
        top_k=1,
        policy=RetrievalPolicy(),
    )

    assert decision.abstained is False
    assert decision.reason == "strong_score"
    assert len(decision.results) == 1


def test_policy_rejects_expanded_corpus_false_positive_boundary() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.6218, 0), _result("b", 0.6182, 0)],
        top_k=5,
        policy=RetrievalPolicy(),
    )

    assert decision.results == ()
    assert decision.abstained is True
    assert decision.reason == "ambiguous_across_documents"


def test_policy_accepts_expanded_corpus_positive_boundary() -> None:
    decision = apply_retrieval_policy(
        [_result("a", 0.6364, 0), _result("b", 0.6335, 0)],
        top_k=5,
        policy=RetrievalPolicy(),
    )

    assert decision.abstained is False
    assert decision.reason == "strong_score"


def test_policy_uses_rerank_order_but_dense_scores_for_answerability() -> None:
    dense_best = _result("dense-best", 0.70, 0)
    rerank_best = _result("rerank-best", 0.96, 0)
    rerank_best.chunk.metadata["dense_score"] = 0.63
    rerank_best.chunk.metadata["rerank_score"] = 0.96
    dense_best.chunk.metadata["dense_score"] = 0.70
    dense_best.chunk.metadata["rerank_score"] = 0.15
    dense_best.score = 0.15

    decision = apply_retrieval_policy(
        [dense_best, rerank_best],
        top_k=2,
        policy=RetrievalPolicy(),
        ranking_strategy="cross_encoder",
    )

    assert [result.chunk.metadata["doc_id"] for result in decision.results] == [
        "rerank-best",
        "dense-best",
    ]
    assert decision.top_score == pytest.approx(0.70)
    assert decision.top_rerank_score == pytest.approx(0.96)
    assert decision.ranking_strategy == "cross_encoder"
    assert decision.abstained is False


def test_reranker_cannot_override_dense_below_weak_score_gate() -> None:
    first = _result("a", 0.99, 0)
    second = _result("b", 0.98, 0)
    first.chunk.metadata["dense_score"] = 0.55
    second.chunk.metadata["dense_score"] = 0.54

    decision = apply_retrieval_policy(
        [first, second],
        top_k=2,
        policy=RetrievalPolicy(),
        ranking_strategy="cross_encoder",
    )

    assert decision.results == ()
    assert decision.abstained is True
    assert decision.reason == "below_weak_score"
    assert decision.top_rerank_score == pytest.approx(0.99)
