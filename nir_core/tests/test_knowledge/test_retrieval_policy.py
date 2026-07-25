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
        [_result("a", 0.63, 0), _result("b", 0.629, 0)],
        top_k=1,
        policy=RetrievalPolicy(),
    )

    assert decision.abstained is False
    assert decision.reason == "strong_score"
    assert len(decision.results) == 1
