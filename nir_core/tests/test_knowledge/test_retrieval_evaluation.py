from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from nir_core.knowledge.evaluation import (
    RetrievalCase,
    compare_retrievers,
    evaluate_retriever,
    load_retrieval_cases,
)


@dataclass
class _Chunk:
    id: str
    metadata: dict


@dataclass
class _Result:
    chunk: _Chunk


class _FakeRetriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def search(self, query: str, top_k: int = 5) -> list[_Result]:
        return [
            _Result(_Chunk(id=value, metadata={"doc_id": value}))
            for value in self.rankings.get(query, [])[:top_k]
        ]


def test_evaluate_retriever_reports_recall_mrr_ndcg_and_no_hit() -> None:
    cases = [
        RetrievalCase(query="中文查询", relevant_ids=("doc-b",), language="zh"),
        RetrievalCase(query="unknown", relevant_ids=(), expect_no_hit=True),
    ]
    retriever = _FakeRetriever({"中文查询": ["doc-a", "doc-b", "doc-c"], "unknown": []})

    report = evaluate_retriever(retriever, cases, top_k=3, clock=lambda: 1.0)

    assert report["n_cases"] == 2
    assert report["retrieval_cases"] == 1
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 0.5
    assert report["ndcg_at_k"] == pytest.approx(1.0 / 1.5849625007)
    assert report["no_hit_accuracy"] == 1.0
    assert report["by_language"]["zh"]["recall_at_k"] == 1.0


def test_compare_retrievers_keeps_model_results_side_by_side() -> None:
    cases = [RetrievalCase(query="cars 波长选择", relevant_ids=("cars",))]

    comparison = compare_retrievers(
        {
            "baseline": _FakeRetriever({"cars 波长选择": ["other"]}),
            "candidate": _FakeRetriever({"cars 波长选择": ["cars"]}),
        },
        cases,
        top_k=1,
    )

    assert comparison["baseline"]["recall_at_k"] == 0.0
    assert comparison["candidate"]["recall_at_k"] == 1.0


def test_retrieval_case_rejects_ambiguous_no_hit_label() -> None:
    with pytest.raises(ValueError, match="relevant_ids must be empty"):
        RetrievalCase(query="ambiguous", relevant_ids=("doc",), expect_no_hit=True)


def test_versioned_bge_m3_case_file_is_valid_and_balanced() -> None:
    path = (
        Path(__file__).parents[2] / "knowledge" / "retrieval_eval_cases.bge-m3.v1.json"
    )

    cases = load_retrieval_cases(path)

    assert len(cases) == 28
    assert sum(case.expect_no_hit for case in cases) == 4
    assert sum("single-document" in case.tags for case in cases) == 20
    assert sum("cross-document" in case.tags for case in cases) == 4
    assert {case.language for case in cases} == {"zh", "en"}
