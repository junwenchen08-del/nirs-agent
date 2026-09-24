"""Offline, model-agnostic evaluation for NIR knowledge retrieval."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One labeled query in a Chinese/multilingual NIR retrieval set."""

    query: str
    relevant_ids: tuple[str, ...]
    expect_no_hit: bool = False
    language: str = "unknown"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query cannot be empty")
        if self.expect_no_hit and self.relevant_ids:
            raise ValueError("relevant_ids must be empty when expect_no_hit is true")
        if not self.expect_no_hit and not self.relevant_ids:
            raise ValueError("retrieval cases require at least one relevant id")


def load_retrieval_cases(path: str | Path) -> list[RetrievalCase]:
    """Load cases from a UTF-8 JSON array without initializing an embedder."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Retrieval evaluation file must contain a JSON array")
    cases: list[RetrievalCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"Evaluation case {index} must be a JSON object")
        cases.append(
            RetrievalCase(
                query=str(item.get("query", "")),
                relevant_ids=tuple(
                    str(value) for value in item.get("relevant_ids", [])
                ),
                expect_no_hit=bool(item.get("expect_no_hit", False)),
                language=str(item.get("language", "unknown")),
                tags=tuple(str(value) for value in item.get("tags", [])),
            )
        )
    return cases


def _result_id(result: Any) -> str:
    chunk = result.chunk
    metadata = getattr(chunk, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("doc_id"):
        return str(metadata["doc_id"])
    return str(chunk.id)


def _retrieval_scores(
    ranked_ids: list[str], relevant_ids: tuple[str, ...]
) -> tuple[float, float, float]:
    relevant = set(relevant_ids)
    hits = [rank for rank, value in enumerate(ranked_ids, start=1) if value in relevant]
    recall = len({value for value in ranked_ids if value in relevant}) / len(relevant)
    reciprocal_rank = 0.0 if not hits else 1.0 / hits[0]
    dcg = sum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal_hits = min(len(relevant), len(ranked_ids))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = 0.0 if ideal_dcg == 0 else dcg / ideal_dcg
    return recall, reciprocal_rank, ndcg


def _aggregate(rows: Iterable[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    values = list(rows)
    retrieval = [row for row in values if not row["expect_no_hit"]]
    no_hit = [row for row in values if row["expect_no_hit"]]
    latency_values = [float(row["latency_ms"]) for row in values]
    return {
        "n_cases": len(values),
        "retrieval_cases": len(retrieval),
        "no_hit_cases": len(no_hit),
        "top_k": top_k,
        "recall_at_k": (
            sum(float(row["recall_at_k"]) for row in retrieval) / len(retrieval)
            if retrieval
            else None
        ),
        "mrr": (
            sum(float(row["reciprocal_rank"]) for row in retrieval) / len(retrieval)
            if retrieval
            else None
        ),
        "ndcg_at_k": (
            sum(float(row["ndcg_at_k"]) for row in retrieval) / len(retrieval)
            if retrieval
            else None
        ),
        "no_hit_accuracy": (
            sum(bool(row["no_hit_correct"]) for row in no_hit) / len(no_hit)
            if no_hit
            else None
        ),
        "latency_ms_mean": (
            sum(latency_values) / len(latency_values) if latency_values else None
        ),
        "latency_ms_max": max(latency_values) if latency_values else None,
    }


def evaluate_retriever(
    retriever: Any,
    cases: Iterable[RetrievalCase],
    *,
    top_k: int = 5,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    """Evaluate one already-built index without changing its embedding model."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    rows: list[dict[str, Any]] = []
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        started = clock()
        results = retriever.search(case.query, top_k=top_k)
        latency_ms = max(0.0, (clock() - started) * 1000.0)
        ranked_ids = list(dict.fromkeys(_result_id(result) for result in results))
        recall = reciprocal_rank = ndcg = 0.0
        if not case.expect_no_hit:
            recall, reciprocal_rank, ndcg = _retrieval_scores(
                ranked_ids, case.relevant_ids
            )
        row = {
            "query": case.query,
            "language": case.language,
            "tags": list(case.tags),
            "relevant_ids": list(case.relevant_ids),
            "ranked_ids": ranked_ids,
            "expect_no_hit": case.expect_no_hit,
            "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
            "no_hit_correct": case.expect_no_hit and not ranked_ids,
            "latency_ms": latency_ms,
        }
        rows.append(row)
        by_language[case.language].append(row)

    report = _aggregate(rows, top_k=top_k)
    report["by_language"] = {
        language: _aggregate(language_rows, top_k=top_k)
        for language, language_rows in sorted(by_language.items())
    }
    report["cases"] = rows
    return report


def compare_retrievers(
    retrievers: Mapping[str, Any],
    cases: Iterable[RetrievalCase],
    *,
    top_k: int = 5,
) -> dict[str, dict[str, Any]]:
    """Evaluate separately indexed baseline/candidate retrievers side by side."""
    frozen_cases = tuple(cases)
    return {
        name: evaluate_retriever(retriever, frozen_cases, top_k=top_k)
        for name, retriever in retrievers.items()
    }
