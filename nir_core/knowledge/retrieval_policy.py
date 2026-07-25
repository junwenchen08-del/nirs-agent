"""Post-retrieval policy for evidence diversity and calibrated abstention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nir_core.knowledge.base import SearchResult


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """Controls candidate over-fetching, diversity, and no-answer rejection."""

    candidate_multiplier: int = 4
    max_candidates: int = 80
    max_chunks_per_document: int = 2
    strong_score_threshold: float = 0.62
    weak_score_threshold: float = 0.58
    min_document_margin: float = 0.02
    answerability_enabled: bool = True
    diversity_enabled: bool = True

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be at least 1")
        if not 0.0 <= self.weak_score_threshold <= 1.0:
            raise ValueError("weak_score_threshold must be in [0, 1]")
        if not 0.0 <= self.strong_score_threshold <= 1.0:
            raise ValueError("strong_score_threshold must be in [0, 1]")
        if self.weak_score_threshold > self.strong_score_threshold:
            raise ValueError(
                "weak_score_threshold cannot exceed strong_score_threshold"
            )
        if not 0.0 <= self.min_document_margin <= 1.0:
            raise ValueError("min_document_margin must be in [0, 1]")

    def candidate_count(self, top_k: int) -> int:
        """Return the bounded number of vector candidates to request."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        return min(
            self.max_candidates,
            max(top_k, top_k * self.candidate_multiplier),
        )


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    """Results plus diagnostics explaining the answerability decision."""

    results: tuple[SearchResult, ...]
    abstained: bool
    reason: str
    candidate_count: int
    top_score: float | None
    runner_up_document_score: float | None
    document_margin: float | None

    def diagnostics(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic payload."""
        return {
            "abstained": self.abstained,
            "reason": self.reason,
            "candidate_count": self.candidate_count,
            "result_count": len(self.results),
            "top_score": self.top_score,
            "runner_up_document_score": self.runner_up_document_score,
            "document_margin": self.document_margin,
        }


def _document_key(result: SearchResult) -> str:
    metadata = result.chunk.metadata or {}
    return str(metadata.get("doc_id") or result.chunk.source or result.chunk.id)


def apply_retrieval_policy(
    candidates: list[SearchResult],
    *,
    top_k: int,
    policy: RetrievalPolicy,
) -> RetrievalDecision:
    """Apply calibrated abstention, then greedy per-document diversity."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        return RetrievalDecision(
            results=(),
            abstained=True,
            reason="no_candidates",
            candidate_count=0,
            top_score=None,
            runner_up_document_score=None,
            document_margin=None,
        )

    ranked = sorted(candidates, key=lambda result: result.score, reverse=True)
    top_score = float(ranked[0].score)
    top_document = _document_key(ranked[0])
    runner_up_document_score = next(
        (
            float(result.score)
            for result in ranked[1:]
            if _document_key(result) != top_document
        ),
        None,
    )
    document_margin = (
        top_score - runner_up_document_score
        if runner_up_document_score is not None
        else None
    )

    abstained = False
    reason = "answerability_disabled"
    if policy.answerability_enabled:
        if top_score >= policy.strong_score_threshold:
            reason = "strong_score"
        elif top_score < policy.weak_score_threshold:
            abstained = True
            reason = "below_weak_score"
        elif runner_up_document_score is None:
            reason = "single_document_support"
        elif (
            document_margin is not None
            and document_margin >= policy.min_document_margin
        ):
            reason = "document_margin"
        else:
            abstained = True
            reason = "ambiguous_across_documents"

    if abstained:
        selected: list[SearchResult] = []
    elif not policy.diversity_enabled:
        selected = ranked[:top_k]
    else:
        selected = []
        document_counts: dict[str, int] = {}
        for result in ranked:
            document = _document_key(result)
            if document_counts.get(document, 0) >= policy.max_chunks_per_document:
                continue
            selected.append(result)
            document_counts[document] = document_counts.get(document, 0) + 1
            if len(selected) >= top_k:
                break

    return RetrievalDecision(
        results=tuple(selected),
        abstained=abstained,
        reason=reason,
        candidate_count=len(ranked),
        top_score=top_score,
        runner_up_document_score=runner_up_document_score,
        document_margin=document_margin,
    )
