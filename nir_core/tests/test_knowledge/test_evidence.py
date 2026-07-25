from __future__ import annotations

from nir_core.knowledge.base import Chunk, SearchResult
from nir_core.knowledge.evidence import assess_evidence


def _result(
    evidence_id: str,
    doc_id: str,
    *,
    tier: str = "C",
    score: float = 0.8,
    with_doi: bool = True,
) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            id=evidence_id,
            content="Evidence passage",
            source=f"{doc_id}.pdf",
            metadata={
                "doc_id": doc_id,
                "title": f"Paper {doc_id}",
                "authors": ["A. Researcher", "B. Author"],
                "year": 2025,
                "doi": f"10.1000/{doc_id}" if with_doi else "",
                "page_start": 3,
                "page_end": 4,
                "quality_tier": tier,
                "section_path": ["Results"],
            },
        ),
        score=score,
    )


def test_answer_can_use_one_cited_document() -> None:
    assessment = assess_evidence([_result("ev-1", "doc-1")])

    assert assessment["answer_allowed"] is True
    assert assessment["decision_allowed"] is False
    assert assessment["status"] == "ready"
    assert assessment["citations"][0]["citation_marker"] == "[KB:ev-1]"
    assert (
        "https://doi.org/10.1000/doc-1" in assessment["citations"][0]["citation_text"]
    )


def test_decision_requires_two_independent_acceptable_documents() -> None:
    assessment = assess_evidence(
        [_result("ev-1", "doc-1"), _result("ev-2", "doc-2")],
        purpose="decision",
    )

    assert assessment["decision_allowed"] is True
    assert assessment["support_level"] == "moderate"
    assert assessment["conflict_status"] == "requires_explicit_comparison"


def test_chunks_from_same_document_do_not_satisfy_decision_gate() -> None:
    assessment = assess_evidence(
        [_result("ev-1", "doc-1"), _result("ev-2", "doc-1")],
        purpose="decision",
    )

    assert assessment["decision_allowed"] is False
    assert assessment["status"] == "insufficient"
    assert "decision_requires_two_independent_documents" in assessment["limitations"]


def test_low_quality_second_document_does_not_satisfy_decision_gate() -> None:
    assessment = assess_evidence(
        [
            _result("ev-1", "doc-1", tier="B"),
            _result("ev-2", "doc-2", tier="D"),
        ],
        purpose="decision",
    )

    assert assessment["decision_allowed"] is False
    assert "insufficient_quality_tier_a_to_c_support" in assessment["limitations"]


def test_empty_results_require_abstention() -> None:
    assessment = assess_evidence([], purpose="decision")

    assert assessment["answer_allowed"] is False
    assert assessment["decision_allowed"] is False
    assert assessment["support_level"] == "insufficient"


def test_missing_doi_lowers_completeness_without_blocking_decision() -> None:
    assessment = assess_evidence(
        [
            _result("ev-1", "doc-1"),
            _result("ev-2", "doc-2", with_doi=False),
        ],
        purpose="decision",
    )

    assert assessment["decision_allowed"] is True
    assert assessment["doi_document_count"] == 1
    assert assessment["missing_doi_document_count"] == 1
    assert assessment["doi_completeness"] == "partial"
    assert "1_independent_documents_missing_doi" in assessment["limitations"]
