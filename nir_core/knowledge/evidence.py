"""Decision-grade evidence packaging for retrieved NIR literature."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from nir_core.knowledge.base import SearchResult

_ACCEPTABLE_DECISION_TIERS = {"A", "B", "C"}


def _metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value:
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return [value]
        if isinstance(decoded, list):
            return [str(item) for item in decoded if str(item).strip()]
        return [value]
    return []


def _citation(result: SearchResult) -> dict[str, Any]:
    metadata = result.chunk.metadata or {}
    authors = _metadata_list(metadata, "authors")
    title = str(metadata.get("title") or result.chunk.source or "Untitled source")
    year = metadata.get("year")
    doi = str(metadata.get("doi") or "").strip()
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if page_start is None:
        locator = ""
    elif page_end is not None and page_end != page_start:
        locator = f"pp. {page_start}-{page_end}"
    else:
        locator = f"p. {page_start}"
    author_label = authors[0] if authors else "Unknown author"
    if len(authors) > 1:
        author_label += " et al."
    year_label = str(year) if year is not None else "n.d."
    parts = [f"{author_label} ({year_label}). {title}."]
    if doi:
        parts.append(f"https://doi.org/{doi.removeprefix('https://doi.org/')}")
    elif result.chunk.source:
        parts.append(str(result.chunk.source))
    if locator:
        parts.append(locator)
    return {
        "evidence_id": result.chunk.id,
        "doc_id": str(metadata.get("doc_id") or result.chunk.source),
        "citation_marker": f"[KB:{result.chunk.id}]",
        "citation_text": " ".join(parts),
        "title": title,
        "authors": authors,
        "year": year,
        "doi": doi,
        "source": result.chunk.source,
        "page_start": page_start,
        "page_end": page_end,
        "section_path": _metadata_list(metadata, "section_path"),
        "quality_tier": str(metadata.get("quality_tier") or "C").upper(),
        "score": round(float(result.score), 4),
    }


def assess_evidence(
    results: Iterable[SearchResult],
    *,
    purpose: str = "answer",
) -> dict[str, Any]:
    """Assess whether retrieved literature is sufficient for an answer/decision.

    ``answer`` permits a clearly cited single source.  ``decision`` requires
    at least two independent documents of quality tier A-C.  Semantic
    agreement is intentionally not guessed; the agent must compare the cited
    passages and explicitly report conflicts before making a recommendation.
    """

    normalized_purpose = str(purpose).strip().lower()
    if normalized_purpose not in {"answer", "decision"}:
        raise ValueError("purpose must be 'answer' or 'decision'")
    materialized = list(results)
    citations = [_citation(result) for result in materialized]
    document_ids = list(dict.fromkeys(citation["doc_id"] for citation in citations))
    document_has_doi: dict[str, bool] = {}
    for citation in citations:
        document = citation["doc_id"]
        document_has_doi[document] = document_has_doi.get(document, False) or bool(
            citation["doi"]
        )
    doi_document_count = sum(document_has_doi.values())
    missing_doi_document_count = len(document_ids) - doi_document_count
    acceptable_documents = {
        citation["doc_id"]
        for citation in citations
        if citation["quality_tier"] in _ACCEPTABLE_DECISION_TIERS
    }
    missing_locator_count = sum(
        citation["page_start"] is None and not citation["section_path"]
        for citation in citations
    )

    answer_allowed = bool(citations)
    decision_allowed = len(document_ids) >= 2 and len(acceptable_documents) >= 2
    limitations: list[str] = []
    if not citations:
        limitations.append("no_reliable_retrieval_results")
    if normalized_purpose == "decision" and len(document_ids) < 2:
        limitations.append("decision_requires_two_independent_documents")
    if normalized_purpose == "decision" and len(acceptable_documents) < 2:
        limitations.append("insufficient_quality_tier_a_to_c_support")
    if missing_locator_count:
        limitations.append(f"{missing_locator_count}_citation_locators_missing")
    if missing_doi_document_count:
        limitations.append(
            f"{missing_doi_document_count}_independent_documents_missing_doi"
        )
    if normalized_purpose == "decision":
        limitations.append("semantic_conflicts_require_explicit_agent_comparison")

    if not citations:
        support_level = "insufficient"
    elif decision_allowed and len(document_ids) >= 3:
        support_level = "strong"
    elif decision_allowed:
        support_level = "moderate"
    else:
        support_level = "limited"

    allowed = answer_allowed if normalized_purpose == "answer" else decision_allowed
    if not document_ids:
        doi_completeness = "not_applicable"
    elif missing_doi_document_count == 0:
        doi_completeness = "complete"
    elif doi_document_count:
        doi_completeness = "partial"
    else:
        doi_completeness = "missing"
    return {
        "schema_version": 1,
        "purpose": normalized_purpose,
        "status": "ready" if allowed else "insufficient",
        "answer_allowed": answer_allowed,
        "decision_allowed": decision_allowed,
        "support_level": support_level,
        "result_count": len(citations),
        "independent_document_count": len(document_ids),
        "acceptable_quality_document_count": len(acceptable_documents),
        "doi_document_count": doi_document_count,
        "missing_doi_document_count": missing_doi_document_count,
        "doi_completeness": doi_completeness,
        "conflict_status": (
            "requires_explicit_comparison"
            if normalized_purpose == "decision"
            else "not_assessed"
        ),
        "limitations": limitations,
        "citations": citations,
        "usage_contract": {
            "claim_citation_format": "[KB:evidence_id]",
            "must_cite_each_material_claim": True,
            "must_distinguish_evidence_from_inference": True,
            "must_report_conflicting_findings": normalized_purpose == "decision",
            "must_abstain_if_not_allowed": True,
        },
    }
