"""Shared calibration-only retrieval and CV selection for regression trainers."""

from __future__ import annotations

import math


def _finite_evidence(value):
    if isinstance(value, dict):
        return {key: _finite_evidence(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_evidence(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def select_preprocessing(X_cal, y_cal, X_tune, y_tune, wv, *, max_components: int = 10):
    """Retrieve references before first candidate fit; never receive holdout rows."""
    from nir_core.model.evaluation import nested_cv_preprocessing
    from nir_core.preprocess.recommendation import recommend_preprocessing

    from .method_catalog import get_method_knowledge_base

    recommendation = recommend_preprocessing(X_cal, wv, budget="standard", knowledge_mode="auto", knowledge_base=get_method_knowledge_base())
    best, evaluation = nested_cv_preprocessing(
        X_cal,
        y_cal,
        X_tune,
        y_tune,
        candidate_pipelines=recommendation.pipelines(),
        inner_folds=3,
        max_components=min(10, max(1, int(max_components))),
        random_state=42,
        wv=wv,
        selection_rule="rmsecv_1pct",
        candidate_ids=[candidate.candidate_id for candidate in recommendation.candidates],
    )
    evidence = {"recommendation": recommendation.as_dict(), "evaluation": _finite_evidence(evaluation)}
    evidence["evaluation"]["cv_configuration"] = {"scope": "calibration_only", "inner_folds": 3, "random_state": 42, "max_components": min(10, max(1, int(max_components)))}
    return best.unfitted_copy().fit(X_cal, wv), evidence


def pipeline_records(pipeline) -> list[dict]:
    return [{"method": step.method, "params": dict(step.params)} for step in pipeline.steps]


def method_knowledge_summary(knowledge: dict | None) -> dict:
    """Keep sources and reasons in tool context without copying whole cards."""
    knowledge = knowledge or {}
    return {key: knowledge.get(key) for key in ("mode", "reason", "query", "index_version", "retrieval_strategy", "profile_scope", "problem_labels", "parameter_policy")} | {
        "sources": [{key: item.get(key) for key in ("method_id", "title", "evidence_id", "source_url", "matched_tags", "auto_eligible", "eligibility_reason")} for item in knowledge.get("results", [])[:6]]
    }
