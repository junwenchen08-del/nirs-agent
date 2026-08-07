"""NIR reflection tool: deterministic decision for the preprocessing retry loop."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _ok, _resolve
from ._knowledge_hint import _build_knowledge_hint


def _classification_reflection(
    metrics: dict,
    history_list: list,
    *,
    attempt: int,
    max_retries: int,
    get_next_pipeline,
) -> str:
    """Build retry evidence for a classification run without regression assumptions."""

    quality = metrics.get("quality") if isinstance(metrics.get("quality"), dict) else {}
    holdout = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    per_class = holdout.get("per_class") if isinstance(holdout.get("per_class"), dict) else {}
    recalls = {str(label): float(values["recall"]) for label, values in per_class.items() if isinstance(values, dict) and isinstance(values.get("recall"), int | float)}
    weakest_class = min(recalls, key=recalls.get) if recalls else None
    diagnostics = {
        "balanced_accuracy": holdout.get("balanced_accuracy"),
        "macro_f1": holdout.get("macro_f1"),
        "weakest_class": weakest_class,
        "weakest_class_recall": recalls.get(weakest_class) if weakest_class else None,
        "class_recalls": recalls,
    }
    current_pipeline = metrics.get("preprocessing_steps") or []
    selection_history = [
        *history_list,
        {"pipeline": current_pipeline, "metrics": {"holdout": holdout}},
    ]
    steps = get_next_pipeline(attempt=attempt, history=selection_history)
    can_retry = quality.get("passed") is not True and attempt < max_retries and steps is not None
    next_steps = [{"method": step.method, "params": step.params} for step in steps] if can_retry and steps is not None else None
    reason = "Classification quality gate passed; no retry is needed." if quality.get("passed") is True else "Classification holdout quality is below the configured gate."
    if weakest_class:
        reason += f" Weakest class is {weakest_class!r} with recall {recalls[weakest_class]:.4f}."
    if can_retry:
        reason += " Retry with a materially different leakage-safe preprocessing pipeline."
    elif quality.get("passed") is not True:
        reason += " Retry budget or approved preprocessing candidates are exhausted."

    records = [*history_list, {"pipeline": current_pipeline, "metrics": {"holdout": holdout}}]
    best = None
    for record in records:
        record_metrics = record.get("metrics", {}) if isinstance(record, dict) else {}
        section = record_metrics.get("holdout") or record_metrics.get("test") or record_metrics
        score = section.get("balanced_accuracy") if isinstance(section, dict) else None
        if isinstance(score, int | float) and (best is None or score > best["balanced_accuracy"]):
            best = {
                "balanced_accuracy": float(score),
                "macro_f1": section.get("macro_f1"),
                "pipeline": record.get("pipeline", []),
            }

    return _ok(
        {
            "task_kind": "classification",
            "should_retry": can_retry,
            "reason": reason,
            "diagnostics": diagnostics,
            "fallback_suggestion": [step["method"] for step in next_steps] if next_steps else None,
            "fallback_suggestion_steps": next_steps,
            "lv_adjustment": {},
            "current_quality": {
                "grade": quality.get("grade", "unknown"),
                "passed": quality.get("passed") is True,
                "action": quality.get("action"),
                "thresholds_used": quality.get("thresholds_used", {}),
            },
            "best_so_far": best,
            "attempt": attempt,
            "next_attempt": attempt + 1 if can_retry else None,
            "knowledge_hint": None,
        }
    )


@tool("nir_reflect", parse_docstring=True)
def nir_reflect_tool(
    runtime: Runtime,
    metrics_path: str,
    history: str = "[]",
    domain: str = "default",
    attempt: int = 1,
    max_retries: int = 3,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Deterministic reflection decision for the NIR preprocessing retry loop.

    ★ V3.6: Returns diagnostics (residual trend/variance/outliers) from the
    metrics file so the LLM can reason about *what* to try next.  The
    deterministic ``fallback_suggestion`` is provided as a safety net — the
    LLM should prefer constructing its own ``pipeline_steps`` based on the
    diagnostics, falling back to the suggestion only when stuck.

    Wraps should_retry + get_next_pipeline + suggest_lv_adjustment into a
    single deterministic call.  Call this after every nir_train_model to
    decide what to do next.

    Args:
        metrics_path: Virtual path to the metrics.json produced by
            nir_train_model or nir_analyze.
        history: JSON string of prior attempt records. Each record should
            have a pipeline field (list of method-name dicts) and a metrics
            field (a metrics dict with R2_val / RPD). Pass "[]" for the
            first attempt.
        domain: Application domain for quality-gate thresholds.
        attempt: Current 1-based attempt number.
        max_retries: Maximum allowed retries (default 3).

    Returns:
        JSON with should_retry, reason, diagnostics, fallback_suggestion,
        fallback_suggestion_steps, lv_adjustment, current_quality,
        best_so_far, attempt, next_attempt, and knowledge_hint.

        ★ knowledge_hint: When non-null, the LLM SHOULD immediately call
        ``nir_search_knowledge(query=hint['query'])`` to retrieve relevant
        paper sections before constructing the next pipeline. Triggered by:
        unknown domain, grade ∈ {C,D,F} after attempt ≥ 2, or R²_val < 0.7.
        The hint includes a ready-to-use query string and a reason.
    """
    try:
        import json as _json

        from nir_core.utils.validation import (
            get_next_pipeline,
            should_retry,
            suggest_lv_adjustment,
        )

        # Load current metrics.
        real_metrics = _resolve(runtime, metrics_path, read_only=True)
        with open(real_metrics, encoding="utf-8") as f:
            metrics = _json.load(f)

        # Parse history.
        try:
            history_list = _json.loads(history) if isinstance(history, str) else history
        except (ValueError, TypeError):
            history_list = []

        if metrics.get("task_kind") == "classification":
            return _classification_reflection(
                metrics,
                history_list if isinstance(history_list, list) else [],
                attempt=attempt,
                max_retries=max_retries,
                get_next_pipeline=get_next_pipeline,
            )

        n_samples = metrics.get("n_samples")

        # Quality + LV adjustment.
        lv_adj = suggest_lv_adjustment(
            metrics,
            domain=domain,
            n_samples=n_samples,
        )
        quality = lv_adj["quality"]

        # Should retry?
        retry = should_retry(
            quality,
            attempt=attempt,
            max_retries=max_retries,
            history=history_list,
        )

        # Next pipeline.
        next_pipeline = None
        next_steps = None
        if retry:
            steps = get_next_pipeline(attempt=attempt, history=history_list)
            if steps is not None:
                next_pipeline = [s.method for s in steps]
                next_steps = [{"method": s.method, "params": s.params} for s in steps]
            else:
                # Candidates exhausted — override retry to False.
                retry = False

        # Find best-so-far from history + current.
        all_records = list(history_list) + [
            {
                "pipeline": metrics.get("preprocessing_steps", []),
                "metrics": metrics,
            }
        ]
        best = None
        for rec in all_records:
            m = rec.get("metrics", {}) if isinstance(rec, dict) else {}
            r2 = m["R2_val"] if "R2_val" in m else m.get("R2")
            rpd_v = m.get("RPD")
            if rpd_v is None:
                rpd_v = 0.0
            best_rpd = best.get("RPD") if best is not None else None
            if best is None or rpd_v > (best_rpd if best_rpd is not None else float("-inf")):
                best = {
                    "R2_val": r2,
                    "RPD": rpd_v,
                    "grade": m.get("quality", {}).get("grade", "unknown"),
                    "pipeline": rec.get("pipeline", []),
                }

        reason_parts = []
        if not retry:
            if quality["passed"]:
                reason_parts.append("质量门禁已通过，无需重试。")
            elif attempt >= max_retries:
                reason_parts.append(f"已达最大重试次数({max_retries})，停止重试。")
            elif get_next_pipeline(attempt=attempt, history=history_list) is None:
                reason_parts.append("所有候选预处理流水线已全部尝试完毕，停止重试。")
            else:
                reason_parts.append("连续两次无显著改善，停止重试。")
        else:
            reason_parts.append(f"当前等级={quality['grade']}，未通过门禁。")
            if lv_adj["issues"]:
                reason_parts.append(f"检测到问题: {', '.join(lv_adj['issues'])}。")
            diagnostics = metrics.get("diagnostics", {})
            if diagnostics:
                reason_parts.append(f"残差诊断: trend={diagnostics.get('residual_trend', 'unknown')}, variance={diagnostics.get('residual_variance', 'unknown')}, outliers={diagnostics.get('outlier_count', 0)}。")
                reason_parts.append("请根据诊断信息自主构造下一步 pipeline_steps（含超参数），经 validate_pipeline 校验后调用 nir_train_model。")
            reason_parts.append(lv_adj["recommendation"])

        # ★ Knowledge-base hint: structured retrieval suggestion so the LLM
        # has a concrete query to pass to nir_search_knowledge instead of
        # recalling SKILL.md trigger rules. See _build_knowledge_hint for
        # the trigger logic.
        knowledge_hint = _build_knowledge_hint(
            domain=domain,
            grade=quality.get("grade"),
            passed=quality.get("passed"),
            attempt=attempt,
            r2_val=metrics.get("R2_val"),
            diagnostics=metrics.get("diagnostics"),
        )
        if knowledge_hint:
            reason_parts.append(f"知识库建议: {knowledge_hint['reason']}")

        return _ok(
            {
                "should_retry": retry,
                "reason": " ".join(reason_parts),
                "diagnostics": metrics.get("diagnostics", {}),
                "fallback_suggestion": next_pipeline,
                "fallback_suggestion_steps": next_steps,
                "lv_adjustment": {
                    "current_n": lv_adj["current_n"],
                    "suggested_n": lv_adj["suggested_n"],
                    "change": lv_adj["change"],
                    "delta": lv_adj["delta"],
                    "issues": lv_adj["issues"],
                    "recommendation": lv_adj["recommendation"],
                },
                "current_quality": {
                    "grade": quality["grade"],
                    "passed": quality["passed"],
                    "action": quality["action"],
                    "thresholds_used": quality["thresholds_used"],
                },
                "best_so_far": best,
                "attempt": attempt,
                "next_attempt": (attempt + 1) if retry else None,
                "knowledge_hint": knowledge_hint,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
