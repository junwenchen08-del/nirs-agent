"""Deterministic grounding rules for NIR final responses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NIRResponseGroundingResult:
    """Explainable result from validating one natural-language response."""

    passed: bool
    violations: tuple[str, ...]


class NIRStreamMessageGate:
    """Delay NIR AI text until a complete state snapshot passes grounding.

    LangGraph's ``messages`` mode publishes model output before ``after_model``
    middleware can replace unsupported claims.  This small stateful gate lets
    runtime consumers suppress those raw AI chunks and publish only the
    authoritative message found in a subsequent ``values`` snapshot.
    """

    def __init__(self, workflow: Mapping[str, Any] | None = None) -> None:
        self._workflow = workflow if isinstance(workflow, Mapping) else None
        self._published_keys: set[str] = set()

    @property
    def active(self) -> bool:
        """Whether the current graph state contains an NIR workflow."""

        return self._workflow is not None

    def suppresses(self, message: Any) -> bool:
        """Return whether a raw messages-mode item must be held back."""

        return self.active and _is_ai_message(message)

    def observe_values(self, state: Mapping[str, Any] | None) -> Any | None:
        """Return the latest publishable AI message from a values snapshot.

        Unsupported final text is not marked as published, so a corrected
        message with the same LangChain id can be emitted by the next snapshot.
        Tool-call messages are publishable because they are agent actions, not
        user-facing result claims.
        """

        if not isinstance(state, Mapping):
            return None
        workflow = state.get("nir_workflow")
        self._workflow = workflow if isinstance(workflow, Mapping) else None
        if self._workflow is None:
            return None

        messages = state.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            return None
        message = messages[-1]
        if not _is_ai_message(message):
            return None

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            response_text = _message_text(message)
            if not response_text or not validate_nir_response(response_text, self._workflow).passed:
                return None

        key = _message_key(message)
        if key in self._published_keys:
            return None
        self._published_keys.add(key)
        return message

    def allows_values(self, state: Mapping[str, Any] | None) -> bool:
        """Return whether a values snapshot is safe to expose to a user."""

        if not self.active or not isinstance(state, Mapping):
            return True
        messages = state.get("messages")
        if not isinstance(messages, (list, tuple)) or not messages:
            return True
        message = messages[-1]
        if not _is_ai_message(message) or getattr(message, "tool_calls", None):
            return True
        response_text = _message_text(message)
        return bool(response_text) and validate_nir_response(response_text, self._workflow).passed


def _is_ai_message(message: Any) -> bool:
    message_type = str(getattr(message, "type", "") or "").lower()
    return message_type in {"ai", "aimessagechunk"} or type(message).__name__ in {
        "AIMessage",
        "AIMessageChunk",
    }


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


def _message_key(message: Any) -> str:
    message_id = getattr(message, "id", None)
    if message_id:
        return f"id:{message_id}"
    return f"content:{_message_text(message)}|tools:{getattr(message, 'tool_calls', None)!r}"


_METRIC_CLAIM_RE = re.compile(
    r"(?P<label>R\s*(?:²|\^?\s*2)(?:[_\s-]*(?:val|test|holdout|external))?"
    r"|RMSE(?:CV|C|P)?|RPD|bias|偏差)"
    r"\s*(?:[:=]|为|是|约为|达到|达|of)?\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<percent>\s*%)?",
    re.IGNORECASE,
)
_MODEL_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|/)[^\s`\"'<>，。；;]+?\.pkl)",
    re.IGNORECASE,
)
_METRICS_PATH_RE = re.compile(
    r"(?P<path>(?:[A-Za-z]:[\\/]|/)[^\s`\"'<>，。；;]*metrics?[^\s`\"'<>，。；;]*\.json)",
    re.IGNORECASE,
)
_EXTERNAL_VALIDATION_RE = re.compile(
    r"独立外部验证|外部验证|external\s+validation|externally\s+validated",
    re.IGNORECASE,
)
_DEPLOYMENT_READY_RE = re.compile(
    r"生产可用|可用于生产|可以用于生产|可部署|可以部署|生产部署|production[-\s]+ready|deployment[-\s]+ready|ready\s+for\s+production",
    re.IGNORECASE,
)
_QUALITY_PASSED_RE = re.compile(
    r"(?:模型|结果|质量门禁|quality\s+gate).{0,16}(?:已经|已|成功)?(?:通过|达标|合格|passed|meets?(?:\s+the)?\s+threshold)",
    re.IGNORECASE,
)
_NEGATIONS = (
    "不是",
    "并非",
    "不能",
    "不可",
    "不等同",
    "不属于",
    "未",
    "没有",
    "not ",
    "no ",
    "cannot",
    "can't",
    "without",
)


def _metric_category(label: str) -> str:
    normalized = re.sub(r"[\s_^²-]+", "", label).lower()
    if normalized.startswith("r2") or normalized == "r":
        return "r2"
    if normalized.startswith("rmsecv"):
        return "rmsecv"
    if normalized.startswith("rmsep"):
        return "rmsep"
    if normalized.startswith("rmsec"):
        return "rmsec"
    if normalized.startswith("rmse"):
        return "rmse"
    if normalized.startswith("rpd"):
        return "rpd"
    return "bias"


def _summary_values(summary: Mapping[str, Any]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}

    def add(category: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return
        values.setdefault(category, []).append(float(value))

    def walk(value: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                walk(nested, (*path, str(key)))
            return
        if isinstance(value, list):
            for item in value:
                walk(item, path)
            return
        if not path:
            return
        key = re.sub(r"[\s_^²-]+", "", path[-1]).lower()
        parent = re.sub(r"[\s_^²-]+", "", path[-2]).lower() if len(path) > 1 else ""
        if key.startswith("r2"):
            add("r2", value)
        elif key == "rpd":
            add("rpd", value)
        elif key == "rmsecv":
            add("rmsecv", value)
            add("rmse", value)
        elif key == "rmsep":
            add("rmsep", value)
            add("rmse", value)
        elif key == "rmsec":
            add("rmsec", value)
            add("rmse", value)
        elif key == "rmse":
            add("rmse", value)
            if parent in {"holdout", "external", "test", "externaltest", "holdouttest"}:
                add("rmsep", value)
        elif key == "bias":
            add("bias", value)

    walk(summary)
    return values


def _claim_tolerance(raw_value: str, *, percent: bool) -> float:
    mantissa = re.split(r"[eE]", raw_value, maxsplit=1)[0]
    decimals = len(mantissa.split(".", maxsplit=1)[1]) if "." in mantissa else 0
    tolerance = 0.5 * (10**-decimals)
    return tolerance / 100 if percent else tolerance


def _has_positive_marker(text: str, pattern: re.Pattern[str]) -> bool:
    lowered = text.lower()
    for match in pattern.finditer(text):
        prefix = lowered[max(0, match.start() - 18) : match.start()]
        if any(negation in prefix for negation in _NEGATIONS):
            continue
        return True
    return False


def _normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _scope_disclosed(text: str, validation_scope: str) -> bool:
    if validation_scope == "independent_external_validation":
        return bool(_EXTERNAL_VALIDATION_RE.search(text))
    if validation_scope == "independent_holdout_not_external":
        has_holdout = bool(re.search(r"独立留出|内部留出|holdout|internal\s+validation", text, re.IGNORECASE))
        external_is_negated = bool(
            re.search(
                r"(?:不是|并非|非|不能视为|不属于|not|no)\s*(?:独立)?外部验证|not\s+(?:an?\s+)?external\s+validation",
                text,
                re.IGNORECASE,
            )
        )
        return has_holdout and external_is_negated
    return False


def validate_nir_response(
    response_text: str,
    workflow: Mapping[str, Any] | None,
) -> NIRResponseGroundingResult:
    """Validate metric, scope, quality, and artifact claims against one run."""

    if not isinstance(workflow, Mapping) or not response_text.strip():
        return NIRResponseGroundingResult(passed=True, violations=())

    evidence = workflow.get("attempt_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    summary = evidence.get("metrics_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    allowed_metrics = _summary_values(summary)
    violations: list[str] = []
    metric_claims = list(_METRIC_CLAIM_RE.finditer(response_text))

    for claim in metric_claims:
        category = _metric_category(claim.group("label"))
        raw_value = claim.group("value")
        is_percent = bool(claim.group("percent"))
        claimed_value = float(raw_value) / 100 if is_percent else float(raw_value)
        allowed = allowed_metrics.get(category, [])
        tolerance = _claim_tolerance(raw_value, percent=is_percent)
        if not allowed:
            violations.append(f"metric_not_in_current_evidence:{category}")
        elif not any(abs(claimed_value - actual) <= tolerance + 1e-12 for actual in allowed):
            violations.append(f"metric_value_mismatch:{category}")

    validation_scope = str(evidence.get("validation_scope") or "")
    if validation_scope != "independent_external_validation" and _has_positive_marker(
        response_text,
        _EXTERNAL_VALIDATION_RE,
    ):
        violations.append("validation_scope_overclaim:external")
    if _has_positive_marker(response_text, _DEPLOYMENT_READY_RE):
        violations.append("deployment_readiness_overclaim")

    quality_passed = summary.get("passed")
    if quality_passed is not True and _has_positive_marker(response_text, _QUALITY_PASSED_RE):
        violations.append("quality_overclaim:passed")

    approved_model_path = _normalized_path(evidence.get("model_path"))
    for match in _MODEL_PATH_RE.finditer(response_text):
        if _normalized_path(match.group("path")) != approved_model_path:
            violations.append("artifact_path_mismatch:model")

    approved_metrics_path = _normalized_path(evidence.get("metrics_path"))
    for match in _METRICS_PATH_RE.finditer(response_text):
        if _normalized_path(match.group("path")) != approved_metrics_path:
            violations.append("artifact_path_mismatch:metrics")

    has_result_claim = bool(metric_claims or _MODEL_PATH_RE.search(response_text) or _QUALITY_PASSED_RE.search(response_text))
    if evidence and has_result_claim and not _scope_disclosed(response_text, validation_scope):
        violations.append("validation_scope_disclosure_missing")

    deduplicated = tuple(dict.fromkeys(violations))
    return NIRResponseGroundingResult(
        passed=not deduplicated,
        violations=deduplicated,
    )


def _render_metric_summary(summary: Mapping[str, Any]) -> str | None:
    labels = (
        ("R2_val", "R²"),
        ("RPD", "RPD"),
        ("RMSEC", "RMSEC"),
        ("RMSECV", "RMSECV"),
        ("RMSEP", "RMSEP"),
    )
    values = [f"{label}={summary[key]}" for key, label in labels if isinstance(summary.get(key), int | float) and not isinstance(summary.get(key), bool)]
    for section_key, section_label in (("holdout", "留出测试"), ("external", "外部测试")):
        section = summary.get(section_key)
        if not isinstance(section, Mapping):
            continue
        section_values = [f"{key}={value}" for key, value in section.items() if key in {"RMSE", "R2", "RPD", "bias"} and isinstance(value, int | float) and not isinstance(value, bool)]
        if section_values:
            values.append(f"{section_label}({', '.join(section_values)})")
    return "；".join(values) if values else None


def render_grounded_nir_response(
    workflow: Mapping[str, Any],
    *,
    violations: tuple[str, ...] = (),
) -> str:
    """Build a bounded, deterministic response from the current attempt only."""

    evidence = workflow.get("attempt_evidence")
    if not isinstance(evidence, Mapping):
        return "当前工作流尚无可引用的建模证据，因此不能报告模型指标、外部验证或部署结论。请先完成与验证目标一致的建模步骤。"

    summary = evidence.get("metrics_summary")
    summary = summary if isinstance(summary, Mapping) else {}
    scope = str(evidence.get("validation_scope") or "")
    if scope == "independent_external_validation":
        scope_text = "独立外部验证"
        if workflow.get("validation_goal") == "production":
            scope_text += "（仅证明外部验证结果，不等同于生产可部署）"
    else:
        scope_text = "同一数据集独立留出（不是外部验证）"

    lines = [
        "原回答包含未被当前运行证据支持的结论，已由证据门禁改写。",
        "",
        f"- 验证协议：`{evidence.get('protocol') or 'unknown'}`",
        f"- 验证范围：{scope_text}",
        f"- 质量门禁：{'通过' if summary.get('passed') is True else '未通过'}",
    ]
    metric_text = _render_metric_summary(summary)
    if metric_text:
        lines.append(f"- 当前运行指标：{metric_text}")
    if evidence.get("model_path"):
        lines.append(f"- 模型产物：`{evidence['model_path']}`")
    if evidence.get("metrics_path"):
        lines.append(f"- 指标产物：`{evidence['metrics_path']}`")
    if violations:
        lines.append("- 说明：未采用原回答中与当前证据不一致的数值、验证范围或产物声明。")
    return "\n".join(lines)
