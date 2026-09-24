"""Deterministic grounding rules for NIR final responses."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MANDATORY_WORKFLOW_ACTIONS = {
    "data_audit": {
        "inspect_data",
    },
    "planning": {
        "prepare_analysis_plan",
        "prepare_retry_plan",
    },
    "execution": {
        "execute_nir_tools",
    },
    "evaluation": {
        "reflect_on_attempt",
    },
    "knowledge": {
        "retrieve_evidence_for_retry",
    },
}
_CONSTANT_COLUMNS_REMOVED_RE = re.compile(
    r"(?:恒定|常量|无变异|constant).{0,24}(?:波长|光谱|列).{0,24}"
    r"(?:已|被|自动)?(?:排除|删除|移除|剔除|removed|excluded|dropped)",
    re.IGNORECASE,
)
_LITERATURE_COMPARISON_RE = re.compile(
    r"(?:与|同).{0,8}(?:文献|论文|研究).{0,16}(?:一致|相符|吻合)"
    r"|(?:文献|论文|研究).{0,24}(?:典型范围|预期表现)"
    r"|(?:consistent\s+with|matches?).{0,16}(?:the\s+)?literature"
    r"|typical\s+(?:literature\s+)?range",
    re.IGNORECASE,
)
_WAVELENGTH_RANGE_RE = re.compile(
    r"(?P<label>原始(?:采集)?(?:光谱|波长)?范围|实际可用(?:光谱|波长)?范围|"
    r"可用(?:光谱|波长)?范围|raw\s+(?:spectral|wavelength)?\s*range|"
    r"usable\s+(?:spectral|wavelength)?\s*range)"
    r"[^0-9+-]{0,16}(?P<low>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:–|—|-|~|至|到)\s*(?P<high>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
    r"(?:nm|纳米)",
    re.IGNORECASE,
)
_WAVELENGTH_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s*个\s*(?P<label>非恒定|可用|恒定|常量|无变异)\s*"
    r"(?:波长|光谱(?:列)?|列)",
    re.IGNORECASE,
)
_WAVELENGTH_KEY_COUNT_RE = re.compile(
    r"(?P<label>constant_wavelength_count|usable_wavelength_count)\s*=\s*(?P<count>\d+)",
    re.IGNORECASE,
)
_WAVELENGTH_LABEL_COUNT_RE = re.compile(
    r"(?P<label>非恒定|可用|恒定|常量|无变异)\s*(?:波长|光谱(?:列)?|列)\s*"
    r"(?P<count>\d+)\s*个",
    re.IGNORECASE,
)
_NO_CONSTANT_WAVELENGTH_RE = re.compile(
    r"(?:无|没有|不存在)\s*(?:恒定|常量|无变异)\s*(?:波长|光谱列)",
    re.IGNORECASE,
)
_AXIS_DIRECTION_RE = re.compile(
    r"(?:(?:光谱|波长|波数)?轴(?:方向|顺序)|axis(?:[_\s-]*direction)?)"
    r"[^。\n]{0,64}?(?P<direction>非单调|升序|降序|递增|递减|"
    r"ascending|descending|non[_\s-]*monotonic)",
    re.IGNORECASE,
)
_WORKFLOW_STAGE_RE = re.compile(
    r"\bstage\s*(?:[:=]|：|为|是)\s*`?(?P<value>[a-z_]+)",
    re.IGNORECASE,
)
_WORKFLOW_NEXT_ACTION_RE = re.compile(
    r"\bnext_action\s*(?:[:=]|：|为|是)\s*`?(?P<value>[a-z_]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NIRResponseGroundingResult:
    """Explainable result from validating one natural-language response."""

    passed: bool
    violations: tuple[str, ...]


def required_nir_workflow_action(
    workflow: Mapping[str, Any] | None,
) -> str | None:
    """Return the mandatory next action that prevents a final response."""

    if not isinstance(workflow, Mapping):
        return None
    stage = str(workflow.get("stage") or "")
    next_action = str(workflow.get("next_action") or "")
    if next_action in _MANDATORY_WORKFLOW_ACTIONS.get(stage, set()):
        return next_action
    return None


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
_CLASSIFICATION_METRIC_CLAIM_RE = re.compile(
    r"(?P<label>balanced[\s_-]*accuracy|macro[\s_-]*F1|MCC|"
    r"min(?:imum)?[\s_-]*class[\s_-]*recall|平衡准确率|宏(?:平均)?F1|"
    r"马修斯相关系数|最(?:低|小)类别召回率)"
    r"\s*(?:[:=]|is|of|为|是|约为|达到|达)?\s*"
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
    if normalized in {"balancedaccuracy", "平衡准确率"}:
        return "balanced_accuracy"
    if normalized in {"macrof1", "宏f1", "宏平均f1"}:
        return "macro_f1"
    if normalized in {"mcc", "马修斯相关系数"}:
        return "mcc"
    if normalized in {
        "minclassrecall",
        "minimumclassrecall",
        "最低类别召回率",
        "最小类别召回率",
    }:
        return "class_recall"
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
        if key in {"minr2", "r2threshold"}:
            add("r2", value)
        elif key in {"minrpd", "rpdthreshold"}:
            add("rpd", value)
        elif key.startswith("r2"):
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
        elif key == "balancedaccuracy":
            add("balanced_accuracy", value)
        elif key == "macrof1":
            add("macro_f1", value)
        elif key == "mcc":
            add("mcc", value)
        elif key in {"minimumclassrecall", "minclassrecall"}:
            add("class_recall", value)
        elif key == "recall" and "perclass" in {re.sub(r"[\s_^虏-]+", "", item).lower() for item in path[:-1]}:
            add("class_recall", value)

    walk(summary)
    return values


def _attempt_evidence_records(workflow: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the bounded attempt ledger plus the latest evidence if needed."""

    records = [item for item in (workflow.get("attempts") or []) if isinstance(item, Mapping)]
    latest = workflow.get("attempt_evidence")
    if isinstance(latest, Mapping):
        latest_path = _normalized_path(latest.get("metrics_path"))
        already_present = any(latest_path and _normalized_path(item.get("metrics_path")) == latest_path for item in records)
        if not already_present:
            records.append(latest)
    return records[-10:]


def _allowed_metric_values(records: list[Mapping[str, Any]]) -> dict[str, list[float]]:
    allowed: dict[str, list[float]] = {}
    for record in records:
        summary = record.get("metrics_summary")
        if not isinstance(summary, Mapping):
            continue
        for category, values in _summary_values(summary).items():
            allowed.setdefault(category, []).extend(values)
    return allowed


def _claim_tolerance(raw_value: str, *, percent: bool) -> float:
    mantissa = re.split(r"[eE]", raw_value, maxsplit=1)[0]
    decimals = len(mantissa.split(".", maxsplit=1)[1]) if "." in mantissa else 0
    tolerance = 0.5 * (10**-decimals)
    return tolerance / 100 if percent else tolerance


def _has_positive_marker(text: str, pattern: re.Pattern[str]) -> bool:
    lowered = text.lower()
    for match in pattern.finditer(text):
        local_context = lowered[max(0, match.start() - 18) : min(len(lowered), match.end() + 1)]
        if any(negation in local_context for negation in _NEGATIONS):
            continue
        return True
    return False


def _normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def _numeric_pair(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        return None
    return float(value[0]), float(value[1])


def _normalized_axis_direction(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"升序", "递增", "ascending"}:
        return "ascending"
    if normalized in {"降序", "递减", "descending"}:
        return "descending"
    if normalized in {"非单调", "non_monotonic", "nonmonotonic"}:
        return "non_monotonic"
    return normalized


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
    attempt_records = _attempt_evidence_records(workflow)
    allowed_metrics = _allowed_metric_values(attempt_records)
    violations: list[str] = []
    required_action = required_nir_workflow_action(workflow)
    if required_action:
        violations.append(f"workflow_incomplete:{required_action}")
    metric_claims = [
        *_METRIC_CLAIM_RE.finditer(response_text),
        *_CLASSIFICATION_METRIC_CLAIM_RE.finditer(response_text),
    ]

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
    validation_scopes = {str(record.get("validation_scope") or validation_scope) for record in attempt_records if record.get("validation_scope") or validation_scope}
    if "independent_external_validation" not in validation_scopes and _has_positive_marker(
        response_text,
        _EXTERNAL_VALIDATION_RE,
    ):
        violations.append("validation_scope_overclaim:external")
    if _has_positive_marker(response_text, _DEPLOYMENT_READY_RE):
        violations.append("deployment_readiness_overclaim")

    quality_passed = any(isinstance(record.get("metrics_summary"), Mapping) and record["metrics_summary"].get("passed") is True for record in attempt_records)
    if quality_passed is not True and _has_positive_marker(response_text, _QUALITY_PASSED_RE):
        violations.append("quality_overclaim:passed")

    approved_model_paths = {_normalized_path(record.get("model_path")) for record in attempt_records if _normalized_path(record.get("model_path"))}
    for match in _MODEL_PATH_RE.finditer(response_text):
        if _normalized_path(match.group("path")) not in approved_model_paths:
            violations.append("artifact_path_mismatch:model")

    approved_metrics_paths = {_normalized_path(record.get("metrics_path")) for record in attempt_records if _normalized_path(record.get("metrics_path"))}
    for match in _METRICS_PATH_RE.finditer(response_text):
        if _normalized_path(match.group("path")) not in approved_metrics_paths:
            violations.append("artifact_path_mismatch:metrics")

    has_result_claim = bool(metric_claims or _MODEL_PATH_RE.search(response_text) or _QUALITY_PASSED_RE.search(response_text))
    scope_is_disclosed = any(_scope_disclosed(response_text, scope) for scope in validation_scopes)
    if evidence and has_result_claim and not scope_is_disclosed:
        violations.append("validation_scope_disclosure_missing")

    result_facts = evidence.get("result_facts")
    result_facts = result_facts if isinstance(result_facts, Mapping) else {}
    selection = result_facts.get("wavelength_selection")
    selection = selection if isinstance(selection, Mapping) else {}
    original_count = selection.get("n_original")
    selected_count = selection.get("n_selected")
    if _CONSTANT_COLUMNS_REMOVED_RE.search(response_text) and isinstance(original_count, int | float) and isinstance(selected_count, int | float) and selected_count >= original_count:
        violations.append("unsupported_preprocessing_claim:constant_columns_removed")

    knowledge_evidence = workflow.get("knowledge_evidence")
    if _LITERATURE_COMPARISON_RE.search(response_text) and not (isinstance(knowledge_evidence, list) and any(str(item).strip() for item in knowledge_evidence)):
        violations.append("knowledge_claim_without_evidence")

    audit_evidence = workflow.get("audit_evidence")
    audit_evidence = audit_evidence if isinstance(audit_evidence, Mapping) else {}
    expected_ranges = {
        "raw": _numeric_pair(audit_evidence.get("raw_wavelength_range")),
        "usable": _numeric_pair(audit_evidence.get("usable_wavelength_range")),
    }
    for claim in _WAVELENGTH_RANGE_RE.finditer(response_text):
        label = claim.group("label").lower()
        category = "usable" if "可用" in label or "usable" in label else "raw"
        expected = expected_ranges[category]
        claimed = float(claim.group("low")), float(claim.group("high"))
        if expected is None:
            violations.append(f"wavelength_range_not_in_audit_evidence:{category}")
        elif any(abs(actual - stated) > 1e-9 for actual, stated in zip(expected, claimed, strict=True)):
            violations.append(f"wavelength_range_mismatch:{category}")

    expected_counts = {
        "constant": audit_evidence.get("constant_wavelength_count"),
        "usable": audit_evidence.get("usable_wavelength_count"),
    }
    count_claims: list[tuple[str, int]] = []
    for claim in _WAVELENGTH_COUNT_RE.finditer(response_text):
        label = claim.group("label").lower()
        category = "usable" if label in {"非恒定", "可用"} else "constant"
        count_claims.append((category, int(claim.group("count"))))
    for claim in _WAVELENGTH_KEY_COUNT_RE.finditer(response_text):
        category = "usable" if claim.group("label").lower().startswith("usable") else "constant"
        count_claims.append((category, int(claim.group("count"))))
    for claim in _WAVELENGTH_LABEL_COUNT_RE.finditer(response_text):
        label = claim.group("label").lower()
        category = "usable" if label in {"非恒定", "可用"} else "constant"
        count_claims.append((category, int(claim.group("count"))))
    if _NO_CONSTANT_WAVELENGTH_RE.search(response_text):
        count_claims.append(("constant", 0))
    for category, claimed_count in count_claims:
        expected_count = expected_counts[category]
        if isinstance(expected_count, bool) or not isinstance(expected_count, int | float):
            violations.append(f"wavelength_count_not_in_audit_evidence:{category}")
        elif claimed_count != int(expected_count):
            violations.append(f"wavelength_count_mismatch:{category}")

    expected_direction = _normalized_axis_direction(audit_evidence.get("axis_direction"))
    for claim in _AXIS_DIRECTION_RE.finditer(response_text):
        claimed_direction = _normalized_axis_direction(claim.group("direction"))
        if not expected_direction:
            violations.append("axis_direction_not_in_audit_evidence")
        elif claimed_direction != expected_direction:
            violations.append("axis_direction_mismatch")

    expected_stage = str(workflow.get("stage") or "")
    for claim in _WORKFLOW_STAGE_RE.finditer(response_text):
        if claim.group("value").lower() != expected_stage.lower():
            violations.append("workflow_stage_mismatch")
    expected_next_action = str(workflow.get("next_action") or "")
    for claim in _WORKFLOW_NEXT_ACTION_RE.finditer(response_text):
        if claim.group("value").lower() != expected_next_action.lower():
            violations.append("workflow_next_action_mismatch")

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
    holdout = summary.get("holdout")
    if isinstance(holdout, Mapping):
        classification_labels = (
            ("balanced_accuracy", "balanced accuracy"),
            ("macro_f1", "macro-F1"),
            ("mcc", "MCC"),
        )
        values.extend(f"{label}={holdout[key]}" for key, label in classification_labels if isinstance(holdout.get(key), int | float) and not isinstance(holdout.get(key), bool))
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
    """Build a bounded, deterministic response from the attempt ledger."""

    audit_evidence = workflow.get("audit_evidence")
    audit_evidence = audit_evidence if isinstance(audit_evidence, Mapping) else {}
    if workflow.get("task_type") == "inspection" and audit_evidence:
        lines = ["数据检查已完成（未执行预处理或建模）。"]
        lines.append(f"- 工作流状态：`{workflow.get('stage') or 'unknown'}` / `{workflow.get('next_action') or 'unknown'}`")
        data_path = str(audit_evidence.get("data_path") or workflow.get("data_path") or "")
        if data_path:
            lines.append(f"- 文件：`{data_path}`")
        format_name = audit_evidence.get("format")
        layout = audit_evidence.get("layout_pattern")
        if format_name or layout:
            lines.append(f"- 格式与布局：{format_name or 'unknown'} / {layout or 'unknown'}")
        n_samples = audit_evidence.get("n_samples")
        n_wavelengths = audit_evidence.get("n_wavelengths")
        if isinstance(n_samples, int) and isinstance(n_wavelengths, int):
            lines.append(f"- 数据规模：{n_samples} 个样本，{n_wavelengths} 个光谱变量")
        axis_first = audit_evidence.get("axis_first")
        axis_last = audit_evidence.get("axis_last")
        axis_direction = _normalized_axis_direction(audit_evidence.get("axis_direction"))
        direction_labels = {
            "ascending": "升序",
            "descending": "降序",
            "non_monotonic": "非单调",
            "duplicate_values": "含重复值",
            "single_point": "单点",
            "invalid": "无效",
        }
        if isinstance(axis_first, int | float) and not isinstance(axis_first, bool) and isinstance(axis_last, int | float) and not isinstance(axis_last, bool):
            direction_label = direction_labels.get(axis_direction, axis_direction)
            lines.append(f"- 光谱轴方向：{float(axis_first)} → {float(axis_last)}（{direction_label}）")
        raw_range = _numeric_pair(audit_evidence.get("raw_wavelength_range"))
        if raw_range:
            lines.append(f"- 光谱轴数值范围：{raw_range[0]}–{raw_range[1]}")
        if isinstance(audit_evidence.get("has_nan"), bool):
            lines.append(f"- 缺失值：{'检测到' if audit_evidence['has_nan'] else '未检测到'}")
        return "\n".join(lines)

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
    raw_range = _numeric_pair(audit_evidence.get("raw_wavelength_range"))
    usable_range = _numeric_pair(audit_evidence.get("usable_wavelength_range"))
    if raw_range:
        lines.append(f"- 原始采集光谱范围：{raw_range[0]}–{raw_range[1]} nm")
    if usable_range:
        lines.append(f"- 实际可用光谱范围：{usable_range[0]}–{usable_range[1]} nm")
    constant_count = audit_evidence.get("constant_wavelength_count")
    usable_count = audit_evidence.get("usable_wavelength_count")
    if isinstance(constant_count, int) and isinstance(usable_count, int):
        lines.append(f"- 光谱列：恒定波长列 {constant_count} 个；可用波长 {usable_count} 个（范围说明不代表已删除恒定列）")
    attempts = _attempt_evidence_records(workflow)
    for index, attempt in enumerate(attempts, start=1):
        attempt_number = attempt.get("attempt") or index
        attempt_summary = attempt.get("metrics_summary")
        attempt_summary = attempt_summary if isinstance(attempt_summary, Mapping) else {}
        metric_text = _render_metric_summary(attempt_summary)
        status = "通过" if attempt.get("passed") is True or attempt_summary.get("passed") is True else "未通过"
        attempt_line = f"- 尝试 {attempt_number}：质量门禁{status}"
        if metric_text:
            attempt_line += f"；{metric_text}"
        lines.append(attempt_line)
        if attempt.get("model_path"):
            lines.append(f"  - 模型产物：`{attempt['model_path']}`")
        if attempt.get("metrics_path"):
            lines.append(f"  - 指标产物：`{attempt['metrics_path']}`")
    if violations:
        lines.append("- 说明：未采用原回答中与当前证据不一致的数值、验证范围或产物声明。")
    return "\n".join(lines)
