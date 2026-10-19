"""Offline official-method reference retrieval, separate from paper evidence.

The reviewed, versioned method cards are a small structured RAG corpus. FTS5
BM25 provides lexical retrieval with bilingual problem expansion; runtime
constraints and diagnostic tags gate planning. No embedding service, website
request, user spectrum, or mutable global database is needed during training.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .method_registry import runtime_reference

DEFAULT_CARDS_PATH = (
    Path(__file__).parent / "method_cards" / "chemotools-preprocessing-v1.json"
)

PROBLEM_TERMS = {
    "high_frequency_noise": (
        "noise",
        "noisy",
        "smooth",
        "smoothing",
        "denoise",
        "噪声",
        "降噪",
        "平滑",
        "高频",
    ),
    "low_signal_to_noise": ("low_signal_to_noise", "信噪比低", "低信噪比"),
    "baseline_drift": (
        "baseline",
        "drift",
        "detrend",
        "基线",
        "漂移",
        "背景",
        "去趋势",
    ),
    "scatter_variation": ("scatter", "multiplicative", "散射", "乘性"),
    "isolated_spikes": ("isolated_spikes", "spike", "尖峰", "脉冲"),
    "heavy_tailed_rows": ("heavy_tailed_rows", "heavy tailed", "重尾", "稳健"),
    "broad_overlapping_bands": ("broad_overlapping_bands", "overlapping", "重叠峰"),
    "feature_scale_variation": ("feature_scale_variation", "scaling", "缩放", "尺度"),
    "high_dimensional_collinearity": (
        "high_dimensional_collinearity",
        "collinearity",
        "共线",
        "高维",
    ),
    "nonlinear_response": ("nonlinear_response", "nonlinear regression", "非线性"),
    "regularization_needed": ("regularization_needed", "regularization", "正则化"),
    "data_augmentation": ("augmentation", "数据增强", "加噪"),
    "feature_selection": ("feature selection", "变量选择", "波段筛选", "变量筛选"),
    "outlier_diagnostics": ("outlier", "异常诊断", "异常样品", "离群点"),
    "calibration_transfer": ("calibration transfer", "仪器迁移", "校准转移"),
    "spectral_projection": ("orthogonal", "正交投影", "干扰子空间"),
    "physical_conversion": (
        "absorbance conversion",
        "物理转换",
        "透射率",
        "吸光度转换",
    ),
    "model_diagnostics": ("inspector", "模型检查", "模型诊断"),
    "visualization": ("plotting", "绘图", "可视化"),
    "example_data": ("datasets", "示例数据", "教程数据"),
}
PROBLEM_LABELS = {
    "high_frequency_noise": "高频噪声",
    "low_signal_to_noise": "低信噪比",
    "baseline_drift": "基线漂移",
    "scatter_variation": "散射变化",
    "isolated_spikes": "孤立尖峰",
    "heavy_tailed_rows": "重尾噪声／异常波段",
    "broad_overlapping_bands": "宽峰重叠",
    "feature_scale_variation": "变量尺度差异",
    "high_dimensional_collinearity": "高维与共线性",
    "nonlinear_response": "非线性响应（需验证）",
    "regularization_needed": "需要正则化",
    "data_augmentation": "数据增强（仅训练内）",
    "feature_selection": "变量与波段选择",
    "outlier_diagnostics": "异常样品诊断",
    "calibration_transfer": "仪器迁移与轴对齐",
    "spectral_projection": "正交投影与干扰去除",
    "physical_conversion": "光学物理量转换",
    "model_diagnostics": "模型与流程检查",
    "visualization": "绘图与显示",
    "example_data": "示例数据接口",
}


def _tokens(value: str) -> set[str]:
    value = value.lower()[:2000]
    tokens = set(re.findall(r"[a-z0-9_]+", value))
    for run in re.findall(r"[\u4e00-\u9fff]+", value):
        tokens.update(run[index : index + 2] for index in range(max(1, len(run) - 1)))
    for tag, aliases in PROBLEM_TERMS.items():
        if tag in value or any(alias in value for alias in aliases):
            tokens.add(tag)
    return tokens


def method_eligibility(card: dict, profile: dict | None) -> str | None:
    """Return why a reference cannot propose an automatic runtime method."""
    from nir_core.preprocess.registry import METHOD_REGISTRY

    try:
        binding = runtime_reference(card["method_id"])
    except ValueError:
        return "method_not_available"
    if (
        binding["provider_class"] != card["provider_class"]
        or binding["runtime_provider_version"] != card["runtime_provider_version"]
    ):
        return "runtime_version_mismatch"
    if binding["method_kind"] == "modeling":
        return "model_reference_only"
    if binding["method_kind"] == "mcp_reference":
        if not binding["mcp_available"]:
            return "mcp_unavailable"
        return (
            "mcp_explicit_only"
            if binding["mcp_execution_supported"]
            else "mcp_not_executable"
        )
    spec = METHOD_REGISTRY[card["method_id"]]
    if spec.auto_level not in {"default", "conditional"}:
        return "explicit_only"
    if profile:
        if card["method_id"] in {
            "derivative1",
            "derivative2",
        } and "low_signal_to_noise" in profile.get("tags", []):
            return "diagnostic_contraindication"
        if (
            (card.get("regular_axis_required") or spec.requires_regular_axis)
            and profile.get("has_wavelength_axis")
            and not profile.get("regular_wavelength_axis")
        ):
            return "irregular_wavelength_axis"
        if spec.requires_wavelengths and not profile.get("has_wavelength_axis"):
            return "missing_wavelength_axis"
        if set(card.get("avoid_tags", [])).intersection(profile.get("tags", [])):
            return "diagnostic_contraindication"
    return None


class MethodKnowledgeBase:
    """A per-call immutable reference snapshot with a bounded FTS5 index."""

    def __init__(
        self,
        cards_path: Path | str = DEFAULT_CARDS_PATH,
        *,
        payload: dict | None = None,
    ):
        payload = (
            payload
            if payload is not None
            else json.loads(Path(cards_path).read_text(encoding="utf-8"))
        )
        if (
            payload.get("schema_version") != 1
            or payload.get("source_kind") != "official_method_reference"
        ):
            raise ValueError("Unsupported method-card schema")
        self.index_version = payload["index_version"]
        self.cards = []
        for entry in payload["cards"]:
            card = {**entry}
            card.setdefault("method_kind", "preprocessing")
            card.setdefault("provider", "chemotools")
            for name in (
                "runtime_provider_version",
                "retrieved_on",
                "documentation_version",
                "parameter_policy",
            ):
                card.setdefault(name, payload[name])
            # Hash is of the reviewed card, not a claim to hash the live page.
            encoded = json.dumps(
                card, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            card["content_sha256"] = hashlib.sha256(encoded).hexdigest()
            card["evidence_id"] = (
                f"method:{card['method_id']}:{card['content_sha256'][:16]}"
            )
            self.cards.append(card)

    def search(
        self, query: str, *, top_k: int = 6, profile: dict | None = None
    ) -> dict[str, Any]:
        """Retrieve reviewed method facts; rank scores are not probabilities."""

        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1 to 2000 characters")
        if not 1 <= top_k <= 12:
            raise ValueError("top_k must be between 1 and 12")
        tokens = _tokens(query)
        exclusions = []
        ranked = []
        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE methods USING fts5(card_index UNINDEXED, content)"
            )
            for index, card in enumerate(self.cards):
                if card.get("review_status") != "published":
                    continue
                text = " ".join(
                    [
                        card["method_id"],
                        card["title"],
                        card.get("mcp_capability_id", ""),
                        card["keywords"],
                        *card["problem_tags"],
                    ]
                )
                connection.execute(
                    "INSERT INTO methods VALUES (?, ?)",
                    (index, " ".join(sorted(_tokens(text)))),
                )
            match = " OR ".join(f'"{token}"' for token in sorted(tokens))
            rows = (
                connection.execute(
                    "SELECT card_index, bm25(methods) FROM methods WHERE methods MATCH ? ORDER BY bm25(methods), card_index",
                    (match,),
                ).fetchall()
                if match
                else []
            )
            for index, score in rows:
                card = dict(self.cards[int(index)])
                if profile and card["method_kind"] != "preprocessing":
                    continue
                reason = method_eligibility(card, profile)
                if reason in {"method_not_available", "runtime_version_mismatch"}:
                    exclusions.append({"method": card["method_id"], "reason": reason})
                    continue
                matched_tags = (
                    sorted(
                        set(card["problem_tags"]).intersection(profile.get("tags", []))
                    )
                    if profile
                    else []
                )
                if profile and not matched_tags:
                    continue
                relevance = sum(
                    {
                        "high_frequency_noise": 10,
                        "low_signal_to_noise": 2,
                        "baseline_drift": 5,
                        "scatter_variation": 4,
                    }.get(tag, 1)
                    for tag in matched_tags
                )
                binding = runtime_reference(card["method_id"])
                from .method_inventory import enrich_card

                card = enrich_card(card)
                card.update(
                    auto_eligible=reason is None,
                    eligibility_reason=reason,
                    matched_tags=matched_tags,
                    runtime_parameters=binding["runtime_parameters"],
                    rank_score=round(-float(score), 8),
                    evidence_kind="official_method_reference",
                    untrusted_evidence=True,
                )
                ranked.append(
                    (
                        (
                            -relevance,
                            -int(
                                profile is None
                                and (
                                    query.strip().casefold()
                                    == card["method_id"].casefold()
                                    or query.strip().casefold()
                                    in card["title"].casefold()
                                    or query.strip().casefold()
                                    in card["keywords"].casefold()
                                )
                            ),
                            int(card.get("priority", 99)),
                            float(score),
                            int(index),
                        ),
                        card,
                    )
                )
                if reason:
                    exclusions.append({"method": card["method_id"], "reason": reason})
        finally:
            connection.close()
        ranked.sort(key=lambda row: row[0])
        results = [card for _, card in ranked[:top_k]]
        return {
            "query": query,
            "index_version": self.index_version,
            "retrieval_strategy": "structured_fts5_bm25",
            "evidence_kind": "official_method_reference",
            "independent_study_count": 0,
            "abstained": not bool(results),
            "reason": "method_references_retrieved"
            if results
            else "no_relevant_method_reference",
            "results": results,
            "count": len(results),
            "exclusions": exclusions,
        }


def retrieve_method_plan(
    profile: dict, *, knowledge_base: MethodKnowledgeBase | None = None
) -> dict:
    """Create a cited plan solely from the already computed calibration profile."""
    tags = [tag for tag in profile["tags"] if tag in PROBLEM_TERMS]
    base = {
        "profile_scope": "calibration_only",
        "problem_labels": [
            PROBLEM_LABELS[tag] for tag in profile["tags"] if tag in PROBLEM_LABELS
        ],
        "parameter_policy": "project_bounded_search_not_official_optimum",
        "independent_study_count": 0,
    }
    if not tags:
        return {
            **base,
            "mode": "rule_fallback",
            "reason": "no_actionable_diagnostic_tags",
            "results": [],
            "exclusions": [],
        }
    try:
        result = (knowledge_base or MethodKnowledgeBase()).search(
            " ".join(tags), top_k=12, profile=profile
        )
        eligible = any(item["auto_eligible"] for item in result["results"])
        return {
            **base,
            **result,
            "mode": "retrieval_guided" if eligible else "rule_fallback",
            "reason": result["reason"]
            if eligible or not result["results"]
            else "no_executable_reference_candidates",
        }
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        return {
            **base,
            "mode": "rule_fallback",
            "reason": "method_retrieval_unavailable",
            "results": [],
            "exclusions": [],
        }
