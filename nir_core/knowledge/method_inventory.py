"""Reviewed website inventory and allowlisted live MCP facts, never execution."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

INVENTORY_PATH = Path(__file__).parent / "method_cards" / "chemotools-inventory-v1.json"

CATEGORY_LABELS = {
    "augmentation": "数据增强",
    "baseline": "基线校正",
    "derivative": "导数",
    "smooth": "平滑降噪",
    "scatter": "散射校正",
    "scale": "尺度变换",
    "projection": "正交投影",
    "feature_selection": "变量选择",
    "regression": "回归建模",
    "outliers": "异常诊断",
    "plotting": "绘图与辅助工具",
    "inspector": "模型检查",
    "physics": "物理量转换",
    "adaptation": "迁移与轴对齐",
    "datasets": "示例数据",
    "native": "项目方法",
}


def safe_schema(value):
    """Preserve non-JSON defaults as explicit text instead of invalid NaN JSON."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "default" and isinstance(item, float) and not math.isfinite(item):
                result["default_repr"] = repr(item)
            else:
                result[key] = safe_schema(item)
        return result
    if isinstance(value, list):
        return [safe_schema(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


@lru_cache(maxsize=1)
def inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def mcp_facts(capability_id: str) -> dict:
    """Read only known public capability IDs; website presence is no execution claim."""
    if capability_id not in inventory()["entries"]:
        raise ValueError("method is not in the reviewed Chemotools inventory")
    from nir_core.chemotools_mcp.catalog import (
        ChemotoolsCatalogError,
        describe_capability,
    )

    try:
        entry = describe_capability(capability_id)
    except ChemotoolsCatalogError:
        return {
            "mcp_capability_id": capability_id,
            "mcp_available": False,
            "mcp_execution_supported": False,
            "mcp_parameters": {},
            "mcp_operations": [],
        }
    return {
        "mcp_capability_id": capability_id,
        "mcp_available": True,
        "mcp_execution_supported": entry["execution_supported"],
        "mcp_parameters": safe_schema(entry["parameters"]),
        "mcp_operations": entry["operations"],
        "mcp_kind": entry["kind"],
        "mcp_provider_version": entry["provider_version"],
    }


def enrich_card(card: dict) -> dict:
    capability_id = card.get("mcp_capability_id")
    return {**card, **mcp_facts(capability_id)} if capability_id else card


def coverage(cards: list[dict]) -> dict:
    manifest = inventory()
    ids = {card.get("mcp_capability_id") for card in cards}
    official = set(manifest["official_ids"])
    runtime = set(manifest["mcp_ids"])
    return {
        "official_document_count": len(official),
        "official_covered_count": len(official & ids),
        "mcp_capability_count": len(runtime),
        "mcp_covered_count": len(runtime & ids),
        "documentation_only_ids": sorted(official - runtime),
        "runtime_provider_version": manifest["provider_version"],
        "retrieved_on": manifest["retrieved_on"],
    }
