"""Knowledge-base retrieval hint generator.

Builds a structured ``knowledge_hint`` dict that tools attach to their return
payload so the LLM has a concrete ``query`` to feed into ``nir_search_knowledge``
instead of having to recall SKILL.md trigger rules.
"""

from __future__ import annotations


# Domains covered by the inline tips docs (docs/soil-tips.md, docs/food-tips.md, ...).
# When the user's domain falls outside this set, the inline rules may not apply, so
# we suggest retrieving from the paper knowledge base instead.
_KNOWN_DOMAINS = frozenset({"food_moisture", "food_protein", "pharma", "feed", "soil", "default"})


def _build_knowledge_hint(
    *,
    domain: str,
    grade: str | None = None,
    passed: bool | None = None,
    attempt: int | None = None,
    r2_val: float | None = None,
    diagnostics: dict | None = None,
) -> dict | None:
    """Construct a structured knowledge-base retrieval hint.

    The hint is returned alongside the tool's primary payload so the LLM has a
    concrete ``query`` to feed into ``nir_search_knowledge`` — much more
    reliable than asking the model to recall SKILL.md trigger rules.

    Returns ``None`` when no hint is warranted.

    Trigger priority (first match wins):
    1. Unknown domain (no inline tips available) — suggest domain overview.
    2. grade ∈ {C, D, F} AND attempt >= 2 — diagnostics-driven query.
    3. R²_val < 0.7 — query typical R²/RPD range for the domain.
    """
    if domain and domain not in _KNOWN_DOMAINS:
        return {
            "should_search": True,
            "query": f"{domain} NIR calibration PLS preprocessing",
            "reason": (
                f"domain='{domain}' 不在已知领域列表内 "
                f"({sorted(_KNOWN_DOMAINS)})，建议先检索论文库了解该领域的常用预处理方法和合理指标范围。"
            ),
        }

    diag = diagnostics or {}
    trend = diag.get("residual_trend", "none")
    variance = diag.get("residual_variance", "none")

    if grade in {"C", "D", "F"} and attempt is not None and attempt >= 2:
        if trend == "upward":
            return {
                "should_search": True,
                "query": f"airpls baseline correction {domain} NIR RPD",
                "reason": (
                    f"重试 {attempt} 次仍未通过门禁 (grade={grade})，残差呈上升趋势，"
                    "建议检索论文库确认该领域的基线校正最佳实践。"
                ),
            }
        if trend == "downward":
            return {
                "should_search": True,
                "query": f"snv vs msc scatter correction {domain} NIR",
                "reason": (
                    f"重试 {attempt} 次仍未通过门禁 (grade={grade})，残差呈下降趋势，"
                    "建议检索论文库确认该领域的散射校正方法选择。"
                ),
            }
        if variance == "high":
            return {
                "should_search": True,
                "query": f"sg_smooth window selection {domain} NIR noise",
                "reason": (
                    f"重试 {attempt} 次仍未通过门禁 (grade={grade})，残差方差高，"
                    "建议检索论文库确认该领域的平滑窗口选择经验。"
                ),
            }
        return {
            "should_search": True,
            "query": f"{domain} NIR PLS improve RPD preprocessing",
            "reason": (
                f"重试 {attempt} 次仍未通过门禁 (grade={grade})，"
                "建议检索论文库寻找该领域提升 RPD 的预处理组合经验。"
            ),
        }

    if r2_val is not None and r2_val < 0.7:
        return {
            "should_search": True,
            "query": f"typical R2 RPD PLS {domain} NIR",
            "reason": (
                f"R²_val={r2_val:.3f} 偏低，建议检索论文库确认 {domain} 领域的合理 R²/RPD 范围，"
                "判断是否数据本身问题还是预处理不当。"
            ),
        }

    return None
