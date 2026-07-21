"""Markdown report builder for NIR analysis tools.

Generates a compact Chinese-language analysis report that references the
plot files written beside it.  Keeping binary PNG payloads out of Markdown
prevents report reads from flooding the model context.
"""

from __future__ import annotations


def _build_report(
    data,
    metrics: dict,
    quality: dict,
    best_pipe,
    raw_spectra_b64: str = "",
    predicted_vs_reference_b64: str = "",
    residuals_b64: str = "",
    cv_curve_b64: str = "",
) -> str:
    """Build a compact Chinese Markdown analysis report with plot references."""
    pp = best_pipe.description() if best_pipe else "无（使用原始光谱）"
    model_decision = metrics.get("model_selection_decision")
    model_decision_lines: list[str] = []
    if isinstance(model_decision, dict):
        candidates = metrics.get("model_candidates", [])
        compared = ", ".join(str(item.get("method")) for item in candidates if isinstance(item, dict))
        model_decision_lines = [
            f"- 模型选择模式: {model_decision.get('mode', '-')}",
            f"- 模型选择原因: {model_decision.get('reason', '-')}",
            f"- 候选模型: {compared or '-'}",
            f"- 最终模型: {model_decision.get('selected_method', metrics.get('method', '-'))}",
        ]
    lines = [
        "# 近红外光谱分析报告",
        "",
        "> 本报告由 NIR 建模工具自动生成。图表作为同目录独立文件保存，避免在报告中嵌入大型 Base64 数据。",
        "",
        "## 1. 数据概览",
        f"- 样本数: {metrics['n_samples']}",
        f"- 波长数: {data.X.shape[1]}",
        (f"- 波长范围: {float(data.wv.min()):.1f} - {float(data.wv.max()):.1f} nm" if data.wv is not None else "- 波长范围: 未提供"),
        (f"- 参考值范围: {float(data.y.min()):.3f} - {float(data.y.max()):.3f}" if data.y is not None else ""),
        "",
        "## 2. 预处理方法",
        f"- 选定流水线: {pp}",
        "",
        "## 3. 模型结果",
        f"- 方法: {metrics['method']}",
        f"- 成分数: {metrics['n_components']}",
        f"- R²(验证集): {metrics['R2_val']:.4f}",
        f"- RPD(测试集): {metrics['RPD']:.4f}",
        f"- RMSEP(测试集): {metrics['RMSEP']:.4f}",
        *model_decision_lines,
        "",
        "## 4. 质量评估",
        f"- 等级: {quality['grade']}",
        f"- 是否通过: {'是' if quality['passed'] else '否'}",
        f"- 行动建议: {quality['action']}",
        f"- 使用阈值: R²≥{quality['thresholds_used']['min_r2']}, RPD≥{quality['thresholds_used']['min_rpd']}",
        f"- 领域: {quality['thresholds_used']['domain']}",
        "",
        "## 5. 可视化",
    ]

    def _plot_reference(title: str, b64: str, filename: str) -> list[str]:
        if not b64:
            return []
        return [
            f"### {title}",
            "",
            f"- 图表文件: `{filename}`",
            "",
        ]

    lines.extend(_plot_reference("原始光谱", raw_spectra_b64, "raw_spectra.png"))
    lines.extend(_plot_reference("预测 vs 参考值（测试集）", predicted_vs_reference_b64, "predicted_vs_reference.png"))
    lines.extend(_plot_reference("残差诊断", residuals_b64, "residuals.png"))
    lines.extend(_plot_reference("交叉验证选成分", cv_curve_b64, "cv_curve.png"))

    lines.extend(
        [
            "## 6. 部署建议",
        ]
    )
    if quality["passed"]:
        lines.append("- 模型质量达标，可用于预测。建议定期做漂移检测。")
    elif quality["action"] == "retry_preprocessing":
        lines.append("- 模型质量未达标，建议尝试不同预处理组合（由协调器反思闭环驱动）。")
    elif quality["action"] == "investigate_data":
        lines.append("- 模型质量较差，建议检查数据质量、异常样本和样本代表性。")
    else:
        lines.append("- 模型质量一般，谨慎使用并持续优化。")
    lines.append("")
    return "\n".join(lines)
