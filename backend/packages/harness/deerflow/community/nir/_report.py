"""Evidence-grounded Chinese reports; Markdown never contains binary payloads."""

from __future__ import annotations

import math


def _text(value) -> str:
    return str(value if value is not None else "未记录").replace("|", "\\|").replace("\n", " ").replace("\r", " ")[:800]


def _number(value) -> str:
    try:
        return f"{float(value):.4f}" if math.isfinite(float(value)) else "未记录"
    except (TypeError, ValueError):
        return "未记录"


def report_context(runtime, metrics: dict, counts: dict, *, final_fit: str = "校准集") -> dict:
    """Add descriptive facts without modifying model/metrics bindings."""
    state = getattr(runtime, "state", None)
    workflow = state.get("nir_workflow") or {} if isinstance(state, dict) else {}
    if not isinstance(workflow, dict):
        workflow = {}
    return {**metrics, "report_context": {"target": workflow.get("analyte"), "unit": workflow.get("unit"), "partition_counts": counts, "final_fit": final_fit}}


def _build_report(data, metrics: dict, quality: dict, best_pipe, raw_spectra_b64: str = "", predicted_vs_reference_b64: str = "", residuals_b64: str = "", cv_curve_b64: str = "", *, extra_plots: dict | None = None) -> str:
    """Describe actual scopes, choices, diagnostics and limitations with figures."""
    context = metrics.get("report_context") or {}
    target = metrics.get("target") or context.get("target")
    unit = context.get("unit") or "单位未记录"
    pp = metrics.get("preprocessing") or (best_pipe.description() if best_pipe else "无（使用原始光谱）")
    external = metrics.get("validation_scope") == "independent_external_validation"
    holdout = "独立外部验证集" if external else "最终留出集"
    validation = "独立外部验证（按记录的来源与划分）" if external else "内部独立留出，不是外部验证"
    if not metrics.get("validation_scope"):
        validation = "验证范围未记录，不能据此声称外部验证"
    test = metrics.get("test") or {"RMSE": metrics.get("RMSEP"), "RPD": metrics.get("RPD")}
    val = {"R2": metrics.get("R2_val"), **(metrics.get("val") or {})}
    counts = context.get("partition_counts") or {}
    lines = [
        "# 近红外光谱建模报告",
        "",
        "## 1. 结论摘要",
        f"- 分析目标：{_text(target)}；参考值单位：{_text(unit)}。",
        f"- 最终方案：{_text(metrics.get('method'))}；预处理：{_text(pp)}；成分数：{_text(metrics.get('n_components'))}。",
        f"- {holdout}：R²={_number(test.get('R2'))}，RMSEP={_number(test.get('RMSE'))}，RPD={_number(test.get('RPD'))}。误差单位与参考值一致。",
        f"- 验证范围：{validation}；质量门禁{'通过' if quality.get('passed') else '未通过'}（{_text(quality.get('grade'))}）。",
        "- 质量达标不等同于生产批准；本报告不代表已完成审批、注册或实际部署。",
        "",
        "## 2. 数据概览与验证协议",
        f"- 样本数：{_text(metrics.get('n_samples'))}；原始光谱变量数：{data.X.shape[1]}。",
        f"- 协议：{_text(metrics.get('protocol'))}。",
    ]
    if data.wv is not None:
        axis_unit = metrics.get("wavelength_unit") or "单位未记录"
        lines.append(f"- 光谱轴范围：{_number(data.wv.min())}–{_number(data.wv.max())}（{_text(axis_unit)}）。")
    else:
        lines.append("- 光谱轴：未提供，图表按变量序号展示。")
    if data.y is not None:
        lines.append(f"- 参考值范围：{_number(data.y.min())}–{_number(data.y.max())}（{_text(unit)}）。")
    lines += [
        "",
        "| 分区 | 样本数 | 用途 |",
        "|---|---:|---|",
        f"| 校准集 | {_text(counts.get('calibration'))} | 拟合预处理、变量筛选和候选模型 |",
        f"| 调参集 | {_text(counts.get('tuning'))} | 选择预处理、变量和模型方案 |",
        f"| {holdout} | {_text(counts.get('holdout'))} | 选择完成后的最终评估 |",
        "",
        f"- 最终拟合范围：{_text(context.get('final_fit'))}。训练拟合指标不能作为独立验证结果。",
        "- 模型选择使用调参/CV证据；最终留出/外部验证指标不用于选择候选。",
        "",
        "## 3. 预处理、波长与模型选择依据",
        f"- 选定流水线：{_text(pp)}。",
    ]
    selection = metrics.get("wavelength_selection") or {}
    lines.append(
        f"- 波长选择：{_text(selection.get('method', '未记录'))}；保留 {_text(selection.get('n_selected', metrics.get('n_wavelengths_model')))} / {_text(selection.get('n_original', metrics.get('n_wavelengths_original')))} 个变量。"
    )
    reasons = {"pls_baseline_sufficient": "PLS基线已达到运行时比较规则，无需扩展模型族。", "cars_improvement_below_threshold": "CARS在调参集上的改善未达到采用阈值，保留全谱。"}
    for key, label in [("wavelength_selection_decision", "波长选择"), ("model_selection_decision", "模型选择")]:
        decision = metrics.get(key) or {}
        if decision:
            lines += [f"- {label}模式: {_text(decision.get('mode'))}。", f"- {label}依据：{reasons.get(decision.get('reason_code')) or _text(decision.get('reason'))}"]
            adoption = decision.get("adoption") or {}
            if adoption:
                lines.append(f"- 采用判断（原始记录）：{_text(adoption.get('reason_code'))}；相对改善={_number(adoption.get('relative_RMSE_improvement'))}；采用阈值={_number(adoption.get('minimum_required_improvement'))}。")
    candidates = metrics.get("model_candidates") or []
    if candidates:
        lines += ["", "| 已执行候选模型 | 成分数 | 调参RMSE（越小越好） |", "|---|---:|---:|"]
        for item in candidates[:30]:
            lines.append(f"| {_text(item.get('method'))} | {_text(item.get('n_components'))} | {_number(item.get('RMSE_tuning'))} |")
        if len(candidates) > 30:
            lines.append(f"- 表格显示前30项，其余 {len(candidates) - 30} 项见 metrics.json。")
    else:
        lines.append("- 未记录模型族比较；不能由当前方案推断其他方法已经执行或较差。")
    wavelength_candidates = metrics.get("wavelength_selection_candidates") or metrics.get("candidate_results") or []
    if wavelength_candidates:
        lines += ["", "| 已执行变量方案 | 保留变量数 | 调参RMSE |", "|---|---:|---:|"]
        for item in wavelength_candidates[:30]:
            lines.append(f"| {_text(item.get('method'))} | {_text(item.get('n_selected'))} | {_number(item.get('RMSE_tuning'))} |")
    preprocessing = metrics.get("preprocessing_selection") or {}
    knowledge = preprocessing.get("recommendation", {}).get("method_knowledge") or {}
    if knowledge:
        guided = knowledge.get("mode") == "retrieval_guided"
        lines += [
            "",
            "### 数据诊断与算法知识依据",
            "",
            f"- 校准数据诊断：{_text('、'.join(knowledge.get('problem_labels', [])) or '未发现可检索的问题标签')}。",
            f"- 首轮策略：{'根据方法知识检索优先安排候选，原始光谱作为同批对照' if guided else '使用规则候选作为回退'}。",
            f"- 检索状态：{_text(knowledge.get('reason'))}；知识索引：{_text(knowledge.get('index_version'))}。",
            "- 方法参考用于提出候选；实际采用仍按校准内部RMSECV与1%简洁性规则判断，不代表文档证明该方法最优。",
            "- 官网默认值、官网约束与本项目有限搜索值分开记录。实际参数必须通过运行时校验；完整方法卡片及校验记录见 metrics.json。",
            "- 数据诊断和方法检索仅使用校准数据；状态拟合在交叉验证每一折重做；最终留出集不参与候选或参数选择。",
        ]
        if knowledge.get("results"):
            lines += ["", "| 方法参考 | 文档说明 | 本项目使用边界 | 来源 |", "|---|---|---|---|"]
            for item in knowledge["results"][:12]:
                lines.append(f"| {_text(item.get('title'))} | {_text(item.get('summary_zh'))} | {_text(item.get('planning_notes_zh'))} | [官方文档]({_text(item.get('source_url'))}) |")
            lines.append("- 以上属于官方方法参考，同站不同页面不作为独立研究证据；不替代论文证据门槛。")
    evaluation = preprocessing.get("evaluation") or {}
    preprocessing_candidates = evaluation.get("candidates") or {}
    if isinstance(preprocessing_candidates, dict) and preprocessing_candidates:
        lines += ["", "| 已执行预处理候选 | 校准数据RMSECV |", "|---|---:|"]
        for label, item in list(preprocessing_candidates.items())[:30]:
            lines.append(f"| {_text(label)} | {_number(item.get('cv_rmse', item.get('rmsecv', item.get('mean_rmse_cv'))))} |")
        lines.append("- 预处理按记录的RMSECV与简洁性规则选择；完整候选与参数见 metrics.json。")
    lines += ["", "## 4. 分区指标与质量评估", "", "| 评估分区 | R² | RMSE | RPD | 偏差（预测−参考） |", "|---|---:|---:|---:|---:|"]
    for label, values in [("最终拟合数据（非独立验证）", metrics.get("train") or {}), ("调参集（参与选择）", val), (holdout, test)]:
        lines.append(f"| {label} | {_number(values.get('R2'))} | {_number(values.get('RMSE'))} | {_number(values.get('RPD'))} | {_number(values.get('bias'))} |")
    thresholds = quality.get("thresholds_used") or {}
    lines += [
        "",
        "- R²描述拟合解释能力；RMSE描述误差大小；RPD受参考值分布影响。三者应结合样本覆盖与偏差共同判断。",
        "- 表中指标读取实际计算结果；未记录字段不补算。RMSEP表示最终评估分区的RMSE，不能与调参误差混称。",
        f"- 领域：{_text(thresholds.get('domain', metrics.get('domain')))}；质量等级：{_text(quality.get('grade'))}；门禁通过：{'是' if quality.get('passed') else '否'}。",
        f"- 使用阈值：R²≥{_text(thresholds.get('min_r2'))}，RPD≥{_text(thresholds.get('min_rpd'))}；其他门禁约束见 metrics.json 的 quality 字段。",
        "",
        "## 5. 可视化与诊断解读",
    ]
    plots = [
        ("原始光谱", raw_spectra_b64, "raw_spectra.png", "查看基线、散射、噪声与样本覆盖；曲线差异本身不能证明存在异常样本。"),
        (f"预测 vs 参考值（{holdout}）", predicted_vs_reference_b64, "predicted_vs_reference.png", "点越接近1:1线，预测越接近参考值；检查低值/高值区是否偏离及误差覆盖。"),
        (f"残差诊断（{holdout}）", residuals_b64, "residuals.png", "残差定义为预测−参考。检查是否围绕零分布，以及趋势、异方差和局部大误差；不自动排除离群点。"),
        ("交叉验证选成分", cv_curve_b64, "cv_curve.png", "曲线来自实际交叉验证计算；成分增加未必改善泛化。CV对应的拟合范围以协议原始记录为准。"),
    ]
    for title, present, filename, explanation in plots:
        if present:
            lines += ["", f"### {title}", "", f"![{title}]({filename})", "", explanation]
    for filename, title in (extra_plots or {}).items():
        lines += ["", f"### {_text(title)}", "", f"![{_text(title)}]({filename})", "", "变量贡献是模型解释证据，不能单独证明化学归属或因果关系。"]
    diagnostics = metrics.get("diagnostics") or {}
    if diagnostics:
        lines += ["", "### 调参集残差诊断（与上方最终评估图分区不同）", "", "| 诊断字段 | 实际结果 |", "|---|---|"]
        for key in ["residual_trend", "residual_variance", "outlier_count", "bias"]:
            if key in diagnostics:
                lines.append(f"| {_text(key)} | {_text(diagnostics[key])} |")
    lines += [
        "",
        "## 6. 适用范围与后续建议",
        "- 预测限于与训练数据一致的仪器、光谱轴、预处理和目标单位；超出参考值覆盖范围时应单独验证。",
        "- 使用前核对样本代表性、批次/仪器差异、偏差和误差是否满足业务要求；内部留出达标仍需针对实际使用场景复核。",
        "- 上线需独立完成用户批准与模型注册；使用时保留漂移监测和定期参考样本复测。",
    ]
    if not quality.get("passed"):
        lines.append(f"- 当前未通过门禁；记录的行动建议：{_text(quality.get('action'))}。先复核数据覆盖和诊断，再决定是否调整方案。")
    repro = metrics.get("reproducibility") or {}
    lines += [
        "",
        "## 7. 复现与交付说明",
        f"- 数据SHA-256：{_text(metrics.get('training_data_hash'))}。",
        f"- 随机种子：{_text(repro.get('random_state'))}；协议：{_text(metrics.get('protocol'))}。",
        "- report.html：可离线打开的完整图文报告；delivery.zip：报告、模型、指标、诊断图和文件校验清单。",
        "- 模型及原有安全清单保持原样；metrics.json保留完整结果与复现参数。下载包不包含原始上传数据。",
        "- 未记录的单位、样本数或诊断保持明确缺失；请勿将缺失字段解释为零或已通过。",
        "",
    ]
    return "\n".join(lines)
