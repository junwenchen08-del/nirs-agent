import { BRAND } from "@/core/branding";
import type { Locale } from "@/core/i18n/locale";

const zh = `# 关于 ChemAgent

ChemAgent 是面向近红外光谱分析的智能工作台，将数据集、专业方法、自主建模、结果诊断和模型管理连接在同一工作流程中。

## 分析工作

- **数据集工作台**：上传和解析数据，查看样本、光谱与字段映射，保存样本角色和分析区间。
- **自主建模**：围绕分析目标提出方案，执行预处理与模型参数搜索，并保留每轮决策和执行证据。
- **诊断与报告**：查看范围明确的指标、预测与残差图，追踪最终保留模型和报告文件。
- **模型库**：查看模型关联的数据集和训练信息，挂载已保存模型开展后续预测。
- **方法与知识**：管理专业方法参考和知识资料，为分析决策提供依据。

## 使用帮助

[ChemAgent 使用文档](/zh/docs) · [ChemAgent 项目](${BRAND.repositoryUrl}) · [反馈问题](${BRAND.issuesUrl})

分析结论以实际记录的数据、指标和报告为准。模型注册遵循现有人工审批流程。
`;

const en = `# About ChemAgent

ChemAgent is an intelligent workspace for near-infrared spectroscopy, connecting datasets, analytical methods, autonomous modeling, diagnostics and model management.

## Your analysis workflow

- **Dataset workspace**: Upload and parse data, inspect samples and spectra, and save sample roles and analysis intervals.
- **Autonomous modeling**: Propose and execute preprocessing and parameter searches while retaining each round's decisions and evidence.
- **Diagnostics and reports**: Inspect scoped metrics, prediction and residual charts, retained models and report files.
- **Model library**: Inspect linked training data and mount saved models for subsequent predictions.
- **Methods and knowledge**: Manage analytical references and knowledge sources that support decisions.

## Help

[ChemAgent documentation](/en/docs) · [ChemAgent project](${BRAND.repositoryUrl}) · [Report an issue](${BRAND.issuesUrl})

Interpret conclusions using the recorded data, metrics and reports. Model registration follows the existing human approval workflow.
`;

export function getAboutMarkdown(locale: Locale): string {
  return locale === "zh-CN" ? zh : en;
}
