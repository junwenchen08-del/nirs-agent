"use client";

import MagicBento, { type BentoCardProps } from "@/components/ui/magic-bento";

import { Section } from "../section";

const COLOR = "#0a0a0a";
const FEATURES: BentoCardProps[] = [
  {
    color: COLOR,
    label: "Data Intelligence",
    title: "多格式智能识别",
    description: "CSV、TXT、MATLAB 数据结构与字段角色自动审计",
  },
  {
    color: COLOR,
    label: "Leakage Safe",
    title: "科学划分与评估",
    description: "官方分区优先，自动划分与调优严格隔离最终测试集",
  },
  {
    color: COLOR,
    label: "Auto Modeling",
    title: "模型与波长自主选择",
    description: "依据调优证据比较模型家族和关键波长方案",
  },
  {
    color: COLOR,
    label: "Knowledge Grounded",
    title: "近红外专业知识库",
    description: "BGE-M3 检索与重排，为复杂决策提供文献证据",
  },
  {
    color: COLOR,
    label: "Governed",
    title: "模型注册与质量门禁",
    description: "审批、指标绑定、复现清单和完整性验证缺一不可",
  },
  {
    color: COLOR,
    label: "Production Aware",
    title: "预测审计与漂移告警",
    description: "持续记录推理行为，识别超出训练适用域的新批次",
  },
];

export function WhatsNewSection({ className }: { className?: string }) {
  return (
    <Section
      className={className}
      title="面向真实业务的核心能力"
      subtitle="把近红外专家经验、确定性算法与智能体协作整合为一个可部署平台"
    >
      <div className="flex w-full items-center justify-center">
        <MagicBento data={FEATURES} />
      </div>
    </Section>
  );
}
