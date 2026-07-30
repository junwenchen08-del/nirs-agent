"use client";

import {
  ArrowRightIcon,
  BrainCircuitIcon,
  CheckCircle2Icon,
  FileSearchIcon,
  LineChartIcon,
  MicroscopeIcon,
} from "lucide-react";

import { cn } from "@/lib/utils";

import { Section } from "../section";

const WORKFLOW = [
  {
    icon: FileSearchIcon,
    step: "01",
    title: "数据审计",
    text: "识别字段、分区与异常，必要时请求用户确认映射。",
  },
  {
    icon: MicroscopeIcon,
    step: "02",
    title: "分析建模",
    text: "在无泄漏边界内选择预处理、波长与模型家族。",
  },
  {
    icon: BrainCircuitIcon,
    step: "03",
    title: "证据反思",
    text: "根据质量门禁与专业知识判断重试、停止或补充证据。",
  },
  {
    icon: CheckCircle2Icon,
    step: "04",
    title: "审批注册",
    text: "绑定指标与复现清单，经用户批准后注册版本化模型。",
  },
  {
    icon: LineChartIcon,
    step: "05",
    title: "预测监控",
    text: "批量预测未知样品，记录审计链并监测适用域漂移。",
  },
];

export function SkillsSection({ className }: { className?: string }) {
  return (
    <Section
      className={cn("w-full bg-white/2 px-4 py-24", className)}
      title="自主化学计量学工作流"
      subtitle="智能体负责规划与协同，nir_core 负责可复现的确定性计算"
    >
      <div className="container-md mx-auto mt-12 grid grid-cols-1 gap-4 md:grid-cols-5">
        {WORKFLOW.map(({ icon: Icon, step, title, text }, index) => (
          <div key={step} className="relative">
            <div className="h-full rounded-2xl border border-white/10 bg-black/30 p-5">
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm text-amber-300">{step}</span>
                <Icon className="size-5 text-zinc-400" />
              </div>
              <h3 className="mt-8 text-lg font-semibold text-white">{title}</h3>
              <p className="mt-3 text-sm leading-6 text-zinc-400">{text}</p>
            </div>
            {index < WORKFLOW.length - 1 && (
              <ArrowRightIcon className="absolute top-1/2 -right-3 z-10 hidden size-5 -translate-y-1/2 text-amber-300/50 md:block" />
            )}
          </div>
        ))}
      </div>
    </Section>
  );
}
