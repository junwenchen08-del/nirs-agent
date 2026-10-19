"use client";

import {
  AnimatedSpan,
  Terminal,
  TypingAnimation,
} from "@/components/ui/terminal";

import { Section } from "../section";

export function SandboxSection({ className }: { className?: string }) {
  return (
    <Section
      className={className}
      title="可复现的光谱分析"
      subtitle="大模型不直接计算光谱；所有关键步骤均由受控工具执行并留下可复核证据"
    >
      <div className="mx-auto mt-10 flex w-full max-w-6xl flex-col items-center gap-12 px-4 lg:flex-row lg:gap-16">
        <div className="w-full flex-1">
          <Terminal className="h-[380px] w-full">
            <TypingAnimation>光谱分析流程示意</TypingAnimation>
            <AnimatedSpan delay={700} className="text-zinc-400">
              实际方案、指标和报告以当前任务的执行记录为准
            </AnimatedSpan>
            <TypingAnimation delay={1400}>
              01 导入并核对光谱数据
            </TypingAnimation>
            <AnimatedSpan delay={2100} className="text-blue-400">
              确认目标变量、波长轴与样本角色
            </AnimatedSpan>
            <TypingAnimation delay={2800}>02 执行预处理与建模</TypingAnimation>
            <AnimatedSpan delay={3500} className="text-blue-400">
              记录方案、参数及训练和验证指标
            </AnimatedSpan>
            <TypingAnimation delay={4200}>
              03 查看诊断与分析报告
            </TypingAnimation>
            <AnimatedSpan delay={4900} className="text-amber-300">
              按评价范围解读结果，确认后再将模型入库
            </AnimatedSpan>
            <TypingAnimation delay={5600}>
              04 使用已保存模型预测新样品
            </TypingAnimation>
            <AnimatedSpan delay={6300} className="text-green-500">
              检查输入兼容性，保留预测记录和分析依据
            </AnimatedSpan>
          </Terminal>
        </div>

        <div className="w-full flex-1 space-y-6">
          <div className="space-y-4">
            <p className="text-sm font-medium tracking-wider text-amber-300 uppercase">
              分析有依据，结果可追踪
            </p>
            <h2 className="text-4xl font-bold tracking-tight lg:text-5xl">
              ChemAgent 分析引擎
            </h2>
          </div>
          <div className="space-y-4 text-lg leading-8 text-zinc-400">
            <p>
              专业的化学计量学算法负责数据加载、预处理、模型训练、指标计算和质量判断。
            </p>
            <p>
              固定随机种子、训练集拟合边界、产物哈希与预测审计链，让每次分析都能够解释、复现和追踪。
            </p>
          </div>
          <div className="flex flex-wrap gap-3 pt-4">
            {[
              "无数据泄漏",
              "适用域监测",
              "产物完整性",
              "多目标建模",
              "资源限制",
            ].map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-zinc-800 bg-zinc-900 px-4 py-2 text-sm text-zinc-300"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}
