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
      title="确定性 NIR 分析引擎"
      subtitle="大模型不直接计算光谱；所有关键步骤均由受控工具执行并留下可复核证据"
    >
      <div className="mx-auto mt-10 flex w-full max-w-6xl flex-col items-center gap-12 px-4 lg:flex-row lg:gap-16">
        <div className="w-full flex-1">
          <Terminal className="h-[380px] w-full">
            <TypingAnimation>$ nir_load_data /uploads/corn.mat</TypingAnimation>
            <AnimatedSpan delay={900} className="text-green-500">
              ✓ 80 samples · 700 wavelengths · schema verified
            </AnimatedSpan>

            <TypingAnimation delay={1500}>
              $ nir_train_auto_split_model --method auto
            </TypingAnimation>
            <AnimatedSpan delay={2400} className="text-blue-400">
              → split: SPXY · preprocessing: SNV + SG
            </AnimatedSpan>
            <AnimatedSpan delay={3000} className="text-blue-400">
              → comparing PLS, Ridge, SVR and Extra Trees
            </AnimatedSpan>
            <AnimatedSpan delay={3700} className="text-green-500">
              ✓ best model: PLS · external test passed
            </AnimatedSpan>

            <TypingAnimation delay={4400}>
              $ nir_register_model --require-approval
            </TypingAnimation>
            <AnimatedSpan delay={5200} className="text-amber-300">
              ✓ scientific gate · quality gate · integrity manifest
            </AnimatedSpan>

            <TypingAnimation delay={5900}>
              $ nir_predict /uploads/unknown.csv
            </TypingAnimation>
            <AnimatedSpan delay={6700} className="text-green-500">
              ✓ predictions.csv · applicability domain monitored
            </AnimatedSpan>
          </Terminal>
        </div>

        <div className="w-full flex-1 space-y-6">
          <div className="space-y-4">
            <p className="text-sm font-medium tracking-wider text-amber-300 uppercase">
              Reproducible by design
            </p>
            <h2 className="text-4xl font-bold tracking-tight lg:text-5xl">
              nir_core
            </h2>
          </div>
          <div className="space-y-4 text-lg leading-8 text-zinc-400">
            <p>
              独立的 Python
              化学计量学算法包负责数据加载、预处理、模型训练、指标计算和质量判断。
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
