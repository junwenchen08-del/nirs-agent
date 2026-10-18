"use client";

import { ArrowRightIcon } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

export function Hero({ className }: { className?: string }) {
  const { locale } = useI18n();
  const zh = locale === "zh-CN";
  return (
    <section
      className={cn(
        "relative mx-auto grid w-full max-w-6xl items-center gap-12 px-6 pt-36 pb-20 lg:grid-cols-[1.1fr_1fr] lg:pt-44 lg:pb-28",
        className,
      )}
    >
      <div>
        <p className="text-primary mb-5 text-sm font-medium">
          ChemAgent ·{" "}
          {zh ? "近红外光谱分析工作台" : "Near-infrared analysis workspace"}
        </p>
        <h1 className="text-4xl leading-tight font-semibold tracking-tight sm:text-5xl">
          {zh ? (
            <>
              近红外分析，
              <br />
              从数据到模型
            </>
          ) : (
            <>
              From spectral data
              <br />
              to a reusable model
            </>
          )}
        </h1>
        <p className="text-muted-foreground mt-6 max-w-lg text-base leading-relaxed">
          {zh
            ? "检查光谱数据，探索处理与建模方案。每一步的方法、参数和结果都有记录，模型保存后可在新的分析中复用。"
            : "Inspect spectra, explore processing and modeling, and review recorded methods, parameters and results. Reuse saved models in a new analysis."}
        </p>
        <div className="mt-8 flex flex-wrap items-center gap-5">
          <Button asChild size="lg">
            <Link href="/workspace">
              {zh ? "进入工作台" : "Open workspace"}
              <ArrowRightIcon className="size-4" />
            </Link>
          </Button>
          <Link
            href={`/${zh ? "zh" : "en"}/docs`}
            className="text-primary text-sm underline underline-offset-4"
          >
            {zh ? "查看使用指南" : "Read the guide"}
          </Link>
        </div>
      </div>
      <div className="bg-card rounded-2xl border p-6 sm:p-8">
        <p className="text-muted-foreground mb-6 text-xs">
          {zh ? "分析流程 · 界面示意" : "Analysis flow · illustration"}
        </p>
        <div className="space-y-5">
          {(zh
            ? [
                ["01", "准备光谱数据", "检查结构、样本与分析目标"],
                ["02", "建模与诊断", "查看方案、参数、预测与残差"],
                ["03", "保存并复用模型", "确认结果后授权保存，用于新数据预测"],
              ]
            : [
                [
                  "01",
                  "Prepare spectra",
                  "Inspect structure, samples and target",
                ],
                [
                  "02",
                  "Model and diagnose",
                  "Review methods, parameters and diagnostics",
                ],
                [
                  "03",
                  "Save and reuse",
                  "Approve saving, then predict new data",
                ],
              ]
          ).map(([n, title, description]) => (
            <div key={n} className="flex items-start gap-4">
              <span className="bg-accent text-primary flex size-9 shrink-0 items-center justify-center rounded-lg text-sm font-medium">
                {n}
              </span>
              <div>
                <h2 className="font-medium">{title}</h2>
                <p className="text-muted-foreground mt-1 text-sm">
                  {description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
