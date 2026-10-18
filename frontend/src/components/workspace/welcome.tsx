"use client";

import { useSearchParams } from "next/navigation";

import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

export function Welcome({
  className,
}: {
  className?: string;
  mode?: "ultra" | "pro" | "thinking" | "flash";
}) {
  const { t, locale } = useI18n();
  const skill = useSearchParams().get("mode") === "skill";
  const zh = locale === "zh-CN";
  return (
    <div
      className={cn(
        "mx-auto flex w-full flex-col items-center gap-3 px-4 py-6 text-center",
        className,
      )}
    >
      <h1 className="text-[28px] leading-snug font-semibold">
        {skill
          ? t.welcome.createYourOwnSkill
          : zh
            ? "开始一次光谱分析"
            : "Start a spectral analysis"}
      </h1>
      <p className="text-muted-foreground max-w-xl text-sm leading-relaxed whitespace-pre-line">
        {skill
          ? t.welcome.createYourOwnSkillDescription
          : zh
            ? "上传光谱数据，或选择已有数据集与模型，告诉我你想分析什么。"
            : "Upload spectra or choose a saved dataset or model, then describe your task."}
      </p>
    </div>
  );
}
