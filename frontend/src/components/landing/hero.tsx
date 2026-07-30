"use client";

import { ChevronRightIcon } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { FlickeringGrid } from "@/components/ui/flickering-grid";
import Galaxy from "@/components/ui/galaxy";
import { WordRotate } from "@/components/ui/word-rotate";
import { cn } from "@/lib/utils";

const NIR_CAPABILITIES = [
  "解析光谱数据",
  "优化预处理流程",
  "筛选关键波长",
  "建立定量模型",
  "评估模型质量",
  "预测未知样品",
  "监测模型漂移",
];

export function Hero({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex size-full flex-col items-center justify-center",
        className,
      )}
    >
      <div className="absolute inset-0 z-0 bg-black/45">
        <Galaxy
          mouseRepulsion={false}
          starSpeed={0.2}
          density={0.6}
          glowIntensity={0.35}
          twinkleIntensity={0.3}
          speed={0.5}
        />
      </div>
      <FlickeringGrid
        className="absolute inset-0 z-0 translate-y-8 mask-[url(/images/nir-spectrum.svg)] mask-size-[96vw] mask-center mask-no-repeat md:mask-size-[115vh]"
        squareSize={4}
        gridGap={4}
        color="white"
        maxOpacity={0.3}
        flickerChance={0.25}
      />
      <div className="container-md relative z-10 mx-auto flex h-screen flex-col items-center justify-center px-6">
        <div className="mb-6 rounded-full border border-amber-300/20 bg-amber-300/10 px-4 py-1.5 text-sm font-medium tracking-[0.2em] text-amber-300 backdrop-blur-sm">
          NIR INTELLIGENCE PLATFORM
        </div>
        <h1 className="flex flex-col items-center gap-3 text-center text-4xl font-bold md:text-6xl lg:flex-row">
          <WordRotate words={NIR_CAPABILITIES} />
          <span className="text-white">尽在 NIR-Agent</span>
        </h1>
        <p className="text-muted-foreground mt-8 max-w-4xl text-center text-lg leading-relaxed text-shadow-sm md:text-2xl">
          面向近红外光谱分析的专有智能体
          <br />
          从数据审计、预处理、建模评估到模型注册与预测监控，全流程自主协同
        </p>
        <Link href="/workspace">
          <Button className="mt-9 h-12 px-6 text-base" size="lg">
            进入近红外工作台
            <ChevronRightIcon className="size-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
