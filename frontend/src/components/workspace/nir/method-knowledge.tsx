"use client";

import { selectMethodKnowledge } from "@/core/nir/method-knowledge";

export function MethodKnowledge({
  evidence,
  zh,
}: {
  evidence: unknown;
  zh: boolean;
}) {
  const view = selectMethodKnowledge(evidence);
  if (!view) return null;
  const fallbackReasons: Record<string, [string, string]> = {
    method_retrieval_unavailable: [
      "方法知识检索暂不可用，采用现有规则候选。",
      "Method retrieval was unavailable; rule candidates were used.",
    ],
    no_actionable_diagnostic_tags: [
      "未发现可检索的问题标签，采用现有规则候选。",
      "No actionable diagnostic tags were recorded; rule candidates were used.",
    ],
    no_relevant_method_reference: [
      "未检索到相关方法依据，采用现有规则候选。",
      "No relevant method references were retrieved; rule candidates were used.",
    ],
    no_executable_reference_candidates: [
      "检索结果没有通过执行条件校验，采用现有规则候选。",
      "Retrieved proposals did not pass execution checks; rule candidates were used.",
    ],
  };
  return (
    <section
      className="space-y-3 rounded-xl border p-5"
      data-testid="method-knowledge"
    >
      <h3 className="font-medium">
        {zh ? "数据诊断与算法知识依据" : "Diagnostics and method references"}
      </h3>
      <p className="text-sm">
        {view.mode === "retrieval_guided"
          ? zh
            ? "依据方法知识优先安排首轮候选，原始光谱保留为同批对照。"
            : "Method references prioritized the first candidates, with raw spectra retained as a same-batch control."
          : (fallbackReasons[view.reason ?? ""]?.[zh ? 0 : 1] ??
            (zh
              ? "未记录知识引导的候选排序。"
              : "No retrieval-guided candidate order was recorded."))}
      </p>
      {view.findings.length > 0 && (
        <p className="text-sm">
          {zh ? "观察到的问题：" : "Observed findings: "}
          {view.findings.join("、")}
        </p>
      )}
      {view.sources.length > 0 && (
        <ul className="space-y-2">
          {view.sources.map((source, index) => (
            <li
              key={`${source.evidenceId ?? source.method}-${index}`}
              className="bg-muted/40 flex items-center justify-between gap-4 rounded-md px-3 py-2 text-sm"
            >
              <span>
                {source.title}
                {source.eligible === false && (
                  <span className="text-muted-foreground ml-2">
                    {zh ? "仅供查阅" : "Reference only"}
                  </span>
                )}
              </span>
              {source.url && (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary shrink-0 underline underline-offset-4"
                >
                  {zh ? "官方说明" : "Official documentation"}
                </a>
              )}
            </li>
          ))}
        </ul>
      )}
      <p className="text-muted-foreground text-xs leading-5">
        {view.scope === "calibration_only" &&
          (zh
            ? "诊断与检索只使用校准数据。"
            : "Diagnostics and retrieval used calibration data only. ")}
        {zh
          ? "方法参考用于提出候选；最终采用由训练内部验证决定。官网默认值、参数约束与项目搜索值分别记录。"
          : "References propose candidates; training-only validation determines adoption. Official defaults, constraints and project search values are recorded separately."}
      </p>
    </section>
  );
}
