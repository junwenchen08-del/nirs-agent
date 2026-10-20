"use client";

import Link from "next/link";

import { urlOfArtifact } from "@/core/artifacts/utils";
import { record, type AutonomousSummary } from "@/core/nir/autonomous";
import { chartColors } from "@/core/nir/chart-theme";
import {
  lifecycleStatus,
  type LifecycleNode,
} from "@/core/nir/process-lifecycle";

import { Metrics } from "./autonomous-evidence";
import { FinalCharts } from "./autonomous-final-views";
import { CalSpectraComparison } from "./cal-spectra";
import {
  NodeDetailTabs,
  type DetailTabOption,
  type NodeDetailTab,
} from "./node-detail-tabs";
import { ProcessDataOverview } from "./process-data-overview";
import { ProcessFacts } from "./process-facts";

export function LifecycleNodeDetail({
  node,
  threadId,
  summary,
  zh,
  tab: requestedTab,
  onTabChange,
}: {
  node: LifecycleNode;
  threadId: string;
  summary: AutonomousSummary;
  zh: boolean;
  tab: NodeDetailTab;
  onTabChange: (tab: NodeDetailTab) => void;
}) {
  const id = `lifecycle-${node.id}`;
  const isData = node.id === "input" || node.id === "inspect";
  const options: DetailTabOption[] = [
    {
      key: "overview",
      label: isData
        ? zh
          ? "数据概览"
          : "Data overview"
        : node.id === "prepare"
          ? zh
            ? "分区概览"
            : "Partition overview"
          : zh
            ? "阶段概览"
            : "Stage overview",
    },
    {
      key: "config",
      label: isData
        ? zh
          ? "数据结构"
          : "Data structure"
        : node.id === "prepare"
          ? zh
            ? "验证协议"
            : "Validation protocol"
          : zh
            ? "记录详情"
            : "Recorded details",
    },
    ...(isData || node.id === "prepare" || node.id === "test"
      ? [
          {
            key: "evaluation" as const,
            label: isData
              ? zh
                ? "原始光谱"
                : "Raw spectra"
              : node.id === "prepare"
                ? zh
                  ? "分区图"
                  : "Partition chart"
                : zh
                  ? "最终诊断"
                  : "Final diagnostics",
          },
        ]
      : []),
  ];
  const tab = options.some((item) => item.key === requestedTab)
    ? requestedTab
    : "overview";
  const counts = Object.entries(
    record(summary.data_context.partition_counts),
  ).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" &&
      Number.isFinite(entry[1]) &&
      entry[1] >= 0,
  );
  return (
    <section
      className="process-detail-pane bg-card rounded-xl border"
      data-testid="lifecycle-detail"
      data-node-id={node.id}
    >
      <header className="shrink-0 space-y-2 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-lg font-semibold">{node.title[zh ? 0 : 1]}</h3>
          <span className="bg-muted text-muted-foreground rounded px-2 py-1 text-xs">
            {lifecycleStatus(node.status, zh)}
          </span>
        </div>
        <p className="text-sm">{node.method[zh ? 0 : 1]}</p>
      </header>
      <NodeDetailTabs
        value={tab}
        onChange={onTabChange}
        id={id}
        zh={zh}
        options={options}
      />
      <div
        className="process-detail-body space-y-4 p-4"
        id={`${id}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-${tab}`}
        tabIndex={0}
      >
        {tab === "overview" && (
          <>
            {(isData || node.id === "prepare") && (
              <ProcessDataOverview summary={summary} node={node} zh={zh} />
            )}
            <details
              open={!isData && node.id !== "prepare"}
              className="rounded-lg border p-3"
            >
              <summary className="cursor-pointer text-sm font-semibold">
                {zh ? "这一步发生的信息" : "Recorded information"}
              </summary>
              {Object.keys(node.facts).length ? (
                <ProcessFacts value={node.facts} zh={zh} />
              ) : (
                <p className="text-muted-foreground text-sm">
                  {zh
                    ? "本次运行未单独保存此步骤的详细记录。后续完成状态不能证明此步骤的具体操作。"
                    : "No separate detail receipt was recorded. A later completed stage does not prove this operation."}
                </p>
              )}
              {node.id === "inspect" && (
                <p className="text-muted-foreground text-xs">
                  {zh
                    ? "以上为搜索准备时固定的数据摘要；未单独记录的解析操作与时间保持缺失。"
                    : "This is the fixed preparation context; separate parsing operations and timestamps remain unrecorded."}
                </p>
              )}
            </details>
            {isData && (
              <CalSpectraComparison
                threadId={threadId}
                summary={summary}
                zh={zh}
                rawOnly
              />
            )}
            {node.id === "delivery" && (
              <div className="flex flex-wrap gap-2">
                {summary.downloads.map((file) => (
                  <a
                    key={file.path}
                    className="hover:bg-accent rounded-md border px-3 py-2 text-sm"
                    href={urlOfArtifact({
                      threadId,
                      filepath: `/${file.path}`,
                      download: true,
                    })}
                    download
                  >
                    {{
                      "report.html": "图文报告（HTML）",
                      "delivery.zip": "完整下载包（ZIP）",
                      "model.joblib": "预测模型",
                      "report.md": "报告（Markdown）",
                      "metrics.json": "完整指标",
                      "decisions.json": "决策记录",
                    }[file.name] && zh
                      ? {
                          "report.html": "图文报告（HTML）",
                          "delivery.zip": "完整下载包（ZIP）",
                          "model.joblib": "预测模型",
                          "report.md": "报告（Markdown）",
                          "metrics.json": "完整指标",
                          "decisions.json": "决策记录",
                        }[file.name]
                      : file.name}
                  </a>
                ))}
              </div>
            )}
          </>
        )}
        {tab === "config" && (
          <>
            <h4 className="text-sm font-semibold">
              {isData
                ? zh
                  ? "已记录的数据结构与解析信息"
                  : "Recorded structure and parsing"
                : zh
                  ? "参数与输入"
                  : "Parameters and inputs"}
            </h4>
            <ProcessFacts
              value={
                node.id === "prepare" || node.id === "inspect"
                  ? { ...node.facts, task: summary.task }
                  : node.facts
              }
              zh={zh}
            />
            {isData && (
              <p className="rounded-lg border border-dashed p-3 text-sm">
                {zh
                  ? "矩阵映射、样本方向、质量检查和采集条件仅在原记录提供时展示；未记录不代表已检查或没有问题。需要补充信息时，请返回原对话。"
                  : "Mapping, orientation, quality checks and acquisition conditions require original records. Missing checks do not imply clean data. Return to the chat to provide corrections."}
              </p>
            )}
            <p className="text-muted-foreground text-xs">
              {zh
                ? "未记录的固定、搜索或最终参数不作补写；实验参数在对应轮次中查看。"
                : "Missing parameters are not inferred. Experiment parameters are available in their own rounds."}
            </p>
          </>
        )}
        {tab === "evaluation" && (
          <>
            {isData ? (
              <CalSpectraComparison
                threadId={threadId}
                summary={summary}
                zh={zh}
                rawOnly
              />
            ) : node.id === "prepare" && counts.length > 0 ? (
              <figure>
                <figcaption className="mb-3 text-sm font-medium">
                  {zh ? "样本用途分布" : "Sample partitions"}
                </figcaption>
                <svg
                  viewBox={`0 0 560 ${counts.length * 52 + 36}`}
                  role="img"
                  aria-label={zh ? "样本用途数量" : "Partition sample counts"}
                  className="w-full"
                >
                  {counts.map(([key, count], index) => (
                    <g key={key}>
                      <text
                        x="8"
                        y={index * 52 + 30}
                        fontSize="14"
                        fill="currentColor"
                      >
                        {{
                          cal: ["训练集 Cal", "Cal"],
                          calibration: ["训练集 Cal", "Cal"],
                          val: ["验证集 Val", "Val"],
                          validation: ["验证集 Val", "Val"],
                          test: ["测试集 Test", "Test"],
                        }[key]?.[zh ? 0 : 1] ?? key}
                      </text>
                      <rect
                        x="140"
                        y={index * 52 + 12}
                        width={Math.max(
                          0,
                          (count /
                            (Math.max(...counts.map(([, n]) => n)) || 1)) *
                            330,
                        )}
                        height="27"
                        rx="4"
                        fill={chartColors.processed}
                      />
                      <text
                        x="485"
                        y={index * 52 + 31}
                        fontSize="14"
                        fill="currentColor"
                      >
                        {count}
                      </text>
                    </g>
                  ))}
                </svg>
                <p className="text-muted-foreground text-xs">
                  {zh
                    ? "仅绘制已记录的分区数量，不读取测试集原值。"
                    : "Recorded partition counts only; no Test values are read."}
                </p>
              </figure>
            ) : node.id === "test" ? (
              <>
                <Metrics
                  value={{
                    ...summary,
                    metrics: { test: summary.metrics.test },
                  }}
                />
                <details className="text-sm">
                  <summary className="cursor-pointer">
                    {zh ? "完整最终拟合指标" : "All final-fit metrics"}
                  </summary>
                  <div className="mt-3">
                    <Metrics
                      value={{
                        ...summary,
                        metrics: {
                          cal: summary.metrics.cal,
                          cal_cv: summary.metrics.cal_cv,
                          val: summary.metrics.val,
                        },
                      }}
                    />
                  </div>
                </details>
                <FinalCharts threadId={threadId} summary={summary} zh={zh} />
              </>
            ) : (
              <p className="text-muted-foreground text-sm">
                {zh
                  ? "此步骤没有单独记录可绘制的诊断数据。实验图表请选择对应轮次；审批与入库使用状态回执展示。"
                  : "No separate diagnostic data recorded for this step. Select an experiment for its plots; approval and registration use receipts."}
              </p>
            )}
          </>
        )}
      </div>
      <footer className="text-muted-foreground flex shrink-0 flex-wrap justify-between gap-2 border-t px-4 py-2 text-xs">
        {zh
          ? "按已记录事实展示；查看不会启动执行"
          : "Recorded facts only; viewing never starts execution"}
        <Link
          className="text-primary underline"
          href={`/workspace/chats/${encodeURIComponent(threadId)}`}
        >
          {zh ? "返回原对话" : "Return to chat"}
        </Link>
      </footer>
    </section>
  );
}
