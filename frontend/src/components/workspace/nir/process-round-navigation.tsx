"use client";

import { useEffect, useId, useState } from "react";

import {
  actionLabel,
  executionLabel,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import {
  lifecycleStatus,
  type LifecycleKey,
  type LifecycleNode,
} from "@/core/nir/process-lifecycle";

type Stage = "data" | "search" | "final" | "delivery";
const stages: { key: Stage; label: [string, string] }[] = [
  { key: "data", label: ["数据准备", "Data preparation"] },
  { key: "search", label: ["逐轮建模", "Round pipelines"] },
  { key: "final", label: ["最终评价", "Final evaluation"] },
  { key: "delivery", label: ["交付审批", "Delivery and approval"] },
];

export function ProcessRoundNavigation({
  stage,
  selectedId,
  lifecycle,
  lifecycleId,
  onLifecycle,
  rows,
  summary,
  zh,
  total,
  loading,
  failed,
  hasMore,
  loadingMore,
  onStage,
  onRound,
  onMore,
  onRefresh,
}: {
  stage: Stage;
  selectedId: string;
  lifecycle: LifecycleNode[];
  lifecycleId: LifecycleKey | null;
  onLifecycle: (id: LifecycleKey) => void;
  rows: DecisionView[];
  summary: AutonomousSummary;
  zh: boolean;
  total: number | null | undefined;
  loading: boolean;
  failed: boolean;
  hasMore: boolean;
  loadingMore: boolean;
  onStage: (stage: Stage) => void;
  onRound: (row: DecisionView) => void;
  onMore: () => void;
  onRefresh: () => void;
}) {
  const id = useId();
  const [open, setOpen] = useState({
    data: stage === "data",
    search: stage === "search",
  });
  useEffect(() => {
    if (stage === "data" || stage === "search")
      setOpen((current) =>
        current[stage] ? current : { ...current, [stage]: true },
      );
    else
      setOpen((current) =>
        current.data || current.search
          ? { data: false, search: false }
          : current,
      );
  }, [stage, selectedId, lifecycleId]);
  const toggleStage = (key: Stage) => {
    if (key === "data" || key === "search") {
      setOpen((current) => ({
        ...current,
        [key]: stage === key ? !current[key] : true,
      }));
      if (stage === key) return;
    }
    onStage(key);
  };
  return (
    <aside
      className="process-round-sidebar"
      data-testid="process-round-sidebar"
    >
      <nav
        aria-label={zh ? "建模阶段" : "Modeling stages"}
        className="process-stage-pipeline"
      >
        {stages.map(({ key, label }, index) => (
          <section key={key} className="process-stage" data-stage={key}>
            <button
              type="button"
              aria-pressed={stage === key}
              aria-expanded={
                key === "data" || key === "search" ? open[key] : undefined
              }
              aria-controls={
                key === "data" || key === "search" ? `${id}-${key}` : undefined
              }
              onClick={() => toggleStage(key)}
              className="process-stage-button"
            >
              <span className="process-stage-number">{index + 1}</span>
              <span className="min-w-0 flex-1">{label[zh ? 0 : 1]}</span>
              {(key === "data" || key === "search") && (
                <span aria-hidden="true" className="process-stage-chevron">
                  {open[key] ? "▾" : "▸"}
                </span>
              )}
            </button>
            {key === "data" && open.data && (
              <nav
                id={`${id}-data`}
                className="process-stage-children"
                aria-label={zh ? "数据准备步骤" : "Data preparation steps"}
              >
                {lifecycle
                  .filter((node) => node.stage === "data")
                  .map((node) => (
                    <button
                      type="button"
                      key={node.id}
                      data-testid="data-nav-node"
                      data-node-id={node.id}
                      aria-pressed={stage === "data" && lifecycleId === node.id}
                      className="process-stage-child"
                      onClick={() => onLifecycle(node.id)}
                    >
                      <span className="block text-sm font-medium">
                        {node.title[zh ? 0 : 1]}
                      </span>
                      <span className="text-muted-foreground mt-1 block text-xs">
                        {lifecycleStatus(node.status, zh)}
                      </span>
                    </button>
                  ))}
              </nav>
            )}
            {key === "search" && open.search && (
              <div
                id={`${id}-search`}
                className="process-stage-children"
                role="navigation"
                aria-label={zh ? "建模轮次节点" : "Modeling round nodes"}
              >
                {rows.map((row) => {
                  const stopping = ["STOP", "RETAIN"].includes(row.action);
                  const retained = Boolean(
                    row.experiment_id &&
                    row.experiment_id === summary.retained_experiment_id,
                  );
                  const active =
                    stage === "search" && selectedId === row.decision_id;
                  return (
                    <button
                      type="button"
                      key={row.decision_id}
                      data-testid="round-nav-node"
                      data-decision-id={row.decision_id}
                      aria-pressed={active}
                      onClick={() => onRound(row)}
                      className="process-stage-child"
                    >
                      <span className="flex flex-wrap items-center justify-between gap-1 text-sm">
                        <strong className="font-medium">
                          {stopping
                            ? actionLabel(row.action, zh)
                            : zh
                              ? `第 ${fmt(row.round_index)} 轮`
                              : `Round ${fmt(row.round_index)}`}
                        </strong>
                        {retained && (
                          <span className="text-primary text-xs">
                            {zh ? "保留" : "Retained"}
                          </span>
                        )}
                      </span>
                      <span className="text-muted-foreground mt-1 block text-xs">
                        {executionLabel(row.status, zh)}
                      </span>
                    </button>
                  );
                })}
                {!rows.length && (
                  <p className="text-muted-foreground py-2 text-xs">
                    {loading
                      ? zh
                        ? "正在读取轮次…"
                        : "Reading rounds…"
                      : zh
                        ? "尚无已接受决策"
                        : "No accepted decisions"}
                  </p>
                )}
                <p className="text-muted-foreground pt-2 text-xs">
                  {zh
                    ? `已加载 ${rows.length} / ${total ?? "—"} 次决策`
                    : `${rows.length} / ${total ?? "—"} decisions loaded`}
                </p>
                {failed && (
                  <p role="alert" className="text-xs">
                    {zh
                      ? "轮次读取失败，已加载记录仍可查看。"
                      : "Read failed; loaded records remain available."}
                  </p>
                )}
                {hasMore && (
                  <button
                    type="button"
                    disabled={loadingMore}
                    onClick={onMore}
                    className="text-primary py-2 text-xs underline"
                  >
                    {zh ? "加载更多轮次" : "Load more rounds"}
                  </button>
                )}
                <button
                  type="button"
                  onClick={onRefresh}
                  className="text-muted-foreground py-2 text-xs underline"
                >
                  {zh ? "刷新轮次" : "Refresh rounds"}
                </button>
              </div>
            )}
          </section>
        ))}
      </nav>
    </aside>
  );
}
