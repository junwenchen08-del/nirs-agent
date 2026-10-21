"use client";

import { useState } from "react";

import {
  parseDecision,
  record,
  text,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import { roundConfiguration } from "@/core/nir/round-process";

export function RoundComparison({
  threadId,
  summary,
  decision,
  zh,
  onSelect,
}: {
  threadId: string;
  summary: AutonomousSummary;
  decision: DecisionView;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
}) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const options = summary.trends.filter(
    (row) => row.experiment_id !== decision.experiment_id,
  );
  const source = options.find(
    (row) => row.experiment_id === (selectedId ?? decision.base_experiment_id),
  );
  const query = useProcessResource<unknown>(
    threadId,
    source
      ? `autonomous/${summary.session_id}/decisions/${source.decision_id}`
      : null,
  );
  const parsed = parseDecision(query.data);
  const previous =
    parsed?.decision_id === source?.decision_id &&
    parsed?.experiment_id === source?.experiment_id
      ? parsed
      : null;
  const manifest = text(record(decision.feedback?.cv_settings).manifest_id);
  const comparable = Boolean(
    summary.comparison_group_id &&
    manifest &&
    manifest === text(record(previous?.feedback?.cv_settings).manifest_id) &&
    roundConfiguration(decision).actual &&
    previous &&
    roundConfiguration(previous).actual,
  );
  const current = record(record(decision.feedback?.metrics).val);
  const before = record(record(previous?.feedback?.metrics).val);
  const validCurrent =
    decision.status === "succeeded" && current.scope === "validation";
  return (
    <section className="space-y-3" data-testid="round-comparison">
      <div className="grid gap-3 sm:grid-cols-2">
        {(
          [
            ["RMSE", zh ? "预测误差 RMSE ↓" : "Prediction error RMSE ↓"],
            ["R2", zh ? "决定系数 R² ↑" : "R² ↑"],
          ] as const
        ).map(([key, label]) => {
          const a = before[key],
            b = current[key];
          const change =
            comparable &&
            validCurrent &&
            before.scope === "validation" &&
            typeof a === "number" &&
            typeof b === "number"
              ? b - a
              : null;
          const improved =
            change !== null && (key === "RMSE" ? change < 0 : change > 0);
          return (
            <div key={key} className="bg-muted/30 rounded-lg p-3">
              <p className="text-muted-foreground text-xs">{label}</p>
              <p className="my-1 text-2xl font-semibold tabular-nums">
                {validCurrent ? fmt(b) : "—"}
              </p>
              {previous &&
                before.scope === "validation" &&
                typeof a === "number" && (
                  <p className="text-muted-foreground text-xs">
                    {zh
                      ? `第 ${fmt(previous.round_index)} 轮：`
                      : `Round ${fmt(previous.round_index)}: `}
                    {fmt(a)}
                    {change !== null && (
                      <span
                        data-testid="metric-change"
                        className={improved ? "text-primary ml-2" : "ml-2"}
                      >
                        {change === 0
                          ? zh
                            ? "不变"
                            : "Unchanged"
                          : `${zh ? (key === "RMSE" ? (change < 0 ? "降低 " : "升高 ") : change > 0 ? "提高 " : "降低 ") : "Change "}${zh ? "" : change > 0 ? "+" : ""}${fmt(zh ? Math.abs(change) : change)}`}
                      </span>
                    )}
                  </p>
                )}
            </div>
          );
        })}
      </div>
      {source && !previous && (
        <p
          role={query.isError ? "alert" : "status"}
          className="text-muted-foreground text-xs"
        >
          {query.isPending
            ? zh
              ? "正在读取参考轮次…"
              : "Loading reference…"
            : zh
              ? "参考记录不可用或身份不匹配，未计算改善幅度。"
              : "Reference unavailable; no improvement computed."}
          {query.isError && (
            <button
              className="ml-2 underline"
              onClick={() => void query.refetch()}
            >
              {zh ? "重试" : "Retry"}
            </button>
          )}
        </p>
      )}
      {previous && !comparable && (
        <p className="text-muted-foreground text-xs">
          {zh
            ? "缺少一致的划分与交叉验证记录，仅展示数值，不计算改善幅度。"
            : "Matching split/CV evidence is unavailable; values are shown without improvement claims."}
        </p>
      )}
      {options.length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer">
            {zh ? "选择其他对照轮次" : "Choose another comparison"}
          </summary>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <label>
              {zh ? "对照轮次：" : "Compare with: "}
              <select
                className="bg-background rounded border p-2"
                value={source?.experiment_id ?? ""}
                onChange={(event) => setSelectedId(event.target.value || null)}
              >
                <option value="">
                  {zh ? "使用记录中的参考轮次" : "Recorded reference"}
                </option>
                {options.map((row) => (
                  <option key={row.experiment_id} value={row.experiment_id}>
                    {zh
                      ? `第 ${row.round_index ?? "—"} 轮`
                      : `Round ${row.round_index ?? "—"}`}
                    {row.experiment_id === decision.base_experiment_id
                      ? zh
                        ? " · 智能体参考"
                        : " · Recorded reference"
                      : ""}
                  </option>
                ))}
              </select>
            </label>
            {previous && (
              <button
                className="text-primary underline"
                onClick={() => onSelect(previous)}
              >
                {zh ? "查看对照轮次" : "View reference round"}
              </button>
            )}
          </div>
          {source && source.experiment_id !== decision.base_experiment_id && (
            <p className="text-muted-foreground mt-2">
              {zh
                ? "这是手动对照，不代表智能体当时参考了这一轮。"
                : "Manual comparison; this was not the agent's recorded reference."}
            </p>
          )}
        </details>
      )}
    </section>
  );
}
