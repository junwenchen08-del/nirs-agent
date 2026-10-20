"use client";

import { useState } from "react";

import {
  parseDecision,
  record,
  text,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import { roundConfiguration } from "@/core/nir/round-process";

import { ProcessFacts } from "./process-facts";

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
  const configuration = roundConfiguration(decision);
  const before = previous ? roundConfiguration(previous) : null;
  const currentManifest = text(
    record(decision.feedback?.cv_settings).manifest_id,
  );
  const comparable = Boolean(
    summary.comparison_group_id &&
    currentManifest &&
    currentManifest ===
      text(record(previous?.feedback?.cv_settings).manifest_id) &&
    configuration.actual &&
    before?.actual,
  );
  const pipeline = (row: DecisionView) => {
    const value = roundConfiguration(row);
    const recordedSteps = record(row.feedback?.actual_configuration).steps;
    return [
      ...(value.steps.length
        ? value.steps.map((step) => methodLabel(text(record(step).method), zh))
        : [
            Array.isArray(recordedSteps)
              ? zh
                ? "原始光谱"
                : "Raw spectra"
              : zh
                ? "预处理未记录"
                : "Preprocessing not recorded",
          ]),
      methodLabel(value.method, zh),
    ].join(" → ");
  };
  return (
    <section
      className="space-y-3 rounded-xl border p-4"
      data-testid="round-comparison"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h4 className="font-medium">
          {zh ? "本轮与其他实验的变化" : "Changes against another experiment"}
        </h4>
        <label className="text-sm">
          <span className="mr-2">{zh ? "对照轮次" : "Compare with"}</span>
          <select
            className="bg-background rounded border p-2"
            value={source?.experiment_id ?? ""}
            onChange={(event) => setSelectedId(event.target.value)}
          >
            <option value="">
              {zh ? "选择已完成实验" : "Select a completed experiment"}
            </option>
            {options.map((row) => (
              <option key={row.experiment_id} value={row.experiment_id}>
                {zh
                  ? `第 ${row.round_index ?? "—"} 轮`
                  : `Round ${row.round_index ?? "—"}`}{" "}
                · {methodLabel(row.method, zh)}
                {row.experiment_id === decision.base_experiment_id
                  ? zh
                    ? " · 反馈来源"
                    : " · Feedback source"
                  : ""}
              </option>
            ))}
          </select>
        </label>
      </div>
      {!source ? (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "本轮未记录可用的前序反馈实验，可选择同一会话内其他已完成实验查看。"
            : "No available feedback source is recorded. Select another completed experiment in this session."}
        </p>
      ) : !previous ? (
        <p
          className="text-muted-foreground text-sm"
          role={query.isError ? "alert" : "status"}
        >
          {query.isPending
            ? zh
              ? "正在读取对照实验…"
              : "Reading comparison…"
            : zh
              ? "对照记录不可用或身份不匹配。"
              : "Comparison unavailable or identity mismatch."}
          {query.isError && (
            <button
              className="ml-2 underline"
              onClick={() => void query.refetch()}
            >
              {zh ? "重新读取" : "Retry"}
            </button>
          )}
        </p>
      ) : (
        <>
          <div className="grid gap-3 text-sm sm:grid-cols-2">
            <div className="bg-muted/30 rounded-lg p-3">
              <p className="text-muted-foreground mb-2 text-xs">
                {zh ? "对照方案" : "Reference plan"}
                {source.experiment_id === decision.base_experiment_id
                  ? zh
                    ? " · 反馈来源"
                    : " · Feedback source"
                  : ""}
              </p>
              {before?.actual
                ? pipeline(previous)
                : zh
                  ? "实际方案未记录"
                  : "Executed plan not recorded"}
            </div>
            <div className="bg-accent/40 rounded-lg p-3">
              <p className="text-muted-foreground mb-2 text-xs">
                {zh ? "本轮方案" : "Current plan"}
              </p>
              {configuration.actual
                ? pipeline(decision)
                : zh
                  ? "本轮尚无成功执行配置"
                  : "No successful configuration yet"}
            </div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <caption className="text-muted-foreground pb-2 text-left text-xs">
                {zh
                  ? "原始全量指标；变化为本轮减对照。"
                  : "Original full-evaluation metrics; change = current minus reference."}
              </caption>
              <thead>
                <tr>
                  <th className="py-2">{zh ? "评价指标" : "Metric"}</th>
                  <th>{zh ? "对照" : "Reference"}</th>
                  <th>{zh ? "本轮" : "Current"}</th>
                  <th>{zh ? "变化" : "Change"}</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["val", "RMSE", "Val RMSE"],
                  ["val", "R2", "Val R²"],
                  ["cal_cv", "RMSE", "Cal CV RMSE"],
                ].map(([scope, metric, label]) => {
                  const priorMetrics = record(
                    record(previous.feedback?.metrics)[scope!],
                  );
                  const currentMetrics = record(
                    record(decision.feedback?.metrics)[scope!],
                  );
                  const a = priorMetrics[metric!];
                  const b = currentMetrics[metric!];
                  const expectedScope =
                    scope === "val" ? "validation" : "calibration_cv";
                  const change =
                    comparable &&
                    priorMetrics.scope === expectedScope &&
                    currentMetrics.scope === expectedScope &&
                    typeof a === "number" &&
                    typeof b === "number"
                      ? b - a
                      : null;
                  return (
                    <tr key={label} className="border-t">
                      <th className="py-2 font-normal">{label}</th>
                      <td>{fmt(a)}</td>
                      <td>{fmt(b)}</td>
                      <td>
                        {change == null
                          ? "—"
                          : `${change > 0 ? "+" : ""}${fmt(change)}`}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {!comparable && (
            <p className="text-muted-foreground text-xs">
              {zh
                ? "缺少一致的划分／交叉验证依据，仅并列展示记录，不计算改善幅度。"
                : "Matching split/CV evidence is missing. Values are shown without improvement claims."}
            </p>
          )}
          <details className="text-sm">
            <summary className="cursor-pointer">
              {zh ? "对照实际参数" : "Compare fitted parameters"}
            </summary>
            <div className="mt-3 grid gap-4 sm:grid-cols-2">
              <div>
                <h5>{zh ? "对照轮次" : "Reference round"}</h5>
                <ProcessFacts
                  value={before?.actual ? before.parameters : {}}
                  zh={zh}
                />
              </div>
              <div>
                <h5>{zh ? "当前轮次" : "Current round"}</h5>
                <ProcessFacts
                  value={configuration.actual ? configuration.parameters : {}}
                  zh={zh}
                />
              </div>
            </div>
          </details>
          <button
            type="button"
            className="text-primary text-sm underline"
            onClick={() => onSelect(previous)}
          >
            {zh ? "查看对照轮次" : "View reference round"}
          </button>
        </>
      )}
    </section>
  );
}
