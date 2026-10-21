"use client";

import { useState, type ReactNode } from "react";

import {
  record,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { type ProcessChart } from "@/core/nir/process-contract";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import {
  parseRoundDiagnostics,
  type RoundPoints,
} from "@/core/nir/round-process";

import { ProcessChartView } from "./process-chart";

const scopes = [
  ["validation", "val", "验证集（Val）", "Validation (Val)"],
  [
    "calibration_cv",
    "cal_cv",
    "校准交叉验证（折外预测）",
    "Calibration CV (OOF)",
  ],
  ["calibration", "cal", "校准集拟合（Cal）", "Calibration fit (Cal)"],
] as const;

function LinkedPlots({
  points,
  sid,
  decision,
  zh,
}: {
  points: RoundPoints;
  sid: string;
  decision: DecisionView;
  zh: boolean;
}) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  return (
    <div className="min-w-0 space-y-3" data-testid="round-linked-plots">
      <div
        className="flex flex-wrap items-center gap-3 text-xs"
        data-testid="diagnostic-linked-controls"
      >
        <label className="flex items-center gap-2">
          {zh ? "选中样本" : "Selected sample"}
          <select
            aria-label={zh ? "选中样本" : "Selected sample"}
            className="bg-background max-w-full rounded border px-2 py-1"
            value={selected[0] ?? ""}
            onChange={(event) =>
              setSelected(event.target.value ? [event.target.value] : [])
            }
          >
            <option value="">{zh ? "全部样本" : "All samples"}</option>
            {points.sample_ids.map((id, index) => (
              <option key={id} value={id}>
                {zh ? `图中样本 ${index + 1}` : `Displayed sample ${index + 1}`}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-2">
          {zh ? "区间起点" : "Range start"}
          <input
            type="range"
            aria-label={zh ? "区间起点" : "Range start"}
            className="w-20"
            min="0"
            max="99"
            value={zoom[0] * 100}
            onChange={(event) =>
              setZoom([
                Math.min(Number(event.target.value) / 100, zoom[1] - 0.01),
                zoom[1],
              ])
            }
          />
        </label>
        <label className="flex items-center gap-2">
          {zh ? "区间终点" : "Range end"}
          <input
            type="range"
            aria-label={zh ? "区间终点" : "Range end"}
            className="w-20"
            min="1"
            max="100"
            value={zoom[1] * 100}
            onChange={(event) =>
              setZoom([
                zoom[0],
                Math.max(Number(event.target.value) / 100, zoom[0] + 0.01),
              ])
            }
          />
        </label>
        <button
          type="button"
          className="underline"
          onClick={() => {
            setZoom([0, 1]);
            setSelected([]);
          }}
        >
          {zh ? "重置图表" : "Reset charts"}
        </button>
        <span aria-live="polite">
          {zh ? "当前显示的所选点" : "Selected displayed points"}:{" "}
          {selected.length}
        </span>
      </div>
      <div className="process-plot-pair">
        {(["prediction", "residual"] as const).map((type) => {
          const chart: ProcessChart = {
            chart_schema_version: 1,
            chart_id: `${decision.experiment_id}_${points.scope}_${type}`,
            run_id: sid,
            attempt_id: decision.experiment_id!,
            step_execution_id: decision.decision_id,
            evidence_status: "available",
            type,
            scope: points.scope,
            x_label: "reference",
            y_label: type,
            x_unit: points.unit,
            y_unit: points.unit,
            x: points.observed,
            y: type === "prediction" ? points.predicted : points.residual,
            sample_ids: points.sample_ids,
            n_total: points.n_total,
            n_returned: points.n_returned,
            sampling: points.sampling,
          };
          return (
            <ProcessChartView
              key={type}
              chart={chart}
              compact
              zh={zh}
              zoom={zoom}
              onZoom={setZoom}
              selected={selected}
              onSelect={setSelected}
            />
          );
        })}
      </div>
    </div>
  );
}

export function RoundDiagnostics({
  threadId,
  summary,
  decision,
  zh,
  comparison,
}: {
  comparison?: ReactNode;
  threadId: string;
  summary: AutonomousSummary;
  decision: DecisionView;
  zh: boolean;
}) {
  const [scope, setScope] = useState<string>("validation");
  const completed =
    decision.status === "succeeded" &&
    !["STOP", "RETAIN"].includes(decision.action);
  const query = useProcessResource<unknown>(
    threadId,
    completed && decision.experiment_id
      ? `autonomous/${summary.session_id}/diagnostics/${decision.experiment_id}`
      : null,
  );
  const charts = parseRoundDiagnostics(
    query.data,
    summary.session_id,
    decision.experiment_id ?? "",
    decision.decision_id,
    summary.comparison_group_id,
  );
  const metrics = record(decision.feedback?.metrics);
  const recordedScopes = scopes.filter(
    (item) => Object.keys(record(metrics[item[1]])).length || charts?.[item[0]],
  );
  const options = recordedScopes.length ? recordedScopes : [...scopes];
  const currentScope =
    options.find((item) => item[0] === scope) ?? options[0] ?? scopes[0];
  const selectedScope = currentScope;
  const points = charts?.[currentScope[0]];
  const selectedMetrics = record(metrics[selectedScope[1]]);
  const validMetrics = completed && selectedMetrics.scope === selectedScope[0];
  return (
    <section
      className="min-w-0 space-y-4"
      aria-label={zh ? "本轮指标与诊断" : "Round metrics and diagnostics"}
    >
      <label className="flex flex-wrap items-center gap-2 text-sm">
        {zh ? "评价范围" : "Evaluation scope"}
        <select
          className="bg-background max-w-full rounded-md border p-2"
          value={currentScope[0]}
          onChange={(event) => setScope(event.target.value)}
        >
          {options.map(([value, , labelZh, labelEn]) => (
            <option key={value} value={value}>
              {zh ? labelZh : labelEn}
            </option>
          ))}
        </select>
      </label>
      <p className="text-muted-foreground text-xs">
        {zh ? "样本数：" : "Samples: "}
        {validMetrics ? fmt(selectedMetrics.n) : "—"}
      </p>
      {selectedScope[0] === "validation" && comparison ? (
        comparison
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {["RMSE", "R2"].map((key) => (
            <div key={key} className="bg-muted/30 rounded-lg p-3">
              <p className="text-muted-foreground text-xs">
                {key === "R2" ? "R² ↑" : "RMSE ↓"}
              </p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">
                {validMetrics ? fmt(selectedMetrics[key]) : "—"}
              </p>
            </div>
          ))}
        </div>
      )}
      {validMetrics && (
        <details className="text-xs">
          <summary className="cursor-pointer">
            {zh ? "更多指标" : "More metrics"}
          </summary>
          <dl className="mt-2 flex flex-wrap gap-5">
            {["RPD", "MAE", "bias"]
              .filter((key) => typeof selectedMetrics[key] === "number")
              .map((key) => (
                <div key={key}>
                  <dt className="text-muted-foreground">
                    {key === "bias" && zh ? "平均偏差" : key}
                  </dt>
                  <dd>{fmt(selectedMetrics[key])}</dd>
                </div>
              ))}
          </dl>
        </details>
      )}
      {!Object.keys(record(metrics[selectedScope[1]])).length && (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "此范围尚无已完成的指标记录。"
            : "No completed metrics recorded for this scope."}
        </p>
      )}
      {points ? (
        <LinkedPlots
          key={`${decision.decision_id}:${currentScope[0]}`}
          points={points}
          sid={summary.session_id}
          decision={decision}
          zh={zh}
        />
      ) : (
        <div
          className="rounded-lg border border-dashed p-5 text-sm"
          role={query.isError ? "alert" : "status"}
        >
          {!completed
            ? zh
              ? "本轮尚未成功执行，暂时没有诊断图。"
              : "Diagnostics require a successfully executed experiment."
            : query.isError
              ? zh
                ? "诊断图读取失败。"
                : "Could not read diagnostic evidence."
              : query.isPending
                ? zh
                  ? "正在读取本轮诊断图…"
                  : "Reading round diagnostics…"
                : record(query.data).evidence_status === "available" && !charts
                  ? zh
                    ? "诊断图身份或结构校验失败，未展示不匹配的数据。"
                    : "Diagnostic identity or schema mismatch; points were not displayed."
                  : record(query.data).evidence_status === "unavailable"
                    ? zh
                      ? "本轮图表文件缺失或损坏。"
                      : "The saved charts are missing or corrupt."
                    : record(query.data).evidence_status === "failed"
                      ? zh
                        ? "本轮训练已完成，但诊断图保存失败。"
                        : "Training succeeded, but saving diagnostics failed."
                      : zh
                        ? "本轮当时未记录可用的诊断图，已有指标保留。"
                        : "No usable diagnostic points were recorded for this round; metrics remain available."}
          {completed && !query.isPending && (
            <button
              type="button"
              className="mt-2 block underline"
              onClick={() => void query.refetch()}
            >
              {zh ? "重新读取图表" : "Retry chart read"}
            </button>
          )}
        </div>
      )}
      <p className="text-muted-foreground text-xs">
        {zh
          ? "散点图与残差图共享样本选择和实测值缩放；残差 = 预测 − 实测。指标来自全量评价，图表可能抽样。独立 Test 在最终评价中展示。"
          : "Scatter and residual plots share sample selection and reference-axis zoom. Residual = predicted − reference. Metrics cover the full evaluation; display points may be sampled. Independent Test is shown separately."}
      </p>
    </section>
  );
}
