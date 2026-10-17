"use client";

import { useState } from "react";

import {
  record,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { type ProcessChart } from "@/core/nir/process-contract";
import { useProcessResource } from "@/core/nir/process-hooks";
import {
  parseRoundDiagnostics,
  type RoundPoints,
} from "@/core/nir/round-process";

import { Metrics } from "./autonomous-evidence";
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
    <div
      className="grid min-w-0 gap-4 xl:grid-cols-2"
      data-testid="round-linked-plots"
    >
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
            zh={zh}
            zoom={zoom}
            onZoom={setZoom}
            selected={selected}
            onSelect={setSelected}
          />
        );
      })}
    </div>
  );
}

export function RoundDiagnostics({
  threadId,
  summary,
  decision,
  zh,
}: {
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
  const selectedScope = scopes.find((item) => item[0] === scope)!;
  const points = charts?.[scope];
  const metrics = record(decision.feedback?.metrics);
  return (
    <section
      className="min-w-0 space-y-4"
      aria-label={zh ? "本轮指标与诊断" : "Round metrics and diagnostics"}
    >
      <label className="flex flex-wrap items-center gap-2 text-sm">
        {zh ? "评价范围" : "Evaluation scope"}
        <select
          className="bg-background max-w-full rounded-md border p-2"
          value={scope}
          onChange={(event) => setScope(event.target.value)}
        >
          {scopes.map(([value, , labelZh, labelEn]) => (
            <option key={value} value={value}>
              {zh ? labelZh : labelEn}
            </option>
          ))}
        </select>
      </label>
      <Metrics
        value={{
          metrics: { [selectedScope[1]]: metrics[selectedScope[1]] },
          stage: "searching",
        }}
      />
      {!Object.keys(record(metrics[selectedScope[1]])).length && (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "此范围尚无已完成的指标记录。"
            : "No completed metrics recorded for this scope."}
        </p>
      )}
      {points ? (
        <LinkedPlots
          key={`${decision.decision_id}:${scope}`}
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
