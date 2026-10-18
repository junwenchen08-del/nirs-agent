"use client";
import { useState } from "react";

import {
  experimentMarkers,
  parseFinalChart,
  validationSeries,
  type AutonomousSummary,
} from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";

import { EvidenceChart } from "./autonomous-evidence";
import { plotLabel } from "./process-chart";
export function Trends({
  summary,
  zh,
}: {
  summary: AutonomousSummary;
  zh: boolean;
}) {
  const [metric, setMetric] = useState<"RMSE" | "R2">("RMSE");
  const series = validationSeries(summary, metric);
  return (
    <section className="space-y-3 rounded-xl border p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">
          {zh ? "实际实验的 Val 性能" : "Executed experiments: Val performance"}
        </h3>
        <select
          aria-label={zh ? "验证趋势指标" : "Validation trend metric"}
          value={metric}
          onChange={(event) => setMetric(event.target.value as "RMSE" | "R2")}
          className="bg-background rounded border p-1"
        >
          <option>RMSE</option>
          <option>R2</option>
        </select>
      </div>
      <p className="text-muted-foreground text-xs">
        {zh
          ? "仅包含同一封存会话中执行成功的实验；最低 Val RMSE 是信息标记，保留实验由 LLM 决定。"
          : "Only successful same-session experiments. The best Val RMSE marker is informational; LLM chooses retention."}
      </p>
      {series.length ? (
        <EvidenceChart
          key={metric}
          x={series.map((item) => item.round_index)}
          y={series.map((item) => item.value)}
          type="trend"
          scope={
            zh ? `验证集（Val）· ${plotLabel(metric, zh)}` : `Val ${metric}`
          }
          zh={zh}
        />
      ) : (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "暂无可比较的实际验证结果。"
            : "No comparable executed validation evidence yet."}
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              <th>{zh ? "轮次" : "Round"}</th>
              <th>{zh ? "实际模型" : "Actual model"}</th>
              <th>
                {zh ? `验证集 · ${plotLabel(metric, zh)}` : `Val ${metric}`}
              </th>
              <th>{zh ? "标记" : "Markers"}</th>
            </tr>
          </thead>
          <tbody>
            {series.map((item) => (
              <tr key={item.experiment_id} className="border-t">
                <td className="py-2">{item.round_index}</td>
                <td>{item.method.toUpperCase()}</td>
                <td>{fmt(item.value)}</td>
                <td>
                  {experimentMarkers(summary, item.experiment_id, zh).join(
                    " · ",
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function FinalCharts({
  threadId,
  summary,
  zh,
}: {
  threadId: string;
  summary: AutonomousSummary;
  zh: boolean;
}) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  const linked = { zoom, onZoom: setZoom, selected, onSelect: setSelected };
  const query = useProcessResource<unknown>(
    threadId,
    summary.stage === "finalized" && summary.chart_scopes.includes("final_test")
      ? `autonomous/${summary.session_id}/charts/final_test`
      : null,
  );
  if (query.isError)
    return (
      <p role="alert">
        {zh
          ? "最终 Test 图表证据读取失败；重新打开图表可重试读取。"
          : "Final Test plot read failed; reopen charts to retry the read."}
      </p>
    );
  if (query.isPending && summary.chart_scopes.includes("final_test"))
    return (
      <p>
        {zh
          ? "正在读取最终 Test 图表证据…"
          : "Reading final Test plot evidence…"}
      </p>
    );
  const value = parseFinalChart(query.data, summary.stage);
  if (!value)
    return (
      <p className="text-muted-foreground text-sm">
        {zh
          ? "最终 Test 图表尚未记录；封存期间不可读取。旧运行不补算图表。"
          : "Final Test plots not recorded. Sealed Test cannot be read; old runs are not recomputed."}
      </p>
    );
  return (
    <div className="process-plot-pair">
      <EvidenceChart
        x={value.observed}
        y={value.predicted}
        type="prediction"
        scope="final_test"
        zh={zh}
        ids={value.sample_ids}
        sampling={value.sampling}
        nTotal={value.n_total}
        linked={linked}
      />
      <EvidenceChart
        x={value.observed}
        y={value.residual}
        type="residual"
        scope="final_test"
        zh={zh}
        ids={value.sample_ids}
        sampling={value.sampling}
        nTotal={value.n_total}
        linked={linked}
      />
    </div>
  );
}
