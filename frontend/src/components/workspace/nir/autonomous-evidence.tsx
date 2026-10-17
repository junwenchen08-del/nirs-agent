"use client";

import { useState } from "react";

import { useI18n } from "@/core/i18n/hooks";
import {
  autonomousDecisionPageSize,
  metricRows,
  parameterTrials,
} from "@/core/nir/autonomous";
import { type ProcessChart } from "@/core/nir/process-contract";
import {
  factValue,
  formatProcessNumber as fmt,
} from "@/core/nir/process-display";

import { ProcessChartView } from "./process-chart";
import { ProcessFacts } from "./process-facts";

export function Metrics({ value }: { value: unknown }) {
  const { locale } = useI18n();
  const zh = locale === "zh-CN";
  const rows = metricRows(value);
  return rows.length ? (
    <div className="grid gap-3 md:grid-cols-2">
      {rows.map((row) => (
        <div key={row.label} className="rounded border p-3 text-sm">
          <strong>
            {zh
              ? ({
                  Cal: "校准集拟合（Cal）",
                  "Cal CV": "校准交叉验证（Cal CV，折外预测）",
                  Val: "验证集（Val）",
                  Test: "最终测试集（Test）",
                }[row.label ?? ""] ?? row.label)
              : row.label}
            {!zh && row.label === "Cal CV" ? " (OOF)" : ""}
          </strong>
          <span className="text-muted-foreground ml-2 text-xs">
            {String(factValue("scope", row.values.scope, zh))} ·{" "}
            {zh ? "样本数" : "n"}={fmt(row.values.n)}
          </span>
          <div className="mt-2 flex gap-4">
            {["RMSE", "R2", "RPD"].map((key) => (
              <span key={key}>
                {key === "R2" ? "R²" : key}:{" "}
                {key in row.values ? fmt(row.values[key]) : "—"}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  ) : null;
}

export function RecordedNarrative({
  value,
  zh,
}: {
  value: string;
  zh: boolean;
}) {
  if (!value) return <span>{zh ? "未记录" : "Not recorded"}</span>;
  if (!zh || /[\u3400-\u9fff]/u.test(value))
    return <span className="whitespace-pre-wrap">{value}</span>;
  return (
    <details className="text-sm">
      <summary className="text-muted-foreground cursor-pointer">
        历史理由为英文，展开查看原始记录
      </summary>
      <p className="mt-2 whitespace-pre-wrap" lang="en">
        {value}
      </p>
    </details>
  );
}

export function EvidenceChart({
  x,
  y,
  type,
  scope,
  zh,
  ids,
  sampling = "persisted_evidence",
  nTotal,
}: {
  x: number[];
  y: number[];
  type: string;
  scope: string;
  zh: boolean;
  ids?: string[];
  sampling?: string;
  nTotal?: number;
}) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  if (
    !x.length ||
    x.length !== y.length ||
    ![...x, ...y].every(Number.isFinite)
  )
    return <p>{zh ? "图表证据不可用。" : "Chart evidence unavailable."}</p>;
  const chart: ProcessChart = {
    chart_schema_version: 1,
    chart_id: `${type}_${scope}`,
    run_id: "",
    attempt_id: "",
    step_execution_id: "",
    evidence_status: "available",
    type,
    scope,
    x,
    y,
    x_label:
      type === "trend"
        ? "round"
        : type === "cv"
          ? "candidate_index"
          : "reference",
    y_label:
      type === "residual"
        ? "residual"
        : type === "cv"
          ? "RMSECV"
          : type === "trend"
            ? scope
            : "prediction",
    sample_ids: ids,
    n_total: nTotal ?? x.length,
    n_returned: x.length,
    sampling,
  };
  return (
    <ProcessChartView
      chart={chart}
      zh={zh}
      zoom={zoom}
      onZoom={setZoom}
      selected={selected}
      onSelect={setSelected}
    />
  );
}

export function ParameterTrials({
  value,
  zh,
}: {
  value: unknown;
  zh: boolean;
}) {
  const trials = parameterTrials(value);
  const [offset, setOffset] = useState(0);
  const page = trials.slice(offset, offset + autonomousDecisionPageSize);
  const points = page.filter(
    (trial): trial is typeof trial & { RMSECV: number } =>
      trial.RMSECV !== null,
  );
  return (
    <div className="mt-3 space-y-3">
      <p className="text-muted-foreground text-xs">
        {zh
          ? "只显示已记录的 Cal CV 参数评价；候选序号保留实际搜索顺序，折叠或翻页不会重新训练。"
          : "Recorded Cal CV evaluations only, in actual search order. Expanding or paging never retrains."}
      </p>
      {points.length > 0 && (
        <EvidenceChart
          x={points.map((trial) => trial.index)}
          y={points.map((trial) => trial.RMSECV)}
          type="cv"
          scope="calibration_cv"
          zh={zh}
          nTotal={trials.length}
        />
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              <th>{zh ? "候选序号" : "Candidate"}</th>
              <th>{zh ? "实际参数" : "Actual parameters"}</th>
              <th>{zh ? "Cal CV RMSE（折均值）" : "Cal CV mean-fold RMSE"}</th>
              <th>{zh ? "各折 RMSE" : "Fold RMSE"}</th>
            </tr>
          </thead>
          <tbody>
            {page.map((trial) => (
              <tr key={trial.index} className="border-t">
                <td className="py-2">{trial.index}</td>
                <td className="max-w-80 break-words">
                  <ProcessFacts value={trial.parameters} zh={zh} />
                </td>
                <td>{fmt(trial.RMSECV)}</td>
                <td>{trial.fold_rmse.map(fmt).join(" / ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!trials.length && (
        <p>
          {zh
            ? "没有可展示的 Cal CV 参数证据。"
            : "No Cal CV parameter evidence available."}
        </p>
      )}
      {trials.length > autonomousDecisionPageSize && (
        <div className="flex justify-between text-sm">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() =>
              setOffset(Math.max(0, offset - autonomousDecisionPageSize))
            }
          >
            {zh ? "上一页" : "Previous"}
          </button>
          <span>
            {offset + 1}–{offset + page.length} / {trials.length}
          </span>
          <button
            type="button"
            disabled={offset + page.length >= trials.length}
            onClick={() => setOffset(offset + autonomousDecisionPageSize)}
          >
            {zh ? "下一页" : "Next"}
          </button>
        </div>
      )}
    </div>
  );
}
