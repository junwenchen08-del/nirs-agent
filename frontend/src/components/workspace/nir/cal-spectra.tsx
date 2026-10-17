"use client";

import { useState } from "react";

import { parseCalSpectra, type AutonomousSummary } from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { type ProcessChart } from "@/core/nir/process-contract";
import { useProcessResource } from "@/core/nir/process-hooks";

import { ProcessChartView } from "./process-chart";
import { ProcessFacts } from "./process-facts";

function CalPlot({ chart, zh }: { chart: ProcessChart; zh: boolean }) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
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

export function CalSpectraComparison({
  threadId,
  summary,
  zh,
  experimentId,
}: {
  threadId: string;
  summary: AutonomousSummary;
  zh: boolean;
  experimentId?: string;
}) {
  const [choice, setChoice] = useState("");
  const experiments = experimentId
    ? summary.spectra_experiments.filter(
        (item) => item.experiment_id === experimentId,
      )
    : summary.spectra_experiments;
  const current =
    experiments.find((item) => item.experiment_id === choice) ??
    experiments.find(
      (item) => item.experiment_id === summary.retained_experiment_id,
    ) ??
    experiments.at(-1);
  const query = useProcessResource<unknown>(
    threadId,
    current
      ? `autonomous/${summary.session_id}/spectra/${current.experiment_id}`
      : null,
  );
  const value = current
    ? parseCalSpectra(query.data, summary.session_id, current.experiment_id)
    : null;
  return (
    <section className="space-y-3 rounded-xl border p-4">
      <h3 className="font-medium">
        {zh ? "Cal 原始与预处理光谱" : "Raw and preprocessed Cal spectra"}
      </h3>
      <p className="text-muted-foreground text-sm">
        {zh
          ? "曲线仅来自 Cal，使用该实验实际拟合的预处理状态。查看、切换与刷新不会训练，也不会读取 Test。"
          : "Cal only, transformed by this experiment's fitted state. Viewing, switching and refreshing never trains or reads Test."}
      </p>
      {current && !experimentId && (
        <label className="flex items-center gap-2 text-sm">
          {zh ? "实验" : "Experiment"}
          <select
            className="bg-background rounded border p-2"
            value={current.experiment_id}
            onChange={(event) => setChoice(event.target.value)}
          >
            {experiments.map((item) => (
              <option key={item.experiment_id} value={item.experiment_id}>
                {item.round_index ?? "—"} · {methodLabel(item.method, zh)} ·{" "}
                {item.experiment_id.slice(-8)}
              </option>
            ))}
          </select>
        </label>
      )}
      {query.isError ? (
        <p role="alert">
          {zh ? "Cal 图表缓存读取失败。" : "Cal plot cache read failed."}
        </p>
      ) : value ? (
        <>
          <ProcessFacts
            value={{
              preprocessing: value.configuration.steps,
              model: value.configuration.method,
              experiment_id: current?.experiment_id,
              decision_id: value.decision_id,
            }}
            zh={zh}
          />
          {!value.before.x_unit && (
            <p className="text-muted-foreground text-xs">
              {zh
                ? "波长单位未注明；缺少波长轴时使用原始通道序号。"
                : "Axis units unspecified; original channel indices are used when no wavelength axis exists."}
            </p>
          )}
          <div className="grid gap-4 xl:grid-cols-2">
            <CalPlot
              key={`${current?.experiment_id}_before`}
              chart={value.before}
              zh={zh}
            />
            <CalPlot
              key={`${current?.experiment_id}_after`}
              chart={value.after}
              zh={zh}
            />
          </div>
        </>
      ) : (
        <p className="text-muted-foreground text-sm">
          {current && query.isPending
            ? zh
              ? "正在读取 Cal 光谱缓存…"
              : "Reading Cal spectra cache…"
            : zh
              ? "本实验未记录可用 Cal 光谱，不为历史运行重新计算。"
              : "No Cal spectra recorded; historical runs are not recomputed."}
        </p>
      )}
    </section>
  );
}
