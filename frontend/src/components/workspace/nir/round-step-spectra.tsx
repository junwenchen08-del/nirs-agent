"use client";

import { useState } from "react";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  type AutonomousSummary,
  type CalSpectra,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { type ProcessChart } from "@/core/nir/process-contract";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { roundPipeline } from "@/core/nir/round-pipeline";

import { ProcessChartView } from "./process-chart";
import { RoundParameters } from "./round-recorded-content";

export type SpectraStatus =
  | "loading"
  | "error"
  | "available"
  | "missing"
  | "mismatch"
  | "failed"
  | "unavailable";
export function RoundStepSpectra({
  decision,
  summary,
  zh,
  selectedStep,
  spectra = null,
  spectraStatus = "missing",
  onRetrySpectra,
}: {
  decision: DecisionView;
  summary: AutonomousSummary;
  zh: boolean;
  selectedStep: string;
  spectra?: CalSpectra | null;
  spectraStatus?: SpectraStatus;
  onRetrySpectra?: () => void;
}) {
  const pipeline = roundPipeline(decision);
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  const [hidden, setHidden] = useState<string[]>([]);
  const [uniformY, setUniformY] = useState(false);
  const [comparison, setComparison] = useState<{
    index: number;
    raw: boolean;
  } | null>(null);
  const chartAt = (index: number) =>
    index < 0
      ? spectra?.before
      : (spectra?.steps?.[index]?.chart ??
        (spectra?.steps === null && index === pipeline.steps.length - 1
          ? spectra.after
          : undefined));
  const charts = spectra
    ? [
        spectra.before,
        ...(spectra.steps?.map((item) => item.chart) ?? [spectra.after]),
      ]
    : [];
  const values = charts.flatMap(
    (chart) =>
      chart.series
        ?.filter((series) => !hidden.includes(series.id))
        .flatMap((series) => series.values) ?? [],
  );
  const valueBounds: [number, number] | undefined = uniformY
    ? [Math.min(0, ...values), Math.max(0, ...values)]
    : undefined;
  const plot = (chart: ProcessChart, enlarged = false) => (
    <ProcessChartView
      chart={chart}
      zh={zh}
      zoom={zoom}
      onZoom={setZoom}
      selected={selected}
      onSelect={setSelected}
      hiddenSeries={hidden}
      onHiddenSeriesChange={setHidden}
      valueBounds={valueBounds}
      expanded={enlarged}
      compact={!enlarged}
    />
  );
  const missing = (index: number) => (
    <p
      className="text-muted-foreground rounded-lg border border-dashed p-5 text-sm"
      role={spectraStatus === "error" ? "alert" : undefined}
    >
      {spectraStatus === "loading"
        ? zh
          ? "正在读取已保存的光谱…"
          : "Reading saved spectra…"
        : spectraStatus === "error"
          ? zh
            ? "光谱读取失败。"
            : "Spectra read failed."
          : spectraStatus === "failed"
            ? zh
              ? "该次运行的光谱记录失败，建模结果仍可查看。"
              : "Spectra recording failed; the model results remain available."
            : spectraStatus === "unavailable"
              ? zh
                ? "已记录的光谱缓存缺失或校验失败。"
                : "The recorded spectra cache is missing or failed integrity verification."
              : spectraStatus === "mismatch"
                ? zh
                  ? "光谱与当前执行记录不匹配，未显示。"
                  : "Spectra do not match this execution."
                : !pipeline.actual
                  ? zh
                    ? "尚无成功执行，此步骤没有实际光谱。"
                    : "No successful execution or actual spectra."
                  : index >= 0 && spectra
                    ? zh
                      ? "该次运行未保存此步骤光谱；其他步骤的已有图表仍可查看。"
                      : "This intermediate output was not recorded. Other saved steps remain available."
                    : zh
                      ? "该次运行未记录可用光谱，不重新计算历史记录。"
                      : "No spectra recorded; historical runs are not recomputed."}
      {spectraStatus === "error" && (
        <button
          type="button"
          className="ml-2 underline"
          onClick={onRetrySpectra}
        >
          {zh ? "重试读取" : "Retry"}
        </button>
      )}
    </p>
  );
  const snapshots = [
    {
      id: "raw",
      title: zh ? "原始光谱 · Cal 输入" : "Raw spectra · Cal input",
      chart: spectra?.before,
      index: -1,
    },
    ...pipeline.steps.map((item, index) => ({
      id: item.id,
      title: `${methodLabel(item.method, zh)}${zh ? " 后光谱" : " output spectra"}`,
      chart: chartAt(index),
      index,
    })),
  ];
  const activeSnapshot = snapshots.find((item) => item.id === selectedStep);
  return (
    <div className="space-y-4" data-testid="pipeline-step-detail">
      {activeSnapshot?.chart && (
        <div
          className="bg-muted/20 space-y-2 rounded-lg p-3 text-xs"
          data-testid="step-spectra-controls"
        >
          <p className="text-muted-foreground">
            {zh
              ? "展示本步累计处理后的 Cal 光谱，可对照已有的步骤图。"
              : "Cumulative processing of the same Cal samples. Enlarge previews; wavelength and sample choices are linked."}
          </p>
          {spectra && (
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
              <label className="flex items-center gap-2">
                {zh ? "聚焦样本" : "Focus sample"}
                <select
                  aria-label={zh ? "聚焦样本" : "Focus sample"}
                  className="bg-background rounded border px-2 py-1"
                  value={selected[0] ?? ""}
                  onChange={(event) =>
                    setSelected(event.target.value ? [event.target.value] : [])
                  }
                >
                  <option value="">{zh ? "全部曲线" : "All curves"}</option>
                  {spectra.before.series?.map((series, i) => (
                    <option value={series.id} key={series.id}>
                      {zh ? `样本 ${i + 1}` : `Sample ${i + 1}`}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2">
                {zh ? "区间起点" : "Range start"}
                <input
                  aria-label={zh ? "区间起点" : "Range start"}
                  type="range"
                  min="0"
                  max="99"
                  className="w-20"
                  value={zoom[0] * 100}
                  onChange={(event) =>
                    setZoom([
                      Math.min(
                        Number(event.target.value) / 100,
                        zoom[1] - 0.01,
                      ),
                      zoom[1],
                    ])
                  }
                />
              </label>
              <label className="flex items-center gap-2">
                {zh ? "区间终点" : "Range end"}
                <input
                  aria-label={zh ? "区间终点" : "Range end"}
                  type="range"
                  min="1"
                  max="100"
                  className="w-20"
                  value={zoom[1] * 100}
                  onChange={(event) =>
                    setZoom([
                      zoom[0],
                      Math.max(
                        Number(event.target.value) / 100,
                        zoom[0] + 0.01,
                      ),
                    ])
                  }
                />
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={uniformY}
                  onChange={(event) => setUniformY(event.target.checked)}
                />
                {zh ? "统一纵轴范围" : "Use a shared vertical range"}
              </label>
              <button
                type="button"
                className="underline"
                onClick={() => {
                  setZoom([0, 1]);
                  setSelected([]);
                  setHidden([]);
                }}
              >
                {zh ? "重置图表" : "Reset charts"}
              </button>
              <details>
                <summary className="cursor-pointer">
                  {zh ? "样本显隐" : "Visible samples"}
                </summary>
                <div className="mt-2 flex flex-wrap gap-2">
                  {spectra.before.series?.map((series, i) => (
                    <button
                      type="button"
                      key={series.id}
                      aria-pressed={!hidden.includes(series.id)}
                      className={`rounded border px-2 py-1 ${hidden.includes(series.id) ? "opacity-40" : ""}`}
                      onClick={() =>
                        setHidden((previous) =>
                          previous.includes(series.id)
                            ? previous.filter((id) => id !== series.id)
                            : [...previous, series.id],
                        )
                      }
                    >
                      {zh ? `样本 ${i + 1}` : `Sample ${i + 1}`}
                    </button>
                  ))}
                </div>
              </details>
            </div>
          )}
          {spectra && !spectra.before.x_unit && (
            <p className="text-muted-foreground">
              {zh
                ? "波长单位未注明；缺少波长轴时显示变量序号。"
                : "Axis unit unspecified; absent wavelength axes use variable indices."}
            </p>
          )}
        </div>
      )}
      <div className="min-w-0" data-testid="round-spectra-grid">
        {snapshots
          .filter((item) => item.id === selectedStep)
          .map((item) => (
            <section
              key={item.id}
              className="min-w-0 space-y-3"
              data-testid="step-spectra"
              data-spectra-step={item.id}
            >
              <div className="round-spectra-heading">
                <h4 className="text-sm font-medium">{item.title}</h4>
                <p className="text-muted-foreground text-xs">
                  {item.index < 0
                    ? zh
                      ? `Cal ${fmt(summary.data_context.calibration_samples)} · ${fmt(summary.data_context.n_features)} 个光谱变量`
                      : `Cal ${fmt(summary.data_context.calibration_samples)} · ${fmt(summary.data_context.n_features)} variables`
                    : item.index === 0
                      ? zh
                        ? "输入：原始光谱"
                        : "Input: raw spectra"
                      : zh
                        ? `输入：步骤 ${item.index} 的输出`
                        : `Input: step ${item.index} output`}
                </p>
              </div>
              {item.chart ? plot(item.chart) : missing(item.index)}
              <div className="flex flex-wrap items-start gap-x-3 gap-y-2 text-xs">
                {item.chart && item.index >= 0 && (
                  <>
                    {chartAt(item.index - 1) && (
                      <button
                        type="button"
                        className="text-primary underline"
                        onClick={() =>
                          setComparison({ index: item.index, raw: false })
                        }
                      >
                        {zh ? "对照上一步" : "Compare previous step"}
                      </button>
                    )}
                    <button
                      type="button"
                      className="text-primary underline"
                      onClick={() =>
                        setComparison({ index: item.index, raw: true })
                      }
                    >
                      {zh ? "对照原始光谱" : "Compare raw spectra"}
                    </button>
                  </>
                )}
                {item.index >= 0 && (
                  <details>
                    <summary className="cursor-pointer">
                      {pipeline.actual
                        ? zh
                          ? "本步实际参数"
                          : "Executed parameters"
                        : zh
                          ? "本步计划参数"
                          : "Planned parameters"}
                    </summary>
                    <RoundParameters
                      value={pipeline.steps[item.index]?.params}
                      zh={zh}
                    />
                  </details>
                )}
              </div>
            </section>
          ))}
      </div>
      <Dialog
        open={comparison !== null}
        onOpenChange={(open) => {
          if (!open) setComparison(null);
        }}
      >
        <DialogContent
          aria-describedby={undefined}
          className="max-h-[90dvh] overflow-auto sm:max-w-6xl"
        >
          <DialogHeader>
            <DialogTitle>
              {zh ? "本步光谱变化对照" : "Step spectra comparison"}
            </DialogTitle>
          </DialogHeader>
          {comparison && (
            <div
              className="process-plot-pair"
              data-testid="step-spectra-comparison"
            >
              {[
                chartAt(comparison.raw ? -1 : comparison.index - 1),
                chartAt(comparison.index),
              ].map(
                (chart, i) =>
                  chart && (
                    <div key={i} className="min-w-0">
                      <p className="mb-2 text-sm font-medium">
                        {i === 1
                          ? zh
                            ? "本步输出"
                            : "Step output"
                          : comparison.raw || comparison.index === 0
                            ? zh
                              ? "原始输入"
                              : "Raw input"
                            : zh
                              ? "上一步输出"
                              : "Previous output"}
                      </p>
                      {plot(chart, true)}
                    </div>
                  ),
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
