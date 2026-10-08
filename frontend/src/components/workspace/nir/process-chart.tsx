"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  displayedPointSummary,
  type ProcessChart,
} from "@/core/nir/process-contract";
import { formatProcessNumber } from "@/core/nir/process-display";
import { cn } from "@/lib/utils";

const colors = [
  "#2563eb",
  "#059669",
  "#d97706",
  "#9333ea",
  "#dc2626",
  "#0891b2",
];
const height = 330,
  left = 64,
  top = 24,
  bottom = 270;
const fmt = formatProcessNumber;
export const plotLabel = (value: string | undefined, zh: boolean): string => {
  const labels: Record<string, [string, string]> = {
    spectra: ["光谱曲线", "Spectra"],
    distribution: ["目标值分布", "Target distribution"],
    prediction: ["实测值与预测值", "Reference vs prediction"],
    residual: ["残差诊断", "Residual diagnostics"],
    wavelength: ["保留波长区间", "Selected wavelength intervals"],
    cv: ["交叉验证误差", "Cross-validation error"],
    all: ["全体样本", "All samples"],
    calibration: ["校准集", "Calibration"],
    tuning: ["调参集", "Tuning"],
    holdout: ["留出集", "Holdout"],
    external: ["外部验证", "External"],
    calibration_cv: ["校准集交叉验证", "Calibration CV"],
    final_fit_cv: ["最终拟合集交叉验证", "Final fitting set CV"],
    selected_variables: ["保留变量", "Retained variables"],
    calibration_before: ["处理前 · 校准集", "Before · calibration"],
    calibration_after: ["处理后 · 校准集", "After · calibration"],
    wavelength_axis: ["波长", "Wavelength"],
    variable: ["变量序号", "Variable index"],
    reference: ["实测值", "Reference"],
    spectral_value: ["光谱数值", "Spectral value"],
    count: ["样本数", "Count"],
    components: ["成分数", "Components"],
  };
  return labels[value ?? ""]?.[zh ? 0 : 1] ?? value ?? "—";
};

const axisLabel = (value: string | undefined, zh: boolean) =>
  value === "prediction"
    ? zh
      ? "预测值"
      : "Prediction"
    : value === "residual"
      ? zh
        ? "残差：预测 − 实测"
        : "Residual: predicted − reference"
      : plotLabel(value === "wavelength" ? "wavelength_axis" : value, zh);

export function ProcessChartView({
  chart,
  zh,
  zoom,
  onZoom,
  selected,
  onSelect,
}: {
  chart: ProcessChart;
  zh: boolean;
  zoom: [number, number];
  onZoom: (range: [number, number]) => void;
  selected: string[];
  onSelect: (ids: string[]) => void;
}) {
  const [hidden, setHidden] = useState<string[]>([]);
  const [hover, setHover] = useState<{ index: number; series: number } | null>(
    null,
  );
  const [brush, setBrush] = useState<{
    from: [number, number];
    to: [number, number];
  } | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(800);
  const right = width - 20;
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry && entry.contentRect.width > 0)
        setWidth(Math.max(280, entry.contentRect.width));
    });
    observer.observe(svg);
    return () => observer.disconnect();
  }, []);
  const lineSeries = useMemo(
    () =>
      chart.series ?? (chart.y ? [{ id: chart.type, values: chart.y }] : []),
    [chart],
  );
  const xs = useMemo(
    () =>
      chart.x ??
      chart.ranges?.flatMap((range) => [range.start, range.end]) ??
      [],
    [chart],
  );
  const fullLow = chart.axis_bounds?.[0] ?? (xs.length ? Math.min(...xs) : 0);
  const fullHigh = chart.axis_bounds?.[1] ?? (xs.length ? Math.max(...xs) : 1);
  const span = fullHigh - fullLow || 1;
  const low = fullLow + zoom[0] * span,
    high = fullLow + zoom[1] * span;
  const ys = lineSeries
    .filter((series) => !hidden.includes(series.id))
    .flatMap((series) =>
      series.values.filter(
        (_, index) => xs[index]! >= low && xs[index]! <= high,
      ),
    );
  const yMin = Math.min(0, ...ys),
    yMax = Math.max(0, ...ys);
  const ySpan = yMax - yMin || 1;
  const sx = (x: number) =>
    left + ((x - low) / (high - low || 1)) * (right - left);
  const sy = (y: number) => bottom - ((y - yMin) / ySpan) * (bottom - top);
  const curvePaths = useMemo(
    () =>
      lineSeries.map((series) =>
        series.values
          .map(
            (y, index) =>
              `${left + ((xs[index]! - low) / (high - low || 1)) * (width - 20 - left)},${bottom - ((y - yMin) / ySpan) * (bottom - top)}`,
          )
          .join(" "),
      ),
    [lineSeries, xs, low, high, width, yMin, ySpan],
  );
  const scatter = ["prediction", "residual"].includes(chart.type);
  const point = (
    event: React.PointerEvent<SVGSVGElement>,
  ): [number, number] => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) / rect.width) * width,
      ((event.clientY - rect.top) / rect.height) * height,
    ];
  };
  const summary = displayedPointSummary(chart, selected);
  if (chart.evidence_status !== "available")
    return (
      <p className="rounded-lg border border-dashed p-6 text-sm">
        {zh
          ? "图表证据不可用：文件缺失或校验失败。"
          : "Chart evidence is unavailable: missing or failed integrity verification."}
      </p>
    );
  return (
    <section
      className="bg-card min-w-0 rounded-xl border p-4"
      aria-label={`${plotLabel(chart.type, zh)} ${plotLabel(chart.scope, zh)}`}
    >
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <h3 className="font-medium">{plotLabel(chart.type, zh)}</h3>
        <span className="text-muted-foreground text-xs">
          {plotLabel(chart.scope, zh)} ·{" "}
          {chart.type === "distribution"
            ? `${chart.n_returned} ${zh ? "分箱" : "bins"} / ${chart.n_total} ${zh ? "样本" : "samples"}`
            : `${chart.n_returned}/${chart.n_total} ${chart.type === "wavelength" ? (zh ? "保留变量 / 全部变量" : "retained / total variables") : chart.type === "cv" ? (zh ? "显示候选 / 记录候选" : "displayed / recorded candidates") : zh ? "显示样本 / 全量样本" : "displayed / total samples"}`}
          {chart.axis_total !== undefined &&
            ` · ${chart.axis_returned}/${chart.axis_total} ${zh ? "显示变量 / 全部变量" : "displayed / total variables"}`}
        </span>
      </div>
      <div
        className="text-muted-foreground flex min-h-7 flex-wrap gap-2 text-xs"
        aria-live="polite"
      >
        {hover ? (
          <span>
            {plotLabel(
              chart.x_label === "wavelength"
                ? "wavelength_axis"
                : chart.x_label,
              zh,
            )}
            : {fmt(xs[hover.index])} {chart.x_unit} ·{" "}
            {axisLabel(chart.y_label, zh)}:{" "}
            {fmt(lineSeries[hover.series]?.values[hover.index])} {chart.y_unit}
            {chart.sample_ids?.[hover.index]
              ? ` · #${chart.sample_ids[hover.index]}`
              : ""}
          </span>
        ) : (
          <span>
            {zh
              ? "悬停读取数值；拖动框选显示点；下方可缩放和重置。"
              : "Hover for values; drag to select displayed points; zoom or reset below."}
          </span>
        )}
      </div>
      <svg
        ref={svgRef}
        role="img"
        aria-label={plotLabel(chart.type, zh)}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full touch-none select-none"
        onPointerDown={(event) => {
          if (scatter) {
            const pos = point(event);
            event.currentTarget.setPointerCapture(event.pointerId);
            setBrush({ from: pos, to: pos });
          }
        }}
        onPointerMove={(event) => {
          const pos = point(event);
          if (brush) setBrush({ ...brush, to: pos });
          let closest: {
            distance: number;
            index: number;
            series: number;
          } | null = null;
          lineSeries.forEach((series, seriesIndex) => {
            if (hidden.includes(series.id)) return;
            series.values.forEach((y, index) => {
              const x = xs[index];
              if (x === undefined || x < low || x > high) return;
              const distance =
                Math.abs(sx(x) - pos[0]) + Math.abs(sy(y) - pos[1]);
              if (!closest || distance < closest.distance)
                closest = { distance, index, series: seriesIndex };
            });
          });
          setHover(closest);
        }}
        onPointerUp={() => {
          if (!brush) return;
          const moved =
            Math.abs(brush.from[0] - brush.to[0]) +
              Math.abs(brush.from[1] - brush.to[1]) >
            5;
          if (moved)
            onSelect(
              (chart.sample_ids ?? []).filter((_, index) => {
                const x = sx(xs[index]!),
                  y = sy(chart.y?.[index] ?? 0);
                return (
                  x >= Math.min(brush.from[0], brush.to[0]) &&
                  x <= Math.max(brush.from[0], brush.to[0]) &&
                  y >= Math.min(brush.from[1], brush.to[1]) &&
                  y <= Math.max(brush.from[1], brush.to[1])
                );
              }),
            );
          else if (hover && chart.sample_ids?.[hover.index])
            onSelect([chart.sample_ids[hover.index]!]);
          setBrush(null);
        }}
        onPointerCancel={() => setBrush(null)}
        onPointerLeave={() => {
          if (!brush) setHover(null);
        }}
      >
        <defs>
          <clipPath id={`clip-${chart.chart_id}`}>
            <rect x={left} y={top} width={right - left} height={bottom - top} />
          </clipPath>
        </defs>
        {[0, 1, 2, 3, 4].map((tick) => (
          <g key={tick}>
            <line
              x1={left}
              x2={right}
              y1={top + (tick / 4) * (bottom - top)}
              y2={top + (tick / 4) * (bottom - top)}
              stroke="currentColor"
              opacity=".08"
            />
            {chart.type !== "wavelength" && (
              <text
                x={left - 8}
                y={top + (tick / 4) * (bottom - top) + 4}
                textAnchor="end"
                fill="currentColor"
                fontSize="11"
              >
                {fmt(yMax - (tick / 4) * ySpan)}
              </text>
            )}
            <text
              x={left + (tick / 4) * (right - left)}
              y={bottom + 22}
              textAnchor="middle"
              fill="currentColor"
              fontSize="11"
            >
              {fmt(low + (tick / 4) * (high - low))}
            </text>
          </g>
        ))}
        <line
          x1={left}
          x2={right}
          y1={bottom}
          y2={bottom}
          stroke="currentColor"
          opacity=".3"
        />
        <text
          x={(left + right) / 2}
          y={height - 8}
          textAnchor="middle"
          fill="currentColor"
          fontSize="12"
        >
          {plotLabel(
            chart.x_label === "wavelength" ? "wavelength_axis" : chart.x_label,
            zh,
          )}
          {chart.x_unit ? ` (${chart.x_unit})` : ""}
        </text>
        {chart.type !== "wavelength" && (
          <text
            transform={`translate(14 ${(top + bottom) / 2}) rotate(-90)`}
            textAnchor="middle"
            fill="currentColor"
            fontSize="12"
          >
            {axisLabel(chart.y_label, zh)}
            {chart.y_unit ? ` (${chart.y_unit})` : ""}
          </text>
        )}
        <g clipPath={`url(#clip-${chart.chart_id})`}>
          {chart.type === "prediction" && (
            <line
              x1={sx(low)}
              x2={sx(high)}
              y1={sy(low)}
              y2={sy(high)}
              stroke="#64748b"
              strokeDasharray="5 5"
            />
          )}
          {chart.type === "residual" && (
            <line
              x1={left}
              x2={right}
              y1={sy(0)}
              y2={sy(0)}
              stroke="#64748b"
              strokeDasharray="5 5"
            />
          )}
          {chart.ranges?.map((range, index) => (
            <g key={index}>
              <rect
                x={Math.min(sx(range.start), sx(range.end))}
                y={top + 65}
                width={Math.max(3, Math.abs(sx(range.end) - sx(range.start)))}
                height={80}
                fill="#059669"
                opacity=".6"
              >
                <title>
                  {range.start}–{range.end} · {range.count}
                </title>
              </rect>
            </g>
          ))}
          {lineSeries.map((series, seriesIndex) =>
            hidden.includes(series.id) ? null : scatter ? (
              series.values.map((y, index) => (
                <circle
                  key={index}
                  cx={sx(xs[index]!)}
                  cy={sy(y)}
                  r={
                    selected.includes(chart.sample_ids?.[index] ?? "") ? 6 : 3.5
                  }
                  fill={
                    selected.includes(chart.sample_ids?.[index] ?? "")
                      ? "#d97706"
                      : colors[seriesIndex % colors.length]
                  }
                  opacity=".8"
                  tabIndex={0}
                  role="button"
                  aria-label={`${chart.sample_ids?.[index]}: ${fmt(xs[index])}, ${fmt(y)}`}
                  onFocus={() => setHover({ index, series: seriesIndex })}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect([chart.sample_ids?.[index] ?? ""]);
                    }
                  }}
                />
              ))
            ) : chart.type === "distribution" ? (
              series.values.map((y, index) => (
                <rect
                  key={index}
                  x={
                    sx(xs[index]!) - (right - left) / Math.max(1, xs.length) / 2
                  }
                  y={sy(y)}
                  width={Math.max(
                    2,
                    (right - left) / Math.max(1, xs.length) - 3,
                  )}
                  height={bottom - sy(y)}
                  fill={colors[seriesIndex % colors.length]}
                  opacity=".7"
                />
              ))
            ) : (
              <polyline
                key={series.id}
                fill="none"
                stroke={colors[seriesIndex % colors.length]}
                strokeWidth="1.7"
                opacity=".8"
                points={curvePaths[seriesIndex]}
              />
            ),
          )}
          {brush && (
            <rect
              x={Math.min(brush.from[0], brush.to[0])}
              y={Math.min(brush.from[1], brush.to[1])}
              width={Math.abs(brush.from[0] - brush.to[0])}
              height={Math.abs(brush.from[1] - brush.to[1])}
              fill="#2563eb"
              opacity=".15"
              stroke="#2563eb"
            />
          )}
        </g>
      </svg>
      {chart.series && (
        <div className="mb-3 flex flex-wrap gap-1.5">
          {chart.series.map((series, index) => (
            <button
              key={series.id}
              type="button"
              aria-pressed={!hidden.includes(series.id)}
              className={cn(
                "rounded border px-2 py-1 text-xs",
                hidden.includes(series.id) && "opacity-40",
              )}
              onClick={() =>
                setHidden((previous) =>
                  previous.includes(series.id)
                    ? previous.filter((id) => id !== series.id)
                    : [...previous, series.id],
                )
              }
            >
              <span style={{ color: colors[index % colors.length] }}>● </span>#
              {series.id.slice(0, 6)}
            </button>
          ))}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-3 border-t pt-3 text-xs">
        <label className="flex items-center gap-2">
          {zh ? "区间起点" : "Range start"}
          <input
            aria-label={zh ? "区间起点" : "Range start"}
            type="range"
            min="0"
            max="99"
            value={zoom[0] * 100}
            onChange={(event) =>
              onZoom([
                Math.min(Number(event.target.value) / 100, zoom[1] - 0.01),
                zoom[1],
              ])
            }
            className="w-24"
          />
        </label>
        <label className="flex items-center gap-2">
          {zh ? "区间终点" : "Range end"}
          <input
            aria-label={zh ? "区间终点" : "Range end"}
            type="range"
            min="1"
            max="100"
            value={zoom[1] * 100}
            onChange={(event) =>
              onZoom([
                zoom[0],
                Math.max(Number(event.target.value) / 100, zoom[0] + 0.01),
              ])
            }
            className="w-24"
          />
        </label>
        <button
          type="button"
          className="rounded border px-2 py-1"
          onClick={() => {
            onZoom([0, 1]);
            onSelect([]);
            setHidden([]);
          }}
        >
          {zh ? "重置图表" : "Reset chart"}
        </button>
      </div>
      {scatter && (
        <p className="mt-3 text-sm" aria-live="polite">
          {zh ? "当前显示的所选点" : "Selected displayed points"}:{" "}
          {summary.count}
          {summary.mean !== null
            ? ` · ${zh ? "纵轴均值" : "Mean y"}: ${fmt(summary.mean)} ${chart.y_unit ?? ""}`
            : ""}
        </p>
      )}
      <p className="text-muted-foreground mt-2 text-xs">
        {chart.sampling.includes("histogram")
          ? zh
            ? "直方图统计来自全量样本。"
            : "Histogram counts use all samples."
          : chart.type === "wavelength"
            ? zh
              ? "区间记录精确保留的变量；缩放不会改变实际选择。"
              : "Intervals describe the exact retained variables; zoom does not change the selection."
            : chart.n_returned < chart.n_total ||
                (chart.axis_returned ?? 0) < (chart.axis_total ?? 0)
              ? zh
                ? "已抽样显示，框选仅统计显示点；全量指标另列。"
                : "Sampled display; selection summarizes displayed points. Full-population metrics are separate."
              : zh
                ? "本图覆盖所标范围的全部记录。"
                : "All records in the labeled scope are shown."}
      </p>
      <details className="mt-3 text-xs">
        <summary className="cursor-pointer">
          {zh ? "数据表与抽样说明" : "Data table and sampling"}
        </summary>
        <p className="mt-2 break-all">{chart.sampling}</p>
        <div className="mt-2 max-h-48 overflow-auto">
          <table className="w-full text-left">
            <thead>
              <tr>
                <th>{zh ? "标识" : "ID"}</th>
                <th>{axisLabel(chart.x_label, zh)}</th>
                <th>{axisLabel(chart.y_label, zh)}</th>
              </tr>
            </thead>
            <tbody>
              {xs.slice(0, 2000).map((x, index) => (
                <tr key={index} className="border-b">
                  <td className="py-1">
                    {chart.sample_ids?.[index]?.slice(0, 8) ?? index + 1}
                  </td>
                  <td>{fmt(x)}</td>
                  <td>{fmt(lineSeries[0]?.values[index])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {chart.ranges?.map((range, index) => (
            <p key={index}>
              {range.start}–{range.end}: {range.count}
            </p>
          ))}
        </div>
      </details>
    </section>
  );
}
