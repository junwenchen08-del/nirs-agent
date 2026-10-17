"use client";

import { chartColors, chartInk, seriesColor } from "@/core/nir/chart-theme";
import { formatProcessNumber } from "@/core/nir/process-display";
import { chartMetric, scopeLabel } from "@/core/nir-library/model-report";
import type { ModelChart, ModelMetric } from "@/core/nir-library/types";

const tick = (value: number) =>
  Number(value.toPrecision(4)).toLocaleString(undefined, {
    maximumSignificantDigits: 4,
  });
const ticks = (min: number, max: number) =>
  Array.from({ length: 5 }, (_, i) => min + ((max - min) * i) / 4);
const unitLabel = (label: string, unit?: string | null) =>
  unit ? `${label}（${unit}）` : label;

export function PredictionFigure({
  chart,
  metrics,
  method,
  zh,
}: {
  chart: ModelChart;
  metrics: ModelMetric[];
  method: string;
  zh: boolean;
}) {
  const text = (cn: string, en: string) => (zh ? cn : en);
  const metric = chartMetric(chart, metrics);
  const values = [...chart.x, ...chart.y!];
  const lower = Math.min(...values),
    upper = Math.max(...values);
  const pad = (upper - lower || Math.abs(lower) || 1) * 0.06;
  const min = lower - pad,
    max = upper + pad;
  const x = (v: number) => 86 + ((v - min) / (max - min)) * 350;
  const y = (v: number) => 420 - ((v - min) / (max - min)) * 350;
  const training = chart.scope === "calibration";
  const color = training ? seriesColor(0) : seriesColor(1);
  const title = `${method} · ${scopeLabel(chart.scope, zh)}`;
  return (
    <figure
      className="min-w-0 rounded-xl border bg-white p-3 text-slate-900"
      data-testid="model-prediction-figure"
    >
      <svg
        viewBox="0 0 500 500"
        className="block w-full"
        role="img"
        aria-label={text(
          `${title}真实值与预测值对比图`,
          `${title}: observed versus predicted`,
        )}
      >
        <title>{title}</title>
        <text x="250" y="24" textAnchor="middle" fontSize="17" fontWeight="600">
          {title}
        </text>
        <text x="250" y="49" textAnchor="middle" fontSize="14">
          R² = {formatProcessNumber(metric?.R2)} · RMSE ={" "}
          {formatProcessNumber(metric?.RMSE)}
        </text>
        <rect
          x="86"
          y="70"
          width="350"
          height="350"
          fill="none"
          stroke={chartInk.axis}
          strokeWidth="1.2"
        />
        {ticks(min, max).map((v, i) => (
          <g key={i} fontSize="12" fill={chartInk.tick}>
            <path
              d={`M ${x(v)} 420 v 5 M 81 ${y(v)} h 5`}
              stroke={chartInk.axis}
            />
            <text x={x(v)} y="442" textAnchor="middle">
              {tick(v)}
            </text>
            <text x="75" y={y(v) + 4} textAnchor="end">
              {tick(v)}
            </text>
          </g>
        ))}
        <line
          x1="86"
          y1="420"
          x2="436"
          y2="70"
          stroke={chartColors.reference}
          strokeWidth="2"
        />
        {chart.x.map((v, i) =>
          training ? (
            <circle
              key={i}
              cx={x(v)}
              cy={y(chart.y![i]!)}
              r="3.6"
              fill={color}
              opacity="0.8"
            >
              <title>
                {text("真实值", "Observed")}: {v}; {text("预测值", "Predicted")}
                : {chart.y![i]}
              </title>
            </circle>
          ) : (
            <path
              key={i}
              d="M0 -4.8 L1.4 -1.5 L4.8 -1.5 L2 0.8 L3.1 4.2 L0 2.2 L-3.1 4.2 L-2 0.8 L-4.8 -1.5 L-1.4 -1.5 Z"
              transform={`translate(${x(v)},${y(chart.y![i]!)})`}
              fill={color}
              opacity="0.82"
            >
              <title>
                {text("真实值", "Observed")}: {v}; {text("预测值", "Predicted")}
                : {chart.y![i]}
              </title>
            </path>
          ),
        )}
        <text x="261" y="474" textAnchor="middle" fontSize="15">
          {unitLabel(text("真实值", "Observed value"), chart.x_unit)}
        </text>
        <text
          transform="translate(23 245) rotate(-90)"
          textAnchor="middle"
          fontSize="15"
        >
          {unitLabel(text("预测值", "Predicted value"), chart.y_unit)}
        </text>
        <rect
          x="259"
          y="356"
          width="165"
          height="52"
          rx="3"
          fill="white"
          fillOpacity="0.9"
          stroke={chartInk.border}
        />
        {training ? (
          <circle cx="275" cy="373" r="4" fill={color} />
        ) : (
          <path
            d="M0 -4.8 L1.4 -1.5 L4.8 -1.5 L2 0.8 L3.1 4.2 L0 2.2 L-3.1 4.2 L-2 0.8 L-4.8 -1.5 L-1.4 -1.5 Z"
            transform="translate(275,373)"
            fill={color}
          />
        )}
        <text x="289" y="377" fontSize="12">
          {scopeLabel(chart.scope, zh)}
        </text>
        <line
          x1="267"
          y1="392"
          x2="283"
          y2="392"
          stroke={chartColors.reference}
          strokeWidth="2"
        />
        <text x="289" y="396" fontSize="12">
          {text("预测完全一致", "Perfect agreement")}
        </text>
      </svg>
      <figcaption className="space-y-1 px-3 pb-2 text-xs leading-relaxed text-slate-600">
        <p>
          {text(
            "点越靠近红线，预测越接近真实值。",
            "Points closer to the red line have predictions closer to their observed values.",
          )}
        </p>
        <p>
          {text(
            `展示 ${chart.n_returned} / ${chart.n_total} 个已记录样本。`,
            `Showing ${chart.n_returned} / ${chart.n_total} recorded samples.`,
          )}
        </p>
      </figcaption>
    </figure>
  );
}

export function SpectraFigure({
  chart,
  zh,
}: {
  chart: ModelChart;
  zh: boolean;
}) {
  const text = (cn: string, en: string) => (zh ? cn : en);
  const first = chart.x[0]!,
    last = chart.x.at(-1)!;
  const ys = chart.series!.flatMap((curve) => curve.values);
  const lower = Math.min(...ys),
    upper = Math.max(...ys);
  const pad = (upper - lower || Math.abs(lower) || 1) * 0.06;
  const min = lower - pad,
    max = upper + pad;
  const x = (v: number) => 80 + ((v - first) / (last - first || 1)) * 370;
  const y = (v: number) => 300 - ((v - min) / (max - min)) * 240;
  const title = scopeLabel(chart.scope, zh);
  const axisLabel =
    chart.x_label === "wavelength"
      ? unitLabel(
          text("光谱位置", "Spectral position"),
          chart.x_unit ?? text("单位未记录", "unit not recorded"),
        )
      : text("光谱变量编号", "Spectral variable index");
  return (
    <figure
      className="min-w-0 rounded-xl border bg-white p-3 text-slate-900"
      data-testid="model-spectra-figure"
    >
      <svg
        viewBox="0 0 500 380"
        className="block w-full"
        role="img"
        aria-label={text(`${title}光谱图`, `${title} spectra`)}
      >
        <title>{title}</title>
        <text x="250" y="28" textAnchor="middle" fontSize="17" fontWeight="600">
          {title}
        </text>
        <rect
          x="80"
          y="60"
          width="370"
          height="240"
          fill="none"
          stroke={chartInk.axis}
          strokeWidth="1.2"
        />
        {ticks(first, last).map((v, i) => (
          <g key={i} fontSize="12" fill={chartInk.tick}>
            <path d={`M ${x(v)} 300 v 5`} stroke={chartInk.axis} />
            <text x={x(v)} y="324" textAnchor="middle">
              {tick(v)}
            </text>
          </g>
        ))}
        {ticks(min, max).map((v, i) => (
          <g key={i} fontSize="12" fill={chartInk.tick}>
            <path d={`M 75 ${y(v)} h 5`} stroke={chartInk.axis} />
            <text x="69" y={y(v) + 4} textAnchor="end">
              {tick(v)}
            </text>
          </g>
        ))}
        {chart.series!.map((curve, index) => (
          <path
            key={curve.id}
            fill="none"
            stroke={seriesColor(index)}
            opacity="0.75"
            strokeWidth="1.2"
            d={curve.values
              .map((v, i) => `${i ? "L" : "M"} ${x(chart.x[i]!)} ${y(v)}`)
              .join(" ")}
          >
            <title>
              {text(`训练样本 ${index + 1}`, `Training sample ${index + 1}`)}
            </title>
          </path>
        ))}
        <text x="265" y="357" textAnchor="middle" fontSize="14">
          {axisLabel}
        </text>
        <text
          transform="translate(20 180) rotate(-90)"
          textAnchor="middle"
          fontSize="14"
        >
          {text("光谱数值", "Spectral value")}
        </text>
      </svg>
      <figcaption className="px-3 pb-2 text-xs text-slate-600">
        {text(
          `展示 ${chart.n_returned} 条训练光谱，每条显示 ${chart.x.length} 个位置。`,
          `Showing ${chart.n_returned} training spectra at ${chart.x.length} positions.`,
        )}
      </figcaption>
    </figure>
  );
}
