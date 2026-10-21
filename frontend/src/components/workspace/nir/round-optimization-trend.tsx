"use client";

import { useEffect, useRef, useState } from "react";

import { record, type AutonomousSummary } from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";

export function RoundOptimizationTrend({
  summary,
  selectedId,
  zh,
  onSelect,
}: {
  summary: AutonomousSummary;
  selectedId: string;
  zh: boolean;
  onSelect: (decisionId: string) => void;
}) {
  const [metric, setMetric] = useState("val:RMSE");
  const chart = useRef<SVGSVGElement>(null);
  const [width, setWidth] = useState(1000);
  const [scope, key] = metric.split(":");
  const points = summary.trends.flatMap((row) => {
    const metrics = record(row.metrics[scope!]);
    const value = metrics[key!];
    return metrics.scope ===
      (scope === "val" ? "validation" : "calibration_cv") &&
      typeof value === "number" &&
      Number.isFinite(value)
      ? [{ ...row, value }]
      : [];
  });
  const available = points.length > 0;
  useEffect(() => {
    const element = chart.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(240, entry.contentRect.width));
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [available]);
  const min = Math.min(...points.map((p) => p.value));
  const max = Math.max(...points.map((p) => p.value));
  const x = (i: number) =>
    points.length === 1
      ? width / 2
      : 65 + (i * (width - 130)) / (points.length - 1);
  const y = (value: number) => 85 - ((value - min) / (max - min || 1)) * 52;
  const label =
    metric === "val:RMSE"
      ? "Val RMSE"
      : metric === "val:R2"
        ? "Val R²"
        : "Cal CV RMSE";
  return (
    <section className="min-w-0 px-4 pb-3" data-testid="optimization-trend">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">
          {zh ? "逐轮优化效果" : "Optimization across rounds"}
        </h3>
        <select
          aria-label={zh ? "优化趋势指标" : "Optimization metric"}
          value={metric}
          onChange={(event) => setMetric(event.target.value)}
          className="bg-background rounded border px-2 py-1 text-xs"
        >
          <option value="val:RMSE">Val RMSE ↓</option>
          <option value="val:R2">Val R² ↑</option>
          <option value="cal_cv:RMSE">Cal CV RMSE ↓</option>
        </select>
      </div>
      {points.length ? (
        <svg
          ref={chart}
          viewBox={`0 0 ${width} 122`}
          className="mt-2 h-[122px] w-full min-w-0"
          role="group"
          aria-label={label}
        >
          <line
            x1="45"
            x2={width - 45}
            y1="90"
            y2="90"
            stroke="currentColor"
            opacity=".15"
          />
          {summary.comparison_group_id && (
            <polyline
              points={points.map((p, i) => `${x(i)},${y(p.value)}`).join(" ")}
              fill="none"
              stroke="var(--primary)"
              strokeWidth="2"
            />
          )}
          {points.map((p, i) => (
            <g
              key={p.experiment_id}
              role="button"
              tabIndex={0}
              aria-label={`${zh ? "查看第" : "View round"} ${p.round_index} ${zh ? "轮" : ""} · ${label} ${fmt(p.value)}`}
              aria-pressed={selectedId === p.decision_id}
              className="focus:stroke-primary cursor-pointer outline-none"
              onClick={() => onSelect(p.decision_id)}
              onKeyDown={(e) => {
                if (["Enter", " "].includes(e.key)) {
                  e.preventDefault();
                  onSelect(p.decision_id);
                }
              }}
            >
              <rect
                x={x(i) - Math.min(32, 400 / points.length)}
                y="0"
                width={Math.min(64, 800 / points.length)}
                height="122"
                fill="transparent"
                pointerEvents="all"
              />
              <circle
                cx={x(i)}
                cy={y(p.value)}
                r={selectedId === p.decision_id ? 7 : 5}
                fill={
                  selectedId === p.decision_id
                    ? "var(--primary)"
                    : "var(--card)"
                }
                stroke="var(--primary)"
                strokeWidth="2"
              />
              {(points.length <= 12 || selectedId === p.decision_id) && (
                <>
                  <text
                    x={x(i)}
                    y={y(p.value) - 15}
                    textAnchor="middle"
                    fill="currentColor"
                    fontSize="12"
                  >
                    {fmt(p.value)}
                  </text>
                  <text
                    x={x(i)}
                    y="115"
                    textAnchor="middle"
                    fill="currentColor"
                    fontSize="12"
                  >
                    {zh
                      ? `第 ${p.round_index ?? "—"} 轮`
                      : `Round ${p.round_index ?? "—"}`}
                    {p.experiment_id === summary.retained_experiment_id
                      ? zh
                        ? " · 保留"
                        : " · Retained"
                      : ""}
                  </text>
                </>
              )}
            </g>
          ))}
        </svg>
      ) : (
        <p className="text-muted-foreground my-4 text-sm">
          {zh
            ? "暂无成功实验的已记录指标。"
            : "No recorded successful experiment metrics."}
        </p>
      )}
      <p className="text-muted-foreground mt-1 text-[11px]">
        {zh
          ? "点击数据点定位轮次；仅展示 Cal CV / Val。最低误差与最终保留分别标识。"
          : "Select a point to locate its round. Cal CV / Val only; best score and final retention are separate."}
      </p>
    </section>
  );
}
