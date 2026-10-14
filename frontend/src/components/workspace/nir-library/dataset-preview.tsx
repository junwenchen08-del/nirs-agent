"use client";

import { useI18n } from "@/core/i18n/hooks";
import type { NIRDatasetPreview } from "@/core/nir-library/types";

export function DatasetPreview({ preview }: { preview: NIRDatasetPreview }) {
  const { locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const count = (value: number | null) =>
    value === null ? "—" : value.toLocaleString();
  return (
    <section className="space-y-5" data-testid="dataset-preview">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          [text("样本数量", "Samples"), count(preview.sample_count)],
          [
            text("光谱变量数量", "Spectral variables"),
            count(preview.feature_count),
          ],
          [
            text("预览样本数量", "Preview samples"),
            count(preview.preview_sample_count),
          ],
          [
            text("预览缺失值比例", "Missing preview values"),
            preview.missing_ratio === null
              ? "—"
              : `${(preview.missing_ratio * 100).toFixed(1)}%`,
          ],
        ].map(([label, value]) => (
          <div key={label} className="bg-muted/40 rounded-lg p-3">
            <p className="text-muted-foreground text-xs">{label}</p>
            <p className="mt-1 text-lg font-semibold">{value}</p>
          </div>
        ))}
      </div>
      {preview.status !== "ready" && (
        <p className="text-muted-foreground rounded-lg border p-4 text-sm">
          {preview.message}
        </p>
      )}
      {preview.spectrum.length > 0 && (
        <section className="space-y-2 rounded-xl border p-4">
          <h3 className="font-medium">
            {text("光谱整体形状", "Spectral overview")}
          </h3>
          <p className="text-muted-foreground text-xs">
            {text(
              "实线为平均光谱，浅色区域为预览样本的最小值至最大值。",
              "Line: mean spectrum. Shaded range: minimum to maximum across preview samples.",
            )}
          </p>
          <SpectrumChart points={preview.spectrum} zh={zh} />
          <p className="text-muted-foreground text-center text-xs">
            {preview.axis_kind === "axis"
              ? text(
                  "光谱位置（按文件中的坐标）",
                  "Spectral position (file coordinates)",
                )
              : text("波段编号", "Spectral variable index")}
          </p>
        </section>
      )}
      {preview.targets.map((target, i) => (
        <section
          key={`${target.name}:${i}`}
          className="space-y-3 rounded-xl border p-4"
        >
          <h3 className="font-medium">
            {target.kind === "classification"
              ? text("各类别样本分布", "Class distribution")
              : text("目标值分布", "Target distribution")}{" "}
            · {target.name}
          </h3>
          {target.bars.length ? (
            <div
              className="space-y-2"
              role="img"
              aria-label={text(
                `${target.name}分布图`,
                `${target.name} distribution`,
              )}
            >
              {target.bars.map((bar, index) => (
                <div
                  key={index}
                  className="grid grid-cols-[minmax(0,8rem)_1fr_3rem] items-center gap-2 text-xs"
                >
                  <span className="break-words">{bar.label}</span>
                  <div className="bg-muted h-5 overflow-hidden rounded">
                    <div
                      className="bg-primary/70 h-full rounded"
                      style={{
                        width: `${(100 * bar.count) / Math.max(1, ...target.bars.map((value) => value.count))}%`,
                      }}
                    />
                  </div>
                  <span className="text-right">{bar.count}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">
              {text(
                "该目标列没有可统计的值。",
                "No usable values in this target column.",
              )}
            </p>
          )}
        </section>
      ))}
      {preview.status === "ready" && preview.targets.length === 0 && (
        <p className="text-muted-foreground text-sm">
          {text(
            "尚未识别出明确的目标列，分析类型暂列为“待确认”。",
            "No definite target column was identified. The analysis type remains unconfirmed.",
          )}
        </p>
      )}
      {preview.fields.length > 0 && (
        <details className="rounded-lg border p-3 text-sm">
          <summary className="cursor-pointer font-medium">
            {text("文件中的字段", "File fields")}
          </summary>
          <p className="text-muted-foreground mt-2 break-words">
            {preview.fields.join(" · ")}
          </p>
        </details>
      )}
      {preview.status === "ready" && (
        <p className="text-muted-foreground text-xs">
          {text(
            `图表统计使用 ${preview.preview_sample_count} 个预览样本，可能只展示部分样本和波段。数据概览不会更改文件，也不会训练模型。`,
            `Charts summarize ${preview.preview_sample_count} preview samples and may show a subset of samples and spectral variables. Viewing does not modify files or train models.`,
          )}
        </p>
      )}
    </section>
  );
}

function SpectrumChart({
  points,
  zh,
}: {
  points: NIRDatasetPreview["spectrum"];
  zh: boolean;
}) {
  const finite = points.filter(
    (point) =>
      Number.isFinite(point.mean) &&
      point.mean !== null &&
      point.low !== null &&
      point.high !== null,
  );
  if (!finite.length)
    return (
      <p>{zh ? "光谱中没有可绘图的数值。" : "No usable spectrum values."}</p>
    );
  const xs = finite.map((point) => point.x);
  const ys = finite.flatMap((point) => [point.low!, point.high!]);
  const xmin = Math.min(...xs),
    xmax = Math.max(...xs);
  const ymin = Math.min(...ys),
    ymax = Math.max(...ys);
  const x = (value: number) => 62 + ((value - xmin) / (xmax - xmin || 1)) * 618;
  const y = (value: number) =>
    240 - ((value - ymin) / (ymax - ymin || 1)) * 210;
  const line = finite
    .map((point, i) => `${i ? "L" : "M"}${x(point.x)},${y(point.mean!)}`)
    .join(" ");
  const area = [
    ...finite.map((point) => `${x(point.x)},${y(point.high!)}`),
    ...finite
      .slice()
      .reverse()
      .map((point) => `${x(point.x)},${y(point.low!)}`),
  ].join(" ");
  return (
    <svg
      viewBox="0 0 720 280"
      className="h-auto w-full"
      role="img"
      aria-label={zh ? "平均光谱与变化范围图" : "Mean spectrum and range"}
    >
      {[0, 1, 2, 3].map((i) => {
        const value = ymin + ((ymax - ymin) * i) / 3;
        return (
          <g key={i}>
            <line
              x1="62"
              x2="680"
              y1={y(value)}
              y2={y(value)}
              stroke="currentColor"
              opacity="0.1"
            />
            <text
              x="54"
              y={y(value) + 4}
              textAnchor="end"
              fill="currentColor"
              fontSize="11"
            >
              {value.toPrecision(3)}
            </text>
          </g>
        );
      })}
      <polygon points={area} fill="var(--primary)" opacity="0.12" />
      <path d={line} fill="none" stroke="var(--primary)" strokeWidth="2" />
      {[xmin, (xmin + xmax) / 2, xmax].map((value, i) => (
        <text
          key={i}
          x={x(value)}
          y="265"
          textAnchor="middle"
          fill="currentColor"
          fontSize="11"
        >
          {Number(value.toPrecision(5))}
        </text>
      ))}
    </svg>
  );
}
