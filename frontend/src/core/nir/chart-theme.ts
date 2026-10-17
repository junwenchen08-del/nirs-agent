/**
 * Single source for every NIR chart color.
 *
 * `chartSeries` is categorical: the same data keeps the same color across views
 * because colors are chosen by index, never per component. `chartColors` carries
 * scientific meaning (raw, processed, retained, stopped, diagnostic, reference)
 * and must not be reassigned inside a component. Inline hex literals in chart
 * components are a regression; import a token instead.
 */
export const chartSeries = [
  "#007eb8",
  "#ff7f0e",
  "#7c3aed",
  "#059669",
  "#dc2626",
  "#ca8a04",
  "#db2777",
  "#475569",
  "#0891b2",
  "#4f46e5",
  "#65a30d",
  "#9333ea",
] as const;

export function seriesColor(index: number): string {
  const length = chartSeries.length;
  const position = Number.isInteger(index) ? index % length : 0;
  return chartSeries[position < 0 ? position + length : position]!;
}

export const chartColors = {
  /** 原始／未处理光谱与参照虚线 */
  raw: "#64748b",
  /** 处理后光谱、实测值序列与框选高亮 */
  processed: "#2563eb",
  /** 保留区间、最佳方案与采用标记 */
  retained: "#059669",
  /** 停止、失败与预算终止 */
  stopped: "#dc2626",
  /** 诊断点与当前选中的显示点 */
  diagnostic: "#d97706",
  /** 实测＝预测的恒等线与基准参照线 */
  reference: "#ef2323",
} as const;

/** 白底报告图的坐标轴、刻度与图例描边墨色。 */
export const chartInk = {
  axis: "#111827",
  tick: "#334155",
  border: "#e2e8f0",
} as const;
