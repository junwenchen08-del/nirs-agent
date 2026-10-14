import type { MethodCard } from "./method-api";

export type MethodGroup = "preprocessing" | "modeling" | "other";
export type MethodFilter = {
  group: MethodGroup | "all";
  category: string;
  query: string;
};

const preprocessingCategories = new Set([
  "smooth",
  "scatter",
  "baseline",
  "derivative",
  "scale",
  "projection",
]);

const nativeCategories: Record<string, string> = {
  sg_smooth: "smooth",
  whittaker_smooth: "smooth",
  median_filter: "smooth",
  despike: "smooth",
  isolated_despike: "smooth",
  snv: "scatter",
  robust_snv: "scatter",
  rnv: "scatter",
  msc: "scatter",
  emsc: "scatter",
  derivative1: "derivative",
  derivative2: "derivative",
  norris_williams1: "derivative",
  norris_williams2: "derivative",
  norris_derivative1: "derivative",
  norris_derivative2: "derivative",
  detrend: "baseline",
  baseline_asls: "baseline",
  asls: "baseline",
  airpls: "baseline",
  arpls: "baseline",
  rubberband: "baseline",
  normalize: "scale",
  max_norm: "scale",
  standard_scale: "scale",
  minmax_scale: "scale",
  mean_center: "scale",
  autoscale: "scale",
};

export const methodCategoryLabels: Record<string, [string, string]> = {
  smooth: ["平滑降噪", "Smoothing"],
  scatter: ["散射校正", "Scatter correction"],
  baseline: ["基线校正", "Baseline correction"],
  derivative: ["导数", "Derivatives"],
  scale: ["尺度变换", "Scaling"],
  projection: ["正交投影", "Projection"],
  regression: ["回归建模", "Regression"],
  augmentation: ["数据增强", "Augmentation"],
  feature_selection: ["变量选择", "Feature selection"],
  outliers: ["异常诊断", "Outlier diagnostics"],
  plotting: ["绘图与辅助工具", "Plotting"],
  inspector: ["模型检查", "Model inspection"],
  physics: ["物理量转换", "Physical transformations"],
  adaptation: ["迁移与轴对齐", "Transfer and axis alignment"],
  datasets: ["示例数据", "Example datasets"],
  native: ["其他项目方法", "Other project methods"],
  uncategorized: ["未分类", "Uncategorized"],
};

type Classification = Pick<
  MethodCard,
  "method_id" | "method_kind" | "category"
>;

export function methodCategory(card: Classification): string {
  if (card.category && card.category !== "native") return card.category;
  if (
    card.method_kind === "preprocessing" &&
    nativeCategories[card.method_id]
  ) {
    return nativeCategories[card.method_id]!;
  }
  if (card.method_kind === "modeling") return "regression";
  return card.category?.length ? card.category : "uncategorized";
}

export function methodGroup(card: Classification): MethodGroup {
  const category = methodCategory(card);
  if (
    card.method_kind === "preprocessing" ||
    preprocessingCategories.has(category)
  ) {
    return "preprocessing";
  }
  if (card.method_kind === "modeling" || category === "regression")
    return "modeling";
  return "other";
}

export function categoryLabel(
  category: string,
  zh: boolean,
  labels?: Record<string, string>,
): string {
  const known = methodCategoryLabels[category];
  return zh
    ? (labels?.[category] ?? known?.[0] ?? category)
    : (known?.[1] ?? category.replaceAll("_", " "));
}

export function filterMethodCards(
  cards: MethodCard[],
  filter: MethodFilter,
  labels?: Record<string, string>,
): MethodCard[] {
  const query = filter.query.trim().toLowerCase();
  return cards.filter((card) => {
    const category = methodCategory(card);
    return (
      (filter.group === "all" || methodGroup(card) === filter.group) &&
      (filter.category === "all" || category === filter.category) &&
      [
        card.title,
        card.method_id,
        card.mcp_capability_id,
        card.summary_zh,
        card.applicability_zh,
        card.limitations_zh,
        card.parameter_guidance_zh,
        card.keywords,
        categoryLabel(category, true, labels),
        categoryLabel(category, false, labels),
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  });
}
