import type {
  ModelChart,
  ModelMetric,
  NIRModelDetail,
  NIRModelVersion,
} from "./types";

export function modelReportPath(
  model: Pick<NIRModelVersion, "model_id" | "version">,
) {
  return `/workspace/nir/models/${encodeURIComponent(model.model_id)}/${encodeURIComponent(model.version)}`;
}

export function modelChatPath(
  model: Pick<NIRModelVersion, "model_id" | "version">,
) {
  return `/workspace/chats/new?model=${encodeURIComponent(model.model_id)}&version=${encodeURIComponent(model.version)}`;
}

export function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function cardMetrics(model: NIRModelVersion): ModelMetric[] {
  if (Array.isArray(model.report_summary?.metrics))
    return model.report_summary.metrics;
  const summary = model.metrics_summary ?? {};
  return [
    {
      scope: "validation" as const,
      R2: finiteNumber(summary.R2_val),
      RMSE: null,
      MAE: null,
      n: null,
    },
    {
      scope: "final_test" as const,
      R2: finiteNumber(summary.R2_test),
      RMSE: finiteNumber(summary.RMSEP),
      MAE: null,
      n: null,
    },
  ].filter((metric) => metric.R2 !== null || metric.RMSE !== null);
}

export function scopeLabel(scope: string, zh: boolean): string {
  const labels: Record<string, [string, string]> = {
    calibration: ["训练数据", "Training data"],
    calibration_before: ["处理前", "Before processing"],
    calibration_after: ["处理后", "After processing"],
    validation: ["调参数据", "Tuning data"],
    tuning: ["调参数据", "Tuning data"],
    final_test: ["测试数据", "Test data"],
    holdout: ["测试数据", "Test data"],
  };
  return (
    labels[scope]?.[zh ? 0 : 1] ?? (zh ? "未记录分组" : "Unrecorded group")
  );
}

export function chartMetric(chart: ModelChart, metrics: ModelMetric[]) {
  const scopes: Record<string, string> = {
    tuning: "validation",
    holdout: "final_test",
  };
  const scope = scopes[chart.scope] ?? chart.scope;
  return metrics.find((metric) => metric.scope === scope);
}

export function validChart(chart: ModelChart): boolean {
  if (
    !chart ||
    !Array.isArray(chart.x) ||
    !chart.x.length ||
    chart.x.length > 500 ||
    chart.x.some((v) => finiteNumber(v) === null)
  )
    return false;
  if (
    !Number.isInteger(chart.n_total) ||
    chart.n_total < chart.n_returned ||
    !Number.isInteger(chart.n_returned) ||
    chart.n_returned < 1
  )
    return false;
  if (chart.type === "prediction")
    return (
      ["calibration", "validation", "tuning", "holdout", "final_test"].includes(
        chart.scope,
      ) &&
      chart.n_returned === chart.x.length &&
      Array.isArray(chart.y) &&
      chart.y.length === chart.x.length &&
      chart.y.every((v) => finiteNumber(v) !== null)
    );
  if (
    chart.type !== "spectra" ||
    !["calibration_before", "calibration_after"].includes(chart.scope) ||
    chart.x.length > 256 ||
    !chart.series?.length ||
    chart.series.length > 12 ||
    chart.series.length !== chart.n_returned
  )
    return false;
  const ascending = chart.x.slice(1).every((v, i) => v > chart.x[i]!);
  const descending = chart.x.slice(1).every((v, i) => v < chart.x[i]!);
  return (
    (ascending || descending) &&
    chart.series.every(
      (curve) =>
        curve.values.length === chart.x.length &&
        curve.values.every((v) => finiteNumber(v) !== null),
    )
  );
}

export function parseModelDetail(
  payload: unknown,
  modelId: string,
  version: string,
): NIRModelDetail {
  const detail = payload as NIRModelDetail;
  if (
    !detail ||
    detail.model?.model_id !== modelId ||
    detail.model.version !== version ||
    detail.report?.schema_version !== 1 ||
    !Array.isArray(detail.report.metrics) ||
    !Array.isArray(detail.report.charts)
  )
    throw new Error("模型详情与所选模型不一致，请重新加载。");
  if (
    detail.dataset &&
    (detail.dataset.id !== detail.model.source_dataset_id ||
      detail.dataset.profile_id !== detail.model.source_profile_id)
  )
    throw new Error("训练数据记录与模型不一致，请重新加载。");
  let charts = detail.report.charts.slice(0, 6).filter(validChart);
  const before = charts.find((chart) => chart.scope === "calibration_before");
  const after = charts.find((chart) => chart.scope === "calibration_after");
  if (
    before &&
    after &&
    (JSON.stringify(before.x) !== JSON.stringify(after.x) ||
      JSON.stringify(before.series?.map((curve) => curve.id)) !==
        JSON.stringify(after.series?.map((curve) => curve.id)))
  )
    charts = charts.filter((chart) => chart.type !== "spectra");
  return {
    ...detail,
    report: {
      ...detail.report,
      charts,
      metrics: detail.report.metrics
        .filter((metric) =>
          ["calibration", "validation", "final_test"].includes(metric.scope),
        )
        .map((metric) => ({
          ...metric,
          R2: finiteNumber(metric.R2),
          RMSE: finiteNumber(metric.RMSE),
          MAE: finiteNumber(metric.MAE),
          n: finiteNumber(metric.n),
        })),
    },
  };
}

const SIMPLE_METHODS: Record<string, [string, string]> = {
  snv: ["统一每条光谱的数值尺度（SNV）", "Normalize each spectrum (SNV)"],
  msc: [
    "校正光谱的整体偏移与尺度（MSC）",
    "Correct spectral offset and scale (MSC)",
  ],
  sg: ["平滑光谱，减少噪声", "Smooth spectra to reduce noise"],
  sg_smooth: ["平滑光谱，减少噪声", "Smooth spectra to reduce noise"],
  derivative1: [
    "突出光谱变化（一阶导数）",
    "Highlight spectral changes (first derivative)",
  ],
  derivative2: [
    "突出局部光谱变化（二阶导数）",
    "Highlight local changes (second derivative)",
  ],
  mean_center: ["调整到共同的中心位置", "Center spectra"],
  normalize: ["统一光谱尺度", "Normalize spectral scale"],
  raw: ["直接使用原始光谱", "Use original spectra"],
};

export function simpleStep(method: string, zh: boolean): string | null {
  return SIMPLE_METHODS[method]?.[zh ? 0 : 1] ?? null;
}
