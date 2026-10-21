import { getModelReport, listModels } from "./api";
import { cardMetrics, finiteNumber } from "./model-report";
import type { NIRModelVersion } from "./types";

export type CatalogModel = NIRModelVersion & { recoveredDatasetName?: string };

// Recover display-only lineage and scoped metrics from the exact verified report.
// Never match names, recompute scores or pair metrics from different partitions.
export async function loadModelCatalog(
  signal?: AbortSignal,
): Promise<CatalogModel[]> {
  const rows: CatalogModel[] = (await listModels(signal)).filter(
    (row) => row.status === "ready",
  );
  let next = 0;
  async function worker() {
    while (!signal?.aborted && next < rows.length) {
      const index = next++;
      const row = rows[index]!;
      const metrics = cardMetrics(row);
      const primary =
        metrics.find(
          (metric) =>
            metric.scope === "final_test" &&
            (finiteNumber(metric.R2) !== null ||
              finiteNumber(metric.RMSE) !== null),
        ) ?? metrics.find((metric) => metric.scope === "validation");
      const missingMetrics =
        modelTask(row) === "regression" &&
        (!primary ||
          finiteNumber(primary.R2) === null ||
          finiteNumber(primary.RMSE) === null);
      if (row.source_dataset_id && !missingMetrics) continue;
      try {
        const detail = await getModelReport(row.model_id, row.version, signal);
        if (
          detail.model.model_id === row.model_id &&
          detail.model.version === row.version &&
          detail.model.artifact_sha256 === row.artifact_sha256 &&
          detail.model.training_data_sha256 === row.training_data_sha256 &&
          detail.model.metrics_sha256 === row.metrics_sha256
        ) {
          rows[index] = {
            ...row,
            ...(!row.source_dataset_id && detail.dataset
              ? {
                  source_dataset_id: detail.dataset.id,
                  source_profile_id: detail.dataset.profile_id,
                  recoveredDatasetName: detail.dataset.name,
                }
              : {}),
            ...(detail.report.metrics.some(
              (metric) =>
                ["validation", "final_test"].includes(metric.scope) &&
                (finiteNumber(metric.R2) !== null ||
                  finiteNumber(metric.RMSE) !== null),
            )
              ? {
                  report_summary: {
                    target: detail.report.target,
                    unit: detail.report.unit,
                    sample_count: detail.report.sample_count,
                    feature_count: detail.report.feature_count,
                    metrics: detail.report.metrics,
                  },
                }
              : {}),
          };
        }
      } catch {
        // Missing or unavailable historical evidence must not hide a usable model.
      }
    }
  }
  await Promise.all([worker(), worker(), worker()]);
  signal?.throwIfAborted();
  return rows.sort(
    (a, b) => savedAt(b) - savedAt(a) || a.id.localeCompare(b.id),
  );
}

export function savedAt(model: NIRModelVersion) {
  const time = Date.parse(model.created_at ?? "");
  return Number.isFinite(time) ? time : 0;
}

export function modelTask(
  model: NIRModelVersion,
): "regression" | "classification" | "unknown" {
  if (
    [
      "pls_da",
      "logistic",
      "linear_svm",
      "rbf_svm",
      "logistic_regression",
      "calibrated_svm",
      "svc",
    ].includes(model.method ?? "")
  )
    return "classification";
  if (
    [
      "pls",
      "ridge",
      "svr",
      "et",
      "extra_trees",
      "pcr",
      "rf",
      "random_forest",
      "cnn",
      "cnn_1d",
    ].includes(model.method ?? "")
  )
    return "regression";
  return "unknown";
}

export function modelLibraryPath(datasetId: string) {
  return `/workspace/nir/models?dataset=${encodeURIComponent(datasetId)}`;
}

export function groupModels(rows: CatalogModel[]) {
  const groups = new Map<string | null, CatalogModel[]>();
  for (const row of rows) {
    const id = row.source_dataset_id ?? null;
    groups.set(id, [...(groups.get(id) ?? []), row]);
  }
  return Array.from(groups, ([datasetId, models]) => ({
    datasetId,
    models: [...models].sort(
      (a, b) => savedAt(b) - savedAt(a) || a.id.localeCompare(b.id),
    ),
  })).sort((a, b) =>
    a.datasetId === null
      ? 1
      : b.datasetId === null
        ? -1
        : savedAt(b.models[0]!) - savedAt(a.models[0]!),
  );
}
