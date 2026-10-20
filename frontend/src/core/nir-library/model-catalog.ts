import { getModelReport, listModels } from "./api";
import type { NIRModelVersion } from "./types";

export type CatalogModel = NIRModelVersion & { recoveredDatasetName?: string };

// Only the existing verified report may recover historical lineage. Never match names.
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
      if (row.source_dataset_id) continue;
      try {
        const detail = await getModelReport(row.model_id, row.version, signal);
        if (
          detail.dataset &&
          detail.model.artifact_sha256 === row.artifact_sha256 &&
          detail.model.training_data_sha256 === row.training_data_sha256
        ) {
          rows[index] = {
            ...row,
            source_dataset_id: detail.dataset.id,
            source_profile_id: detail.dataset.profile_id,
            recoveredDatasetName: detail.dataset.name,
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
