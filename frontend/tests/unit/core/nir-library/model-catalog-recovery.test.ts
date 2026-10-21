import { beforeEach, expect, rs, test } from "@rstest/core";

rs.mock("@/core/nir-library/api", () => ({
  listModels: rs.fn(),
  getModelReport: rs.fn(),
}));

import { getModelReport, listModels } from "@/core/nir-library/api";
import { loadModelCatalog } from "@/core/nir-library/model-catalog";
import { cardMetrics } from "@/core/nir-library/model-report";
import type { NIRModelDetail, NIRModelVersion } from "@/core/nir-library/types";

const legacy = {
  id: "legacy",
  model_id: "legacy_pls",
  version: "v1",
  status: "ready",
  method: "pls",
  artifact_sha256: "artifact",
  metrics_sha256: "metrics",
  training_data_sha256: "training",
  source_dataset_id: null,
  source_profile_id: null,
  created_at: null,
  metrics_summary: { R2_val: 0.96, RMSEP: 0.3494 },
} as unknown as NIRModelVersion;
const scoped = [
  { scope: "validation" as const, R2: 0.96, RMSE: 0.3, MAE: null, n: 62 },
  { scope: "final_test" as const, R2: 0.9282, RMSE: 0.3494, MAE: null, n: 62 },
];
function report(model = legacy, metrics = scoped): NIRModelDetail {
  return {
    model,
    dataset: {
      id: "ds_exact",
      profile_id: null,
      name: "原训练数据",
      source_sha256: "source",
    },
    report: {
      schema_version: 1,
      target: null,
      unit: null,
      sample_count: null,
      feature_count: null,
      metrics,
      charts: [],
    },
  } as unknown as NIRModelDetail;
}
beforeEach(() => {
  rs.mocked(listModels)
    .mockReset()
    .mockResolvedValue([structuredClone(legacy)]);
  rs.mocked(getModelReport).mockReset().mockResolvedValue(report());
});

test("legacy catalog recovers scoped scores from the same verified report used for lineage", async () => {
  const [model] = await loadModelCatalog();
  expect(
    cardMetrics(model!).find((item) => item.scope === "final_test"),
  ).toMatchObject({ R2: 0.9282, RMSE: 0.3494, n: 62 });
  expect(model!.source_dataset_id).toBe("ds_exact");
  expect(getModelReport).toHaveBeenCalledTimes(1);
  expect(legacy.report_summary).toBeUndefined();
});

test("known dataset linkage does not skip recovery of an incomplete metrics summary", async () => {
  const linked = { ...legacy, source_dataset_id: "ds_exact" };
  rs.mocked(listModels).mockResolvedValue([linked]);
  rs.mocked(getModelReport).mockResolvedValue({
    ...report(linked),
    dataset: null,
  });
  const [model] = await loadModelCatalog();
  expect(
    cardMetrics(model!).find((item) => item.scope === "final_test")?.R2,
  ).toBe(0.9282);
  expect(model!.source_dataset_id).toBe("ds_exact");
});

test("complete scoped card metrics do not cause another report read", async () => {
  rs.mocked(listModels).mockResolvedValue([
    {
      ...legacy,
      source_dataset_id: "ds_exact",
      report_summary: {
        target: null,
        unit: null,
        sample_count: null,
        feature_count: null,
        metrics: scoped,
      },
    },
  ]);
  await loadModelCatalog();
  expect(getModelReport).not.toHaveBeenCalled();
});

test("different artifact, training, metrics, model or version identity cannot supply catalog scores", async () => {
  for (const mismatch of [
    { artifact_sha256: "different" },
    { training_data_sha256: "different" },
    { metrics_sha256: "different" },
    { model_id: "different" },
    { version: "v2" },
  ]) {
    rs.mocked(getModelReport).mockResolvedValue(
      report({ ...legacy, ...mismatch }),
    );
    const [model] = await loadModelCatalog();
    expect(model!.source_dataset_id).toBeNull();
    expect(
      cardMetrics(model!).find((item) => item.scope === "final_test")?.R2,
    ).toBeNull();
  }
});

test("missing final R2 never borrows the validation score", async () => {
  const detail = report();
  detail.report.metrics = scoped.map((metric) =>
    metric.scope === "final_test" ? { ...metric, R2: null } : metric,
  );
  rs.mocked(getModelReport).mockResolvedValue(detail);
  const [model] = await loadModelCatalog();
  expect(
    cardMetrics(model!).find((item) => item.scope === "final_test"),
  ).toMatchObject({ R2: null, RMSE: 0.3494 });
});

test("unavailable report leaves the saved model and existing metrics usable", async () => {
  rs.mocked(getModelReport).mockRejectedValue(new Error("Unavailable"));
  const rows = await loadModelCatalog();
  expect(rows).toHaveLength(1);
  expect(cardMetrics(rows[0]!)).toEqual(cardMetrics(legacy));
});
