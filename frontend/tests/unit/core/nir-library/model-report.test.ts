import { expect, test } from "@rstest/core";

import {
  cardMetrics,
  chartMetric,
  modelChatPath,
  modelReportPath,
  parseModelDetail,
  validChart,
} from "@/core/nir-library/model-report";
import type { ModelChart, NIRModelVersion } from "@/core/nir-library/types";

const model = {
  id: "model_1",
  status: "ready",
  method: "pls",
  preprocessing: {},
  validation_scope: "independent_holdout_not_external",
  artifact_size_bytes: 100,
  artifact_sha256: "a",
  metrics_sha256: "b",
  training_data_sha256: "c",
  source_thread_id: "thread1",
  source_run_id: null,
  source_attempt: 1,
  created_at: null,
  updated_at: null,
  last_used_at: null,
  reused_existing: false,
  model_id: "玉米 / PLS",
  version: "版本 1",
  source_dataset_id: "ds1",
  source_profile_id: "profile1",
  metrics_summary: { R2_val: 0.91, RMSEP: 0.32 },
} satisfies NIRModelVersion;
const chart: ModelChart = {
  type: "prediction",
  scope: "holdout",
  x: [1, 2],
  y: [1.1, 1.9],
  n_total: 2,
  n_returned: 2,
};
const detail = {
  model,
  dataset: null,
  report: {
    schema_version: 1,
    charts: [chart],
    metrics: [{ scope: "final_test", R2: null, RMSE: 0.32, MAE: null, n: 2 }],
  },
};

test("historical summaries keep tuning R² separate from final prediction error", () => {
  const metrics = cardMetrics(model);
  expect(metrics[0]).toMatchObject({
    scope: "validation",
    R2: 0.91,
    RMSE: null,
  });
  expect(metrics[1]).toMatchObject({
    scope: "final_test",
    R2: null,
    RMSE: 0.32,
  });
  expect(chartMetric(chart, metrics)).toEqual(metrics[1]);
});

test("model routes preserve exact version and encode Chinese names and delimiters", () => {
  expect(modelChatPath(model)).toBe(
    `/workspace/chats/new?model=${encodeURIComponent(model.model_id)}&version=${encodeURIComponent(model.version)}`,
  );
  expect(modelReportPath(model)).toContain(
    `${encodeURIComponent(model.model_id)}/${encodeURIComponent(model.version)}`,
  );
});

test("report projection rejects wrong model or dataset lineage and omits invalid charts", () => {
  expect(() => parseModelDetail(detail, "wrong", model.version)).toThrow();
  expect(() =>
    parseModelDetail(
      { ...detail, dataset: { id: "ds2", profile_id: "profile1" } },
      model.model_id,
      model.version,
    ),
  ).toThrow();
  expect(() =>
    parseModelDetail(
      { ...detail, dataset: { id: "ds1", profile_id: "changed-profile" } },
      model.model_id,
      model.version,
    ),
  ).toThrow();
  const parsed = parseModelDetail(
    {
      ...detail,
      report: { ...detail.report, charts: [chart, { ...chart, y: [NaN, 2] }] },
    },
    model.model_id,
    model.version,
  );
  expect(parsed.report.charts).toEqual([chart]);
  expect(parsed.report.metrics[0]?.R2).toBeNull();
});

test("chart projection bounds arrays and rejects unsafe scopes and misleading alignment", () => {
  expect(validChart(chart)).toBe(true);
  expect(validChart({ ...chart, scope: "all" })).toBe(false);
  expect(validChart({ ...chart, n_total: 1 })).toBe(false);
  expect(validChart({ ...chart, y: [1] })).toBe(false);
  expect(
    validChart({ ...chart, x: Array.from({ length: 501 }, (_, i) => i) }),
  ).toBe(false);
  const spectrum: ModelChart = {
    type: "spectra",
    scope: "calibration_before",
    x: [1200, 1100, 1000],
    series: [{ id: "a", values: [1, 2, 3] }],
    n_returned: 1,
    n_total: 10,
  };
  expect(validChart(spectrum)).toBe(true);
  expect(validChart({ ...spectrum, x: [1000, 1100, 1050] })).toBe(false);
});

test("before-and-after charts require matching training samples and axes", () => {
  const before: ModelChart = {
    type: "spectra",
    scope: "calibration_before",
    x: [1000, 1100],
    series: [{ id: "sample-a", values: [1, 2] }],
    n_returned: 1,
    n_total: 5,
  };
  const after = {
    ...before,
    scope: "calibration_after",
    series: [{ id: "other-sample", values: [0.1, 0.2] }],
  };
  const parsed = parseModelDetail(
    { ...detail, report: { ...detail.report, charts: [chart, before, after] } },
    model.model_id,
    model.version,
  );
  expect(parsed.report.charts).toEqual([chart]);
});
