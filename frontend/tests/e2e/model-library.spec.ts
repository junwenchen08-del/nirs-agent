import { expect, test, type Page } from "@playwright/test";

import { previewOf } from "../fixtures/dataset-library";

import { mockLangGraphAPI } from "./utils/mock-api";

const dataset = {
  id: "ds_corn",
  name: "玉米光谱数据",
  original_filename: "corn.csv",
  sha256: "a".repeat(64),
  status: "ready",
  size_bytes: 1024,
  media_type: "text/csv",
  created_at: null,
};
const metrics = [
  { scope: "calibration", R2: 0.99, RMSE: 0.12, MAE: 0.1, n: 80 },
  { scope: "validation", R2: 0.96, RMSE: 0.24, MAE: 0.2, n: 20 },
  { scope: "final_test", R2: 0.95, RMSE: 0.28, MAE: 0.22, n: 20 },
];
const model = {
  id: "model_corn",
  model_id: "corn_moisture_pls",
  version: "v2",
  status: "ready",
  method: "pls",
  preprocessing: { steps: [{ method: "snv", params: {} }] },
  validation_scope: "independent_holdout_not_external",
  artifact_size_bytes: 2048,
  artifact_sha256: "b".repeat(64),
  metrics_sha256: "c".repeat(64),
  training_data_sha256: "d".repeat(64),
  metrics_summary: { R2_val: 0.96, RMSEP: 0.28 },
  source_dataset_id: dataset.id,
  source_profile_id: "profile_training",
  source_thread_id: "source_thread",
  source_run_id: null,
  source_attempt: 1,
  created_at: "2026-10-14T10:00:00Z",
  updated_at: null,
  last_used_at: null,
  reused_existing: false,
  report_summary: {
    target: "水分",
    unit: "%",
    sample_count: 120,
    feature_count: 700,
    metrics,
  },
};
const x = [8, 10, 12, 14, 16, 18];
const prediction = {
  type: "prediction",
  scope: "holdout",
  x,
  y: [8.2, 9.8, 12.1, 14.3, 15.8, 18.1],
  n_total: 20,
  n_returned: 6,
  x_unit: "%",
  y_unit: "%",
};
const spectrum = {
  type: "spectra",
  scope: "calibration_before",
  x: [1000, 1100, 1200, 1300, 1400],
  series: [
    { id: "a", values: [1, 2.5, 2, 4, 2] },
    { id: "b", values: [1.5, 3, 2.5, 5, 3] },
  ],
  n_total: 80,
  n_returned: 2,
  x_label: "wavelength",
  x_unit: "nm",
};
const detail = {
  model,
  dataset: {
    id: dataset.id,
    name: dataset.name,
    source_sha256: dataset.sha256,
    profile_id: model.source_profile_id,
  },
  report: {
    schema_version: 1,
    target: "水分",
    unit: "%",
    method: "pls",
    sample_count: 120,
    feature_count: 700,
    model_feature_count: 700,
    parameters: { n_components: 8 },
    preprocessing: [
      { method: "snv", params: {} },
      { method: "sg_smooth", params: { window_length: 11, polyorder: 2 } },
    ],
    metrics,
    charts: [
      prediction,
      {
        ...prediction,
        scope: "tuning",
        y: [8.1, 10.2, 11.9, 14.1, 15.9, 17.8],
      },
      spectrum,
      {
        ...spectrum,
        scope: "calibration_after",
        series: spectrum.series.map((curve) => ({
          ...curve,
          values: curve.values.map((v) => v / 2),
        })),
      },
    ],
  },
};
const modelUrl = `/api/nir/models/${encodeURIComponent(model.model_id)}`;
const reportUrl = `${modelUrl}/versions/v2/report`;
const detailUrl = `/workspace/nir/models/${encodeURIComponent(model.model_id)}/v2`;

async function setup(page: Page) {
  await page.context().addCookies([
    {
      name: "locale",
      value: "zh-CN",
      url: process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000",
    },
  ]);
  await mockLangGraphAPI(page).ready;
  const state = { runs: 0, attachments: 0, usage: 0, profiles: [] as string[] };
  await page.route("**/api/nir/storage/usage", (route) => {
    state.usage++;
    return route.abort();
  });
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          model,
          { ...model, id: "old", model_id: "已归档模型", status: "archived" },
        ],
      },
    }),
  );
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({ json: { datasets: [dataset] } }),
  );
  await page.route(`**${modelUrl}?version=v2`, (route) =>
    route.fulfill({ json: model }),
  );
  await page.route(`**${reportUrl}`, (route) =>
    route.fulfill({ json: detail }),
  );
  await page.route("**/api/nir/datasets/ds_corn/preview?*", (route) => {
    state.profiles.push(
      new URL(route.request().url()).searchParams.get("profile_id") ?? "",
    );
    return route.fulfill({
      json: { ...previewOf(dataset), profile_id: model.source_profile_id },
    });
  });
  await page.route(`**${modelUrl}/versions/v2/attach`, (route) => {
    state.attachments++;
    expect(route.request().postDataJSON().thread_id).toBeTruthy();
    return route.fulfill({
      json: {
        status: "attached",
        model_id: model.model_id,
        version: model.version,
        model_path: `/mnt/user-data/outputs/models/${model.model_id}/v2/model.pkl`,
      },
    });
  });
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/runs/stream", (route) => {
    state.runs++;
    return route.fallback();
  });
  return state;
}

test("Chinese model cards open data, methods and scientific charts, then mount in a new chat", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/models");
  await expect(
    page.getByRole("heading", { name: "模型库", exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId("nir-model-card")).toHaveCount(1);
  await expect(page.getByText("个人空间", { exact: true })).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "归档", exact: true }),
  ).toHaveCount(0);
  await page.getByLabel("搜索模型").fill("水分");
  if (process.env.MODEL_LIBRARY_QA_SCREENSHOT)
    await page.screenshot({
      path: process.env.MODEL_LIBRARY_QA_SCREENSHOT.replace(
        /\.png$/,
        "-cards.png",
      ),
    });
  await page
    .getByRole("link", { name: "水分 · 偏最小二乘回归（PLS）", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "水分 · 偏最小二乘回归（PLS）",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.getByTestId("model-prediction-figure")).toHaveCount(1);
  await page.getByLabel("评价范围").selectOption("validation");
  await expect(page.getByTestId("model-prediction-figure")).toHaveCount(1);
  await page.getByRole("tab", { name: "处理方法", exact: true }).click();
  await expect(page.getByTestId("model-spectra-figure")).toHaveCount(2);
  await page.getByRole("tab", { name: "训练数据", exact: true }).click();
  await expect(page.getByTestId("dataset-preview")).toBeVisible();
  await page.getByRole("tab", { name: "处理方法", exact: true }).click();
  await expect(page.getByText("统一每条光谱的数值尺度（SNV）")).toBeVisible();
  expect(state.profiles).toEqual([model.source_profile_id]);
  expect(state.usage).toBe(0);
  if (process.env.MODEL_LIBRARY_QA_SCREENSHOT) {
    await page.screenshot({ path: process.env.MODEL_LIBRARY_QA_SCREENSHOT });
    await page.getByRole("tab", { name: "预测效果", exact: true }).click();
    await page
      .getByTestId("model-prediction-figure")
      .first()
      .scrollIntoViewIfNeeded();
    await page.screenshot({
      path: process.env.MODEL_LIBRARY_QA_SCREENSHOT.replace(
        /\.png$/,
        "-predictions.png",
      ),
    });
    await page.getByRole("tab", { name: "处理方法", exact: true }).click();
    await page
      .getByTestId("model-spectra-figure")
      .first()
      .scrollIntoViewIfNeeded();
    await page.screenshot({
      path: process.env.MODEL_LIBRARY_QA_SCREENSHOT.replace(
        /\.png$/,
        "-processing.png",
      ),
    });
  }
  await page.getByRole("link", { name: "在新对话中使用", exact: true }).click();
  await expect(page.getByTestId("mounted-model")).toContainText(model.model_id);
  expect(state.attachments).toBe(1);
  expect(state.runs).toBe(0);
  await expect(page.getByLabel("Target chat ID")).toHaveCount(0);
  await expect(page).not.toHaveURL(/[?&]model=/);
});

test("unavailable model report and dataset preview retry independently", async ({
  page,
}) => {
  await setup(page);
  let fail = true;
  await page.route(`**${reportUrl}`, (route) =>
    fail
      ? route.fulfill({ status: 503, json: {} })
      : route.fulfill({ json: detail }),
  );
  await page.goto(detailUrl);
  await expect(
    page.getByRole("alert").filter({ hasText: "模型详情暂时无法读取" }),
  ).toBeVisible();
  fail = false;
  await page.route("**/api/nir/datasets/ds_corn/preview?*", (route) =>
    route.fulfill({
      json: {
        ...previewOf(dataset),
        source_sha256: "wrong",
        profile_id: model.source_profile_id,
      },
    }),
  );
  await page.getByRole("button", { name: "重新加载", exact: true }).click();
  await page.getByRole("tab", { name: "训练数据", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "数据特征暂时无法读取" }),
  ).toBeVisible();
  await expect(page.getByTestId("model-prediction-figure")).toHaveCount(1);
  await page.route("**/api/nir/datasets/ds_corn/preview?*", (route) =>
    route.fulfill({
      json: { ...previewOf(dataset), profile_id: model.source_profile_id },
    }),
  );
  await page.getByRole("button", { name: "重试数据预览" }).click();
  await expect(page.getByTestId("dataset-preview")).toBeVisible();
});

test("missing old charts stay absent, and mounting failure can retry without sending", async ({
  page,
}) => {
  const state = await setup(page);
  await page.route(`**${reportUrl}`, (route) =>
    route.fulfill({
      json: {
        ...detail,
        dataset: null,
        report: { ...detail.report, charts: [], preprocessing: null },
      },
    }),
  );
  await page.goto(detailUrl);
  await expect(
    page.getByText("当时未保存逐样本的预测结果", { exact: false }),
  ).toBeVisible();
  await expect(page.getByTestId("model-prediction-figure")).toHaveCount(0);
  await page.getByRole("tab", { name: "训练数据", exact: true }).click();
  await expect(
    page.getByText("这个模型未记录可核实的数据集关联。", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByText("这个早期模型没有关联到数据集库。", { exact: false }),
  ).toHaveCount(0);
  let fail = true;
  await page.route(`**${modelUrl}/versions/v2/attach`, (route) =>
    fail ? route.fulfill({ status: 503, json: {} }) : route.fallback(),
  );
  await page.getByRole("link", { name: "在新对话中使用", exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "模型未能挂载" }),
  ).toBeVisible();
  expect(state.runs).toBe(0);
  fail = false;
  await page.getByRole("button", { name: "重新挂载模型" }).click();
  await expect(page.getByTestId("mounted-model")).toContainText(model.model_id);
  expect(state.runs).toBe(0);
});
