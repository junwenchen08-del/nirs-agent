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

async function setup(
  page: Page,
  qualityStatus?: "below_threshold_user_approved",
) {
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
          { ...model, quality_status: qualityStatus },
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
    route.fulfill({
      json: { ...detail, model: { ...model, quality_status: qualityStatus } },
    }),
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

async function expandGroups(page: Page) {
  await expect(page.getByTestId("model-dataset-group").first()).toBeVisible();
  for (const group of await page.getByTestId("model-dataset-group").all()) {
    const button = group.getByRole("button", { expanded: false });
    if (await button.count()) await button.click();
  }
}

test("user-approved below-threshold models remain visible and usable with truthful notice", async ({
  page,
}) => {
  const state = await setup(page, "below_threshold_user_approved");
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
  await expect(page.getByTestId("model-quality-notice")).toContainText(
    "未达质量门槛，已由用户确认保存",
  );
  await expect(
    page.getByRole("link", { name: "在新对话中使用", exact: true }),
  ).toBeVisible();
  await page.goto(detailUrl);
  await expect(page.getByTestId("model-quality-notice")).toContainText(
    "评价指标保持原样",
  );
  await expect(
    page.getByRole("link", { name: "在新对话中使用", exact: true }),
  ).toBeVisible();
  expect(state.runs).toBe(0);
  expect(state.attachments).toBe(0);
});

test("Chinese model cards open data, methods and scientific charts, then mount in a new chat", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
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

test("groups by verified dataset identity and filters targets without changing model versions", async ({
  page,
}) => {
  const state = await setup(page);
  const rows = [
    model,
    {
      ...model,
      id: "protein",
      model_id: "corn_protein",
      version: "v3",
      created_at: "2026-10-16T00:00:00Z",
      source_profile_id: "profile_second",
      report_summary: { ...model.report_summary, target: "蛋白质" },
    },
    {
      ...model,
      id: "same_name",
      model_id: "other_corn",
      source_dataset_id: "ds_other",
      created_at: "2026-10-15T00:00:00Z",
    },
    {
      ...model,
      id: "missing",
      model_id: "unknown_model",
      source_dataset_id: null,
      source_profile_id: null,
    },
    {
      ...model,
      id: "unavailable",
      model_id: "unnamed_data",
      source_dataset_id: "ds_unavailable",
    },
  ];
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({ json: { models: rows } }),
  );
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({
      json: { datasets: [dataset, { ...dataset, id: "ds_other" }] },
    }),
  );
  await page.route(
    "**/api/nir/models/unknown_model/versions/v2/report",
    (route) => route.fulfill({ status: 404, json: { detail: "No report" } }),
  );
  await page.goto("/workspace/nir/models");
  await expect(page.getByTestId("model-dataset-group")).toHaveCount(4);
  await expect(page.getByTestId("nir-model-card")).toHaveCount(0);
  await expect(
    page
      .getByTestId("model-dataset-group")
      .getByRole("button", { expanded: false }),
  ).toHaveCount(4);
  await expandGroups(page);
  await expect(page.getByTestId("nir-model-card")).toHaveCount(5);
  const corn = page.locator(
    '[data-testid="model-dataset-group"][data-dataset-id="ds_corn"]',
  );
  await expect(corn.getByTestId("nir-model-card")).toHaveCount(2);
  await expect(corn.getByTestId("nir-model-card").first()).toContainText(
    "corn_protein",
  );
  await expect(
    corn.getByRole("link", { name: "profile_second", exact: true }),
  ).toHaveAttribute(
    "href",
    "/workspace/nir/datasets/ds_corn?profile=profile_second",
  );
  await corn.getByLabel("组内预测目标").selectOption("蛋白质");
  await expect(corn.getByTestId("nir-model-card")).toHaveCount(1);
  await expect(
    corn.getByRole("link", { name: "在新对话中使用", exact: true }),
  ).toHaveAttribute(
    "href",
    "/workspace/chats/new?model=corn_protein&version=v3",
  );
  await corn.getByLabel("组内预测目标").selectOption("");
  await corn.getByRole("button", { expanded: true }).click();
  await expect(corn.getByTestId("nir-model-card")).toHaveCount(0);
  await page.getByRole("button", { name: "全部模型", exact: true }).click();
  await expect(page.getByTestId("nir-model-card")).toHaveCount(5);
  await page.getByRole("button", { name: "按数据集", exact: true }).click();
  await expect(corn.getByRole("button", { expanded: false })).toBeVisible();
  await page.getByLabel("预测目标", { exact: true }).selectOption("蛋白质");
  await expect(page.getByTestId("model-dataset-group")).toHaveCount(1);
  await page.getByLabel("预测目标", { exact: true }).selectOption("");
  await page.getByLabel("搜索模型").fill("unknown_model");
  await expect(page.getByTestId("model-dataset-group")).toContainText(
    "训练数据来源待确认",
  );
  await page.getByLabel("搜索模型").fill("unnamed_data");
  await expect(page.getByTestId("model-dataset-group")).toContainText(
    "数据集名称暂不可用",
  );
  await page.goto("/workspace/nir/models?dataset=ds_other");
  await expandGroups(page);
  await expect(page.getByTestId("nir-model-card")).toHaveCount(1);
  await expect(page.getByTestId("nir-model-card")).toContainText("other_corn");
  await page.getByRole("link", { name: "清除数据集筛选" }).click();
  await expect(page.getByTestId("model-dataset-group")).toHaveCount(4);
  expect(state.runs).toBe(0);
  expect(state.attachments).toBe(0);
});

test("historical verified report lineage also drives dataset model counts", async ({
  page,
}) => {
  const state = await setup(page);
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          { ...model, source_dataset_id: null, source_profile_id: null },
        ],
      },
    }),
  );
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
  await expect(page.getByTestId("model-dataset-group")).toHaveCount(1);
  await expect(page.getByTestId("model-dataset-group")).toContainText(
    dataset.name,
  );
  await expect(
    page.getByRole("link", { name: "profile_training", exact: true }),
  ).toHaveAttribute(
    "href",
    "/workspace/nir/datasets/ds_corn?profile=profile_training",
  );
  await page.goto("/workspace/nir/datasets");
  await page
    .getByRole("link", { name: "已保存模型 1 个", exact: true })
    .click();
  await expect(page).toHaveURL(/models\?dataset=ds_corn/);
  await expandGroups(page);
  await expect(page.getByTestId("nir-model-card")).toHaveCount(1);
  expect(state.runs).toBe(0);
  expect(state.attachments).toBe(0);
});

test("historical recovery rejects a different model artifact", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          { ...model, source_dataset_id: null, source_profile_id: null },
        ],
      },
    }),
  );
  await page.route(`**${reportUrl}`, (route) =>
    route.fulfill({
      json: { ...detail, model: { ...model, artifact_sha256: "e".repeat(64) } },
    }),
  );
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
  await expect(page.getByTestId("model-dataset-group")).toContainText(
    "训练数据来源待确认",
  );
  await expect(page.getByTestId("nir-model-card")).toHaveCount(1);
  await expect(
    page.getByRole("link", { name: "查看数据集", exact: true }),
  ).toHaveCount(0);
});

test("legacy model cards recover the same scoped R2 shown by their verified report", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          {
            ...model,
            source_dataset_id: null,
            source_profile_id: null,
            report_summary: null,
          },
        ],
      },
    }),
  );
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
  const card = page.getByTestId("nir-model-card");
  await expect(card).toHaveCount(1);
  await expect(card).toContainText("R² 0.95");
  await expect(card).toContainText("RMSE 0.28");
  await card.getByRole("link", { name: "查看详情", exact: true }).click();
  await expect(page.getByText("0.95", { exact: true })).toBeVisible();
});

test("known dataset lineage still recovers missing final R2 without borrowing tuning R2", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          {
            ...model,
            report_summary: {
              ...model.report_summary,
              metrics: metrics.map((metric) =>
                metric.scope === "final_test"
                  ? { ...metric, R2: null }
                  : metric,
              ),
            },
          },
        ],
      },
    }),
  );
  await page.goto("/workspace/nir/models");
  await expandGroups(page);
  const card = page.getByTestId("nir-model-card");
  await expect(card).toContainText("R² 0.95");
  await expect(card).not.toContainText("R² 0.96");
  await expect(card).toContainText("测试数据");
});
