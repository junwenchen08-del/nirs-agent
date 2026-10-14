import { expect, test, type Page } from "@playwright/test";

import { libraryDatasets, previewOf } from "../fixtures/dataset-library";

import { mockLangGraphAPI } from "./utils/mock-api";

async function setup(page: Page, format: "mat" | "table" = "mat") {
  await mockLangGraphAPI(page).ready;
  const dataset = { ...libraryDatasets[0]!, name: "Melamine_Dataset.mat" };
  const layouts = [
    {
      id: "layout-a",
      group: "Dataset.R562",
      label: "全部光谱（X1 + X2）",
      sample_count: 3032,
      feature_count: 346,
      targets: [{ id: "target-a", label: "Dataset.R562.Y" }],
      axes: [{ id: "axis", label: "Dataset.wn1 + Dataset.wn2" }],
      default_target: "target-a",
      default_axis: "axis",
    },
    {
      id: "layout-b",
      group: "Dataset.R568",
      label: "全部光谱（X1 + X2）",
      sample_count: 733,
      feature_count: 346,
      targets: [{ id: "target-b", label: "Dataset.R568.Y" }],
      axes: [{ id: "axis", label: "Dataset.wn1 + Dataset.wn2" }],
      default_target: "target-b",
      default_axis: "axis",
    },
  ];
  const profiles: Record<string, unknown>[] = [];
  const saves: Record<string, unknown>[] = [];
  const attaches: Record<string, unknown>[] = [];
  let failSave = true;
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({ json: { datasets: [dataset] } }),
  );
  await page.route("**/api/nir/datasets/**", async (route) => {
    const tail = new URL(route.request().url()).pathname
      .split("/")
      .slice(5)
      .join("/");
    if (tail === "preview")
      return route.fulfill({
        json: {
          ...previewOf(dataset),
          status: profiles.length ? "ready" : "needs_mapping",
          message: "请选择数据组",
        },
      });
    if (tail === "uses") return route.fulfill({ json: { uses: [] } });
    if (tail === "parsing" && route.request().method() === "GET")
      return route.fulfill({
        json: {
          dataset_id: dataset.id,
          source_sha256: dataset.sha256,
          profile_id: profiles.at(-1)?.id ?? null,
          format,
          task_type: "inspection",
          current: null,
          suggested: null,
          layouts: format === "mat" ? layouts : [],
          columns: ["样本编号", "含量", "光谱一", "光谱二", "光谱三"].map(
            (name, index) => ({ name, index }),
          ),
        },
      });
    if (tail === "parsing/preview") {
      const body = route.request().postDataJSON();
      return route.fulfill({
        json: {
          ...previewOf(dataset),
          sample_count: body.selection.layout_id === "layout-b" ? 733 : 3032,
          feature_count: 346,
          samples: [
            { name: "样本 1", values: [0.5] },
            { name: "样本 2", values: [1.5] },
          ],
        },
      });
    }
    if (tail === "parsing" && route.request().method() === "PUT") {
      const body = route.request().postDataJSON();
      saves.push(body);
      if (failSave)
        return route.fulfill({
          status: 409,
          json: { detail: { message: "设置已更新，当前选择仍保留。" } },
        });
      const profile = {
        id: "dsp_selected",
        dataset_id: dataset.id,
        profile_version: 1,
        profile_status: "confirmed",
        task_type: body.task_type,
        schema_status: "confirmed_mapping",
        mapping: { library_parse: { selection: body.selection } },
      };
      profiles.push(profile);
      return route.fulfill({ json: profile });
    }
    if (tail === "attach") {
      attaches.push(route.request().postDataJSON());
      return route.fulfill({
        json: {
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/selected-v1.csv",
          attachment_size_bytes: 1024,
        },
      });
    }
    return route.fulfill({ json: { ...dataset, profiles } });
  });
  return {
    saves,
    attaches,
    allowSave: () => {
      failSave = false;
    },
  };
}

test.beforeEach(async ({ page, baseURL }) => {
  await page
    .context()
    .addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
});

test("MAT groups are selectable, previewed, saved and pinned when attached", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/datasets");
  await page.getByRole("button", { name: "解析设置", exact: true }).click();
  const pane = page.getByTestId("dataset-parsing-settings");
  await expect(
    pane.getByRole("button", { name: "保存解析设置", exact: true }),
  ).toBeDisabled();
  await pane.getByLabel("数据组", { exact: true }).selectOption("Dataset.R568");
  await expect(pane.getByTestId("parsing-preview")).toContainText("733");
  await pane
    .getByLabel("分析用途", { exact: true })
    .selectOption("calibration");
  await expect(
    pane.getByRole("button", { name: "保存解析设置", exact: true }),
  ).toBeEnabled();
  await expect(pane.getByTestId("parsing-preview")).toContainText("0.5");
  if (process.env.DATASET_PARSING_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_PARSING_QA_DIR}/parsing-zh.png`,
      animations: "disabled",
    });
  await pane.getByRole("button", { name: "保存解析设置", exact: true }).click();
  await expect(pane.getByRole("alert")).toContainText("当前选择仍保留");
  await expect(pane.getByLabel("数据组", { exact: true })).toHaveValue(
    "Dataset.R568",
  );
  state.allowSave();
  await pane.getByRole("button", { name: "保存解析设置", exact: true }).click();
  await expect(
    page.getByText("解析设置已保存。查看详情和挂载到对话会使用当前选择。", {
      exact: true,
    }),
  ).toBeVisible();
  expect(state.saves[1]?.selection).toEqual({
    layout_id: "layout-b",
    target_id: "target-b",
    axis_id: "axis",
  });
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await page.getByRole("button", { name: "挂载到对话", exact: true }).click();
  await expect(page.getByTestId("mounted-dataset")).toContainText(
    "Melamine_Dataset.mat",
  );
  expect(state.attaches[0]?.profile_id).toBe("dsp_selected");
});

test("CSV columns use Chinese controls and retain selection on failure", async ({
  page,
}) => {
  const state = await setup(page, "table");
  await page.goto("/workspace/nir/datasets");
  await page.getByRole("button", { name: "解析设置", exact: true }).click();
  const pane = page.getByTestId("dataset-parsing-settings");
  await pane.getByLabel("光谱起始列", { exact: true }).fill("3");
  await pane.getByLabel("光谱结束列", { exact: true }).fill("5");
  await pane.getByLabel("目标值所在列", { exact: true }).selectOption("1");
  await pane.getByLabel("样本名称所在列", { exact: true }).selectOption("0");
  await pane.getByLabel("首行是列名", { exact: true }).check();
  await expect(pane.getByTestId("parsing-preview")).toBeVisible();
  await expect(
    pane.getByRole("button", { name: "保存解析设置", exact: true }),
  ).toBeEnabled();
  await pane.getByRole("button", { name: "保存解析设置", exact: true }).click();
  await expect(pane.getByRole("alert")).toBeVisible();
  expect(state.saves[0]?.selection).toEqual({
    spectral_start: 2,
    spectral_end: 4,
    target_columns: [1],
    sample_column: 0,
    has_header: true,
  });
});
