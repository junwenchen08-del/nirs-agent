import { expect, test, type Page } from "@playwright/test";

import type {
  DatasetWorkspace,
  DatasetWorkspaceSpectra,
} from "@/core/nir-library/types";

import { libraryDatasets, previewOf } from "../fixtures/dataset-library";

import { mockLangGraphAPI } from "./utils/mock-api";

async function setup(page: Page) {
  await mockLangGraphAPI(page).ready;
  const dataset = libraryDatasets[0]!;
  let workspace: DatasetWorkspace = {
    dataset_id: dataset.id,
    source_sha256: dataset.sha256!,
    profile_id: null,
    version: 0,
    saved: false,
    sample_count: 80,
    feature_count: 6,
    axis_kind: "axis",
    axis: [1000, 1100, 1200, 1300, 1400, 1500],
    fields: ["水分"],
    usages: Array.from({ length: 80 }, (_, i) => (i < 50 ? 1 : i < 65 ? 2 : 3)),
    range: [1000, 1500],
    selected_feature_count: 6,
  };
  const saves: Record<string, unknown>[] = [];
  const attaches: Record<string, unknown>[] = [];
  let failSave = false;
  let failIdentity = false;
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({ json: { datasets: [dataset], count: 1 } }),
  );
  await page.route("**/api/nir/datasets/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const identity = {
      dataset_id: dataset.id,
      source_sha256: failIdentity ? "wrong" : dataset.sha256,
      profile_id: workspace.profile_id,
    };
    if (path.endsWith("/preview"))
      return route.fulfill({ json: previewOf(dataset) });
    if (path.endsWith("/rows")) {
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const query = url.searchParams.get("query") ?? "";
      const rows = Array.from({ length: 80 }, (_, i) => ({
        index: i,
        name: `玉米-${String(i + 1).padStart(3, "0")}`,
        values: [10 + i / 10],
      })).filter((r) => r.name.includes(query));
      return route.fulfill({
        json: {
          ...identity,
          total: rows.length,
          offset,
          rows: rows.slice(offset, offset + 50),
        },
      });
    }
    if (path.endsWith("/spectra")) {
      const ids = url.searchParams.has("indices")
        ? url.searchParams
            .get("indices")!
            .split(",")
            .filter(Boolean)
            .map(Number)
        : Array.from({ length: 24 }, (_, i) => i * 3);
      const data: DatasetWorkspaceSpectra = {
        ...identity,
        source_sha256: identity.source_sha256!,
        axis: workspace.axis,
        total: 80,
        point_count: 6,
        curves: ids.map((index) => ({
          index,
          name: `玉米-${index + 1}`,
          values: workspace.axis.map((_, i) => 1 + Math.sin(i) + index / 20),
        })),
      };
      return route.fulfill({ json: data });
    }
    if (path.endsWith("/workspace")) {
      if (route.request().method() === "PUT") {
        const body = route.request().postDataJSON();
        saves.push(body);
        if (failSave)
          return route.fulfill({
            status: 409,
            json: {
              detail: { message: "设置已更新，请重新载入。当前修改仍保留。" },
            },
          });
        workspace = {
          ...workspace,
          saved: true,
          profile_id: `dsp_${workspace.version + 1}`,
          version: workspace.version + 1,
          usages: body.usages,
          range: body.range,
          selected_feature_count: workspace.axis.filter(
            (x) => x >= body.range[0] && x <= body.range[1],
          ).length,
        };
      }
      return route.fulfill({ json: workspace });
    }
    if (path.endsWith("/attach")) {
      attaches.push(route.request().postDataJSON());
      return route.fulfill({
        json: {
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/workspace-v1.csv",
          attachment_size_bytes: 1024,
          workspace_version: workspace.version,
        },
      });
    }
    return route.fulfill({
      json: {
        ...dataset,
        profiles: workspace.profile_id
          ? [
              {
                id: workspace.profile_id,
                profile_version: workspace.version,
                profile_status: "confirmed",
                mapping: { workspace: {} },
              },
            ]
          : [],
      },
    });
  });
  return {
    saves,
    attaches,
    failSave: (value: boolean) => {
      failSave = value;
    },
    failIdentity: (value: boolean) => {
      failIdentity = value;
    },
  };
}

test.beforeEach(async ({ page, baseURL }) => {
  await page
    .context()
    .addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
});

test("card and details open a Chinese sample table with linked spectra", async ({
  page,
}) => {
  await setup(page);
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("/workspace/nir/datasets");
  await page.getByRole("button", { name: "打开工作台", exact: true }).click();
  await expect(page).toHaveURL(/datasets\/ds_corn$/);
  await expect(page.getByRole("table", { name: "样本信息" })).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "样本用途" }),
  ).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "水分" })).toBeVisible();
  await page.getByRole("button", { name: "玉米-003", exact: true }).click();
  await expect(page.getByRole("img", { name: "选中样本的光谱" })).toBeVisible();
  await expect(page.getByRole("img", { name: "数据集光谱概览" })).toBeVisible();
  await expect(
    page.getByRole("table").locator("input[type=number]"),
  ).toHaveCount(0);
  if (process.env.DATASET_WORKSPACE_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_WORKSPACE_QA_DIR}/workspace-zh.png`,
      fullPage: true,
      animations: "disabled",
    });
});

test("individual roles and spectral range save and survive reload and attachment", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/datasets/ds_corn");
  await page
    .getByRole("combobox", { name: "样本用途 1", exact: true })
    .selectOption("2");
  await page.getByRole("spinbutton", { name: "波段起始值" }).fill("1100");
  await page.getByRole("spinbutton", { name: "波段结束值" }).fill("1400");
  await expect(
    page.getByRole("button", { name: "在新对话中使用", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "保存修改", exact: true }).click();
  await expect(
    page.getByText("已保存 · 版本 1", { exact: true }),
  ).toBeVisible();
  expect(state.saves).toHaveLength(1);
  expect((state.saves[0]!.usages as number[])[0]).toBe(2);
  expect(state.saves[0]!.range).toEqual([1100, 1400]);
  expect(Object.keys(state.saves[0]!).sort()).toEqual([
    "profile_id",
    "range",
    "source_sha256",
    "usages",
  ]);
  await page.reload();
  await expect(
    page.getByRole("combobox", { name: "样本用途 1", exact: true }),
  ).toHaveValue("2");
  await expect(
    page.getByRole("spinbutton", { name: "波段起始值" }),
  ).toHaveValue("1100");
  await page
    .getByRole("button", { name: "在新对话中使用", exact: true })
    .click();
  await expect(page.getByTestId("mounted-dataset")).toContainText("玉米水分");
  expect(state.attaches[0]!.profile_id).toBe("dsp_1");
});

test("bulk roles apply beyond the current page and sample search stays aligned", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/datasets/ds_corn");
  await page.getByRole("button", { name: "选择全部样本", exact: true }).click();
  await page.getByRole("combobox", { name: "批量样本用途" }).selectOption("4");
  await page.getByRole("button", { name: "应用到选中样本" }).click();
  await expect(page.getByText("不参与 80", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "下一页" }).click();
  await expect(
    page.getByRole("combobox", { name: "样本用途 51", exact: true }),
  ).toHaveValue("4");
  await page.getByRole("textbox", { name: "搜索样本名称" }).fill("080");
  await expect(
    page.getByRole("button", { name: "玉米-080", exact: true }),
  ).toBeVisible();
  await expect(page.getByRole("table").locator("tbody tr")).toHaveCount(1);
  await page.getByRole("button", { name: "保存修改" }).click();
  await expect.poll(() => state.saves.length).toBe(1);
  expect(state.saves[0]!.usages).toEqual(Array(80).fill(4));
});

test("dragging a plot selects a range and invalid ranges cannot be saved", async ({
  page,
}) => {
  await setup(page);
  await page.goto("/workspace/nir/datasets/ds_corn");
  const plot = page.getByRole("img", { name: "数据集光谱概览" });
  await expect(plot).toBeVisible();
  await plot.scrollIntoViewIfNeeded();
  const box = (await plot.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.5, {
    steps: 8,
  });
  await page.mouse.up();
  await expect(
    page.getByRole("spinbutton", { name: "波段起始值" }),
  ).not.toHaveValue("1000");
  await expect(page.getByRole("button", { name: "保存修改" })).toBeEnabled();
  await page.getByRole("spinbutton", { name: "波段起始值" }).fill("2000");
  await expect(page.getByRole("button", { name: "保存修改" })).toBeDisabled();
  await expect(
    page.getByRole("alert").filter({ hasText: "有效范围" }),
  ).toBeVisible();
});

test("a save conflict preserves the draft and discarding requires confirmation", async ({
  page,
}) => {
  const state = await setup(page);
  state.failSave(true);
  await page.goto("/workspace/nir/datasets/ds_corn");
  await page
    .getByRole("combobox", { name: "样本用途 1", exact: true })
    .selectOption("2");
  await page.getByRole("button", { name: "保存修改" }).click();
  await expect(
    page.getByText("设置已更新，请重新载入。当前修改仍保留。"),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "样本用途 1", exact: true }),
  ).toHaveValue("2");
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.getByRole("button", { name: "返回数据集库" }).click();
  await expect(page).toHaveURL(/ds_corn$/);
  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "重新载入" }).click();
  await expect(
    page.getByRole("combobox", { name: "样本用途 1", exact: true }),
  ).toHaveValue("1");
});

test("different source identity does not display spectra or sample values", async ({
  page,
}) => {
  const state = await setup(page);
  state.failIdentity(true);
  await page.goto("/workspace/nir/datasets/ds_corn");
  await expect(
    page
      .getByText("数据版本已变化，请重新载入。当前未保存修改仍保留。", {
        exact: true,
      })
      .first(),
  ).toBeVisible();
  await expect(page.getByRole("table")).toHaveCount(0);
  await expect(page.getByRole("img", { name: "数据集光谱概览" })).toHaveCount(
    0,
  );
});

test("mobile table scrolling is contained and charts fit the viewport", async ({
  page,
}) => {
  await setup(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/workspace/nir/datasets/ds_corn");
  await expect(page.getByRole("table")).toBeVisible();
  await expect(page.getByRole("img", { name: "数据集光谱概览" })).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});

test("workspace tabs retain sample drafts and block parsing changes until the draft is saved", async ({
  page,
}) => {
  const state = await setup(page);
  await page.goto("/workspace/nir/datasets/ds_corn");
  await page
    .getByRole("combobox", { name: "样本用途 1", exact: true })
    .selectOption("2");
  await page.getByRole("spinbutton", { name: "波段起始值" }).fill("1100");
  await page.getByRole("tab", { name: "数据概览", exact: true }).click();
  await expect(page.getByText("有未保存修改", { exact: true })).toBeVisible();
  await page.getByRole("tab", { name: "解析设置", exact: true }).click();
  await expect(
    page.getByText("请先保存样本用途与波段修改，再更改解析设置。", {
      exact: true,
    }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "样本与光谱", exact: true }).click();
  await expect(
    page.getByRole("combobox", { name: "样本用途 1", exact: true }),
  ).toHaveValue("2");
  await expect(
    page.getByRole("spinbutton", { name: "波段起始值" }),
  ).toHaveValue("1100");
  expect(state.saves).toHaveLength(0);
});
