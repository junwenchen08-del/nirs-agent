import { expect, test, type Page } from "@playwright/test";

import { libraryDatasets, previewOf } from "../fixtures/dataset-library";

import { mockLangGraphAPI } from "./utils/mock-api";

async function setupLibrary(page: Page) {
  await mockLangGraphAPI(page).ready;
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({ json: { models: [] } }),
  );
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  const datasets = structuredClone(libraryDatasets);
  const attaches: Record<string, unknown>[] = [];
  const runs: Record<string, unknown>[] = [];
  let failAttach = false;
  await page.route("**/runs/stream", (route) => {
    runs.push(route.request().postDataJSON());
    return route.fallback();
  });
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({ json: { datasets, count: datasets.length } }),
  );
  await page.route("**/api/nir/datasets/**", async (route) => {
    const segments = new URL(route.request().url()).pathname.split("/");
    const id = segments[4]!;
    const dataset = datasets.find((row) => row.id === id);
    if (!dataset)
      return route.fulfill({ status: 404, json: { detail: "数据集不存在" } });
    if (segments[5] === "preview")
      return route.fulfill({
        json: previewOf(
          dataset,
          id === "ds_wheat"
            ? "classification"
            : id === "ds_unknown"
              ? "unknown"
              : "regression",
        ),
      });
    if (segments[5] === "uses")
      return route.fulfill({ json: { uses: [], count: 0 } });
    if (segments[5] === "attach") {
      attaches.push(route.request().postDataJSON());
      return failAttach
        ? route.fulfill({ status: 503, json: { detail: "挂载暂不可用" } })
        : route.fulfill({
            json: {
              status: "attached",
              virtual_path: `/mnt/user-data/uploads/${dataset.original_filename}`,
            },
          });
    }
    if (route.request().method() === "PATCH")
      dataset.name = route.request().postDataJSON().name;
    return route.fulfill({ json: { ...dataset, profiles: [] } });
  });
  return {
    datasets,
    attaches,
    runs,
    failAttach: (value: boolean) => {
      failAttach = value;
    },
  };
}

test.beforeEach(async ({ page, baseURL }) => {
  await page
    .context()
    .addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
});

test("local upload appears in the library and can be mounted into a new chat", async ({
  page,
}) => {
  const state = await setupLibrary(page);
  let attempts = 0;
  await page.route("**/api/nir/datasets/upload", async (route) => {
    attempts++;
    expect(route.request().headers()["content-type"]).toContain(
      "multipart/form-data; boundary=",
    );
    const body = route.request().postDataBuffer()!.toString("utf8");
    expect(body).toContain('filename="本地玉米.csv"');
    expect(body).toContain("本地玉米水分");
    expect(body).not.toContain("thread_id");
    if (attempts === 1)
      return route.fulfill({
        status: 507,
        json: { detail: { message: "空间不足，请重试。" } },
      });
    const dataset = {
      ...state.datasets[0]!,
      id: "ds_local",
      name: "本地玉米水分",
      original_filename: "本地玉米.csv",
    };
    state.datasets.push(dataset);
    return route.fulfill({ json: { ...dataset, reused_existing: false } });
  });
  await page.goto("/workspace/nir/datasets");
  await expect(page.getByTestId("nir-dataset-card")).toHaveCount(3);
  await page.getByRole("tab", { name: /^分类识别/ }).click();
  await page.getByRole("button", { name: "上传数据集", exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "上传数据集", exact: true });
  await expect(
    dialog.getByRole("button", { name: "上传到数据集库", exact: true }),
  ).toBeDisabled();
  await dialog.getByLabel("选择本地文件", { exact: true }).setInputFiles({
    name: "empty.csv",
    mimeType: "text/csv",
    buffer: Buffer.from(""),
  });
  await expect(dialog.getByRole("alert")).toContainText("请选择包含数据");
  await dialog.getByLabel("选择本地文件", { exact: true }).setInputFiles({
    name: "本地玉米.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("moisture,1000,1100\n1,2,3\n"),
  });
  await expect(dialog.getByRole("textbox")).toHaveValue("本地玉米.csv");
  await dialog.getByRole("textbox").fill("本地玉米水分");
  await expect(dialog).not.toContainText("源会话");
  if (process.env.DATASET_LIBRARY_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_LIBRARY_QA_DIR}/local-upload-zh.png`,
      animations: "disabled",
    });
  await dialog
    .getByRole("button", { name: "上传到数据集库", exact: true })
    .click();
  await expect(
    page.getByText("空间不足，请重试。", { exact: true }),
  ).toBeVisible();
  await expect(dialog.getByRole("textbox")).toHaveValue("本地玉米水分");
  await dialog
    .getByRole("button", { name: "上传到数据集库", exact: true })
    .click();
  await expect(dialog).not.toBeVisible();
  const card = page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "本地玉米水分" });
  await expect(card).toBeVisible();
  await card
    .getByRole("button", { name: "在新对话中使用", exact: true })
    .click();
  await expect(page.getByTestId("mounted-dataset")).toContainText(
    "本地玉米水分",
  );
  expect(state.attaches).toHaveLength(1);
  expect(state.runs).toHaveLength(0);
});

test("Chinese dataset summary, classification filters and charts replace storage space", async ({
  page,
}) => {
  await setupLibrary(page);
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.goto("/workspace/nir/datasets");
  await expect(page.getByTestId("nir-dataset-card")).toHaveCount(3);
  await expect(page.getByTestId("dataset-summary")).toContainText("个数据集");
  for (const label of ["个人空间", "长期模型", "长期数据集"])
    await expect(page.getByText(label, { exact: true })).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "归档", exact: true }),
  ).toHaveCount(0);
  await expect(page.getByRole("tab", { name: /^定量分析/ })).toContainText("1");
  if (process.env.DATASET_LIBRARY_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_LIBRARY_QA_DIR}/library-zh.png`,
      animations: "disabled",
    });
  await page.getByRole("tab", { name: /^分类识别/ }).click();
  await expect(page.getByTestId("nir-dataset-card")).toHaveCount(1);
  await page.getByRole("button", { name: /更多操作/ }).click();
  await page.getByRole("menuitem", { name: "数据概览与解析版本" }).click();
  const detail = page.getByRole("dialog", { name: "小麦品种", exact: true });
  await expect(
    detail.getByRole("img", { name: "平均光谱与变化范围图" }),
  ).toBeVisible();
  await expect(detail).toContainText("各类别样本分布");
  await expect(detail).toContainText("品种一");
  if (process.env.DATASET_LIBRARY_QA_DIR)
    await page.screenshot({
      path: `${process.env.DATASET_LIBRARY_QA_DIR}/detail-zh.png`,
      animations: "disabled",
    });
  await detail.getByRole("button", { name: "Close", exact: true }).click();
  await page.getByRole("tab", { name: /^待确认/ }).click();
  await expect(page.getByTestId("nir-dataset-card")).toContainText(
    "待分析光谱",
  );
});

test("attach creates a new chat with a file and starts no run before manual submission", async ({
  page,
}) => {
  const state = await setupLibrary(page);
  await page.goto("/workspace/nir/datasets");
  await page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "玉米水分" })
    .getByRole("button", { name: "在新对话中使用", exact: true })
    .click();
  await expect(page).toHaveURL(/\/workspace\/chats\/new/);
  await expect(page.getByTestId("mounted-dataset")).toContainText("玉米水分");
  expect(state.attaches).toHaveLength(1);
  expect(state.attaches[0]?.thread_id).toMatch(/^[0-9a-f-]{36}$/);
  expect(state.runs).toHaveLength(0);
  await expect(page.getByLabel("目标会话 ID", { exact: true })).toHaveCount(0);
  const input = page.getByPlaceholder("今天我能为你做些什么？");
  await input.fill("请分析这个数据集");
  await input.press("Enter");
  await expect.poll(() => state.runs.length).toBe(1);
  expect(JSON.stringify(state.runs[0])).toContain(
    "/mnt/user-data/uploads/corn.mat",
  );
});

test("failed attachment is visible, can retry and never starts an automatic run", async ({
  page,
}) => {
  const state = await setupLibrary(page);
  state.failAttach(true);
  await page.goto("/workspace/chats/new?dataset=ds_corn");
  await expect(
    page.getByText("文件未能挂载，请重试。", { exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId("mounted-dataset")).toHaveCount(0);
  expect(state.runs).toHaveLength(0);
  state.failAttach(false);
  await page.getByRole("button", { name: "重新挂载", exact: true }).click();
  await expect(page.getByTestId("mounted-dataset")).toContainText("玉米水分");
  expect(state.runs).toHaveLength(0);
});

test("rename still saves and mobile overview charts fit", async ({ page }) => {
  await setupLibrary(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/workspace/nir/datasets");
  const card = page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "玉米水分" });
  await card.getByRole("button", { name: /更多操作/ }).click();
  await page.getByRole("menuitem", { name: "重命名", exact: true }).click();
  const rename = page.getByRole("dialog", { name: "重命名", exact: true });
  await rename.getByRole("textbox").fill("玉米含水量");
  await rename.getByRole("button", { name: "保存", exact: true }).click();
  const renamed = page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "玉米含水量" });
  await expect(renamed).toBeVisible();
  await renamed.getByRole("button", { name: /更多操作/ }).click();
  await page.getByRole("menuitem", { name: "数据概览与解析版本" }).click();
  const detail = page.getByRole("dialog", { name: "玉米含水量", exact: true });
  await expect(
    detail.getByRole("img", { name: "平均光谱与变化范围图" }),
  ).toBeVisible();
  const box = await detail.boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(391);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});

test("preview failure keeps the dataset usable and can refresh", async ({
  page,
}) => {
  await setupLibrary(page);
  let failed = true;
  await page.route("**/api/nir/datasets/ds_corn/preview", (route) =>
    failed
      ? route.fulfill({ status: 503, json: { detail: "Unavailable" } })
      : route.fallback(),
  );
  await page.goto("/workspace/nir/datasets");
  await page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "玉米水分" })
    .getByRole("button", { name: /更多操作/ })
    .click();
  await page.getByRole("menuitem", { name: "数据概览与解析版本" }).click();
  const detail = page.getByRole("dialog", { name: "玉米水分", exact: true });
  await expect(detail).toContainText("数据预览读取失败");
  failed = false;
  await detail.getByRole("button", { name: "刷新", exact: true }).click();
  await expect(
    detail.getByRole("img", { name: "平均光谱与变化范围图" }),
  ).toBeVisible();
});

test("preview identity mismatch is not displayed", async ({ page }) => {
  await setupLibrary(page);
  await page.route("**/api/nir/datasets/ds_corn/preview", (route) =>
    route.fulfill({
      json: { ...previewOf(libraryDatasets[0]!), source_sha256: "incorrect" },
    }),
  );
  await page.goto("/workspace/nir/datasets");
  await page
    .getByTestId("nir-dataset-card")
    .filter({ hasText: "玉米水分" })
    .getByRole("button", { name: /更多操作/ })
    .click();
  await page.getByRole("menuitem", { name: "数据概览与解析版本" }).click();
  const detail = page.getByRole("dialog", { name: "玉米水分", exact: true });
  await expect(detail).toContainText("数据预览读取失败");
  await expect(detail.getByRole("img")).toHaveCount(0);
});

test("leaving during attachment cancels the request without carrying a file to another page", async ({
  page,
}) => {
  const state = await setupLibrary(page);
  let release: () => void = () => {
    /* Assigned when the blocked promise is created. */
  };
  const blocked = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requested = false;
  await page.route("**/api/nir/datasets/ds_corn/attach", async (route) => {
    requested = true;
    await blocked;
    await route
      .fulfill({
        json: {
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/corn.mat",
        },
      })
      .catch(() => {
        /* The browser may have cancelled this intercepted request after navigation. */
      });
  });
  await page.goto("/workspace/chats/new?dataset=ds_corn");
  await expect.poll(() => requested).toBe(true);
  await page
    .locator("[data-sidebar='sidebar'] a[href='/workspace/nir/datasets']")
    .click();
  release();
  await expect(page).toHaveURL(/\/workspace\/nir\/datasets/);
  await expect(page.getByTestId("mounted-dataset")).toHaveCount(0);
  expect(state.runs).toHaveLength(0);
});
