import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

test("/datasets mounts without sending and includes the file only on manual submission", async ({
  page,
}) => {
  const routes = mockLangGraphAPI(page);
  await routes.ready;
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        datasets: [
          {
            id: "ds_corn",
            name: "corn-moisture",
            original_filename: "corn_moisture.mat",
            size_bytes: 454553,
            status: "ready",
            sha256: "a".repeat(64),
            media_type: "application/octet-stream",
            created_at: "2026-01-01T00:00:00Z",
          },
          {
            id: "ds_older",
            name: "older-sample",
            original_filename: "older.csv",
            size_bytes: 100,
            status: "archived",
            sha256: "b".repeat(64),
            media_type: "text/csv",
            created_at: "2026-01-01T00:00:00Z",
          },
        ],
      }),
    }),
  );
  let runCalls = 0;
  let submitted: Record<string, unknown> | undefined;
  let mountedThread = "";
  let uploads = 0;
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/datasets/ds_corn/attach", (route) => {
    mountedThread = route.request().postDataJSON().thread_id;
    return route.fulfill({
      json: {
        status: "attached",
        virtual_path: "/mnt/user-data/uploads/dataset-ds_corn.mat",
      },
    });
  });
  await page.route("**/api/threads/*/uploads", (route) => {
    uploads += 1;
    return route.abort();
  });
  await page.route("**/runs/stream", (route) => {
    runCalls += 1;
    submitted = route.request().postDataJSON();
    return route.fallback();
  });

  await page.goto("/workspace/chats/new");
  const textarea = page.getByPlaceholder(/how can i assist you/i);
  await expect(textarea).toBeVisible({ timeout: 15_000 });
  await textarea.fill("/datasets");
  await textarea.press("Enter");

  const dialog = page.getByRole("dialog", { name: "Spectral dataset library" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText("corn-moisture")).toBeVisible();
  await expect(dialog.getByText("older-sample")).toBeVisible();
  expect(runCalls).toBe(0);
  await expect(
    dialog
      .getByText("older-sample")
      .locator("..")
      .locator("..")
      .getByRole("button"),
  ).toBeDisabled();

  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .first()
    .click();
  await expect(dialog).toBeHidden();
  await expect(page.getByTestId("mounted-dataset")).toContainText(
    "corn-moisture",
  );
  await expect(textarea).toHaveValue("");
  expect(runCalls).toBe(0);
  expect(mountedThread).not.toBe("");
  await textarea.fill("Please inspect the target column before modeling");
  await textarea.press("Enter");
  await expect.poll(() => runCalls).toBe(1);
  expect(submitted).toMatchObject({
    input: {
      messages: [
        {
          content: [
            {
              type: "text",
              text: "Please inspect the target column before modeling",
            },
          ],
          additional_kwargs: {
            files: [
              {
                filename: "dataset-ds_corn.mat",
                path: "/mnt/user-data/uploads/dataset-ds_corn.mat",
                size: 454553,
                status: "uploaded",
              },
            ],
          },
        },
      ],
    },
  });
  expect(uploads).toBe(0);
  await expect(page.getByTestId("mounted-dataset")).toBeHidden();
});

for (const scenario of ["failure", "clear"] as const) {
  test(`dataset mounting ${scenario} never sends a synthetic message`, async ({
    page,
    context,
  }) => {
    await context.addCookies([
      { name: "locale", value: "zh-CN", domain: "localhost", path: "/" },
    ]);
    await mockLangGraphAPI(page).ready;
    await page.route("**/api/langgraph/threads", (route) =>
      route.fulfill({
        json: { ...route.request().postDataJSON(), status: "idle", values: {} },
      }),
    );
    await page.route("**/api/nir/datasets", (route) =>
      route.fulfill({
        json: {
          datasets: [
            {
              id: "ds_test",
              name: "光谱样本",
              original_filename: "sample.csv",
              size_bytes: 12,
              status: "ready",
              sha256: "a".repeat(64),
              media_type: "text/csv",
              created_at: "2026-10-06T00:00:00Z",
            },
          ],
        },
      }),
    );
    await page.route("**/api/nir/datasets/ds_test/attach", (route) =>
      route.fulfill(
        scenario === "failure"
          ? { status: 409, json: { detail: "挂载失败，请重试" } }
          : {
              json: {
                status: "attached",
                virtual_path: "/mnt/user-data/uploads/dataset-ds_test.csv",
              },
            },
      ),
    );
    let runs = 0;
    let submitted: Record<string, unknown> | undefined;
    await page.route("**/runs/stream", (route) => {
      runs++;
      submitted = route.request().postDataJSON();
      return route.fallback();
    });
    await page.goto("/workspace/chats/new");
    const textarea = page.getByPlaceholder("今天我能为你做些什么？");
    await textarea.fill("/datasets");
    await textarea.press("Enter");
    const dialog = page.getByRole("dialog", { name: "光谱数据集库" });
    await dialog
      .getByRole("button", { name: "挂载到会话", exact: true })
      .click();
    if (scenario === "failure") {
      await expect(dialog).toContainText("挂载失败，请重试");
      await expect(page.getByTestId("mounted-dataset")).toBeHidden();
      expect(runs).toBe(0);
    } else {
      await expect(dialog).toBeHidden();
      expect(runs).toBe(0);
      await page.getByRole("button", { name: "取消本次数据集选择" }).click();
      await expect(page.getByTestId("mounted-dataset")).toBeHidden();
      await textarea.fill("解释交叉验证");
      await textarea.press("Enter");
      await expect.poll(() => runs).toBe(1);
      expect(JSON.stringify(submitted)).not.toContain("dataset-ds_test.csv");
    }
  });
}

test("closing the picker cancels an unfinished mount without sending", async ({
  page,
}) => {
  await mockLangGraphAPI(page).ready;
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({
      json: {
        datasets: [
          {
            id: "ds_pending",
            name: "pending dataset",
            original_filename: "sample.csv",
            size_bytes: 12,
            status: "ready",
            sha256: "a".repeat(64),
            media_type: "text/csv",
            created_at: "2026-10-06T00:00:00Z",
          },
        ],
      },
    }),
  );
  let releaseMount!: () => void;
  const mountResponse = new Promise<void>((resolve) => {
    releaseMount = resolve;
  });
  let started = false;
  await page.route("**/api/nir/datasets/ds_pending/attach", async (route) => {
    started = true;
    await mountResponse;
    await route
      .fulfill({
        json: {
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/dataset-ds_pending.csv",
        },
      })
      .catch(() => {
        // An aborted browser request may already be disposed before fulfillment.
      });
  });
  let runs = 0;
  let submitted: Record<string, unknown> | undefined;
  await page.route("**/runs/stream", (route) => {
    runs++;
    submitted = route.request().postDataJSON();
    return route.fallback();
  });
  await page.goto("/workspace/chats/new");
  const textarea = page.getByPlaceholder(/how can i assist you/i);
  await textarea.fill("/datasets");
  await textarea.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Spectral dataset library" });
  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .click();
  await expect.poll(() => started).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  releaseMount();
  expect(runs).toBe(0);
  await textarea.fill("Explain cross validation");
  await textarea.press("Enter");
  await expect.poll(() => runs).toBe(1);
  expect(JSON.stringify(submitted)).not.toContain("dataset-ds_pending.csv");
  await expect(page.getByTestId("mounted-dataset")).toBeHidden();
});

for (const status of ["saved", "reused", "failed"] as const) {
  test(`uploaded dataset shows ${status} library outcome and keeps chat submission working`, async ({
    page,
    context,
  }) => {
    await context.addCookies([
      { name: "locale", value: "zh-CN", domain: "localhost", path: "/" },
    ]);
    const routes = mockLangGraphAPI(page);
    await routes.ready;
    let uploaded = false;
    let runs = 0;
    await page.route("**/api/threads/*/uploads", (route) => {
      uploaded = true;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          success: true,
          message: "Uploaded",
          skipped_files: [],
          files: [
            {
              filename: "光谱.csv",
              size: 8,
              path: "光谱.csv",
              virtual_path: "/mnt/user-data/uploads/光谱.csv",
              artifact_url: "/api/threads/test/uploads/光谱.csv",
              dataset_library: {
                status,
                dataset_id: status === "failed" ? null : "ds_uploaded",
                error_code: status === "failed" ? "quota_exceeded" : null,
              },
            },
          ],
        }),
      });
    });
    await page.route("**/runs/stream", (route) => {
      runs += 1;
      return route.fallback();
    });
    await page.route("**/api/nir/datasets", (route) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          datasets:
            status === "failed"
              ? []
              : [
                  {
                    id: "ds_uploaded",
                    name: "光谱.csv",
                    original_filename: "光谱.csv",
                    size_bytes: 8,
                    status: "ready",
                    sha256: "a".repeat(64),
                    media_type: "text/csv",
                    created_at: "2026-10-05T00:00:00Z",
                  },
                ],
        }),
      }),
    );
    await page.goto("/workspace/chats/new");
    const textarea = page.getByPlaceholder("今天我能为你做些什么？");
    await expect(textarea).toBeVisible({ timeout: 15_000 });
    await page.locator('input[type="file"]').setInputFiles({
      name: "光谱.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("x,y\n1,2\n"),
    });
    await textarea.fill("建立定量分析模型");
    await textarea.press("Enter");
    await expect.poll(() => uploaded).toBe(true);
    await expect.poll(() => runs).toBe(1);
    if (status === "failed") {
      await expect(
        page.getByText(/未保存至数据集库：数据集库容量不足/),
      ).toBeVisible();
      await expect(page.getByText(/当前对话仍可使用此文件/)).toBeVisible();
    } else {
      await expect(
        page.getByText(
          status === "saved"
            ? /1 个数据集已自动保存至数据集库/
            : /1 个数据集已在库中/,
        ),
      ).toBeVisible();
      await page.goto("/workspace/chats/new");
      await expect(textarea).toBeVisible();
      await textarea.fill("/datasets");
      await textarea.press("Enter");
      const dialog = page.getByRole("dialog", { name: "光谱数据集库" });
      await expect(dialog.getByText("光谱.csv", { exact: true })).toBeVisible();
      await expect(
        dialog.getByRole("button", { name: "挂载到会话", exact: true }),
      ).toBeEnabled();
    }
  });
}
