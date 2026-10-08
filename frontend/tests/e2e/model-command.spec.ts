import { expect, test, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const model = {
  id: "nmv_corn",
  model_id: "corn_pls",
  version: "v2",
  status: "ready",
  method: "pls",
  validation_scope: "independent_holdout_not_external",
};
const attachment = {
  status: "attached",
  model_id: "corn_pls",
  version: "v2",
  model_path: "/mnt/user-data/outputs/models/corn_pls/v2/model.pkl",
};

async function setup(page: Page) {
  await mockLangGraphAPI(page).ready;
  await page.route("**/api/langgraph/threads", (route) =>
    route.fulfill({
      json: { ...route.request().postDataJSON(), status: "idle", values: {} },
    }),
  );
  await page.route("**/api/nir/models?include_archived=true", (route) =>
    route.fulfill({
      json: {
        models: [
          model,
          { ...model, id: "nmv_old", version: "v1", status: "archived" },
        ],
      },
    }),
  );
  await page.route("**/api/nir/models/corn_pls/versions/v2/attach", (route) =>
    route.fulfill({ json: attachment }),
  );
  const state = {
    runs: 0,
    uploads: 0,
    submitted: {} as Record<string, unknown>,
  };
  await page.route("**/runs/stream", (route) => {
    state.runs++;
    state.submitted = route.request().postDataJSON();
    return route.fallback();
  });
  await page.route("**/api/threads/*/uploads", (route) => {
    state.uploads++;
    return route.abort();
  });
  await page.goto("/workspace/chats/new");
  return state;
}

async function openPicker(page: Page) {
  const input = page.getByPlaceholder(
    /how can i assist you|今天我能为你做些什么/i,
  );
  await input.fill("/models");
  await input.press("Enter");
  const dialog = page.getByRole("dialog", {
    name: /Persistent model library|持久模型库/,
  });
  await expect(dialog).toBeVisible();
  return dialog;
}

test("/models mounts an exact version without sending, then combines with /datasets on manual send", async ({
  page,
}) => {
  const state = await setup(page);
  await page.route("**/api/nir/datasets", (route) =>
    route.fulfill({
      json: {
        datasets: [
          {
            id: "ds_new",
            name: "New spectra",
            status: "ready",
            original_filename: "new.csv",
            size_bytes: 8,
          },
        ],
      },
    }),
  );
  await page.route("**/api/nir/datasets/ds_new/attach", (route) =>
    route.fulfill({
      json: {
        status: "attached",
        virtual_path: "/mnt/user-data/uploads/new.csv",
      },
    }),
  );
  const dialog = await openPicker(page);
  await expect(
    dialog
      .getByTestId("model-picker-row")
      .filter({ hasText: "v1" })
      .getByRole("button"),
  ).toBeDisabled();
  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .first()
    .click();
  await expect(dialog).toBeHidden();
  await expect(page.getByTestId("mounted-model")).toContainText(
    "corn_pls · v2",
  );
  expect(state.runs).toBe(0);
  const input = page.getByPlaceholder(/how can i assist you/i);
  await input.fill("/datasets");
  await input.press("Enter");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Attach to chat", exact: true })
    .click();
  await expect(page.getByTestId("mounted-dataset")).toBeVisible();
  expect(state.runs).toBe(0);
  await input.fill("Predict this new dataset without retraining");
  await input.press("Enter");
  await expect.poll(() => state.runs).toBe(1);
  const payload = JSON.stringify(state.submitted);
  expect(payload).toContain("Predict this new dataset without retraining");
  expect(payload).toContain(attachment.model_path);
  expect(payload).toContain('\\"version\\":\\"v2\\"');
  expect(state.submitted).toMatchObject({
    input: {
      messages: [
        {
          additional_kwargs: {
            files: [{ path: "/mnt/user-data/uploads/new.csv" }],
          },
        },
      ],
    },
  });
  expect(state.uploads).toBe(0);
  await expect(page.getByTestId("mounted-model")).toBeHidden();
});

for (const scenario of ["failure", "mismatch", "clear"] as const) {
  test(`Chinese model picker ${scenario} never sends a message`, async ({
    page,
    context,
  }) => {
    await context.addCookies([
      { name: "locale", value: "zh-CN", domain: "localhost", path: "/" },
    ]);
    const state = await setup(page);
    if (scenario !== "clear") {
      await page.route(
        "**/api/nir/models/corn_pls/versions/v2/attach",
        (route) =>
          route.fulfill(
            scenario === "failure"
              ? { status: 409, json: { detail: "模型挂载失败，请重试" } }
              : { json: { ...attachment, version: "other" } },
          ),
      );
    }
    const dialog = await openPicker(page);
    await dialog
      .getByRole("button", { name: "挂载到会话", exact: true })
      .first()
      .click();
    if (scenario !== "clear") {
      await expect(dialog.getByRole("alert")).toContainText(
        scenario === "failure"
          ? "模型挂载失败，请重试"
          : "模型挂载未成功，请重试",
      );
      await expect(page.getByTestId("mounted-model")).toBeHidden();
      expect(state.runs).toBe(0);
      // A failed mount can be retried without reopening or losing the selection.
      await page.route(
        "**/api/nir/models/corn_pls/versions/v2/attach",
        (route) => route.fulfill({ json: attachment }),
      );
      await dialog
        .getByRole("button", { name: "挂载到会话", exact: true })
        .first()
        .click();
    }
    await expect(dialog).toBeHidden();
    await expect(page.getByTestId("mounted-model")).toBeVisible();
    expect(state.runs).toBe(0);
    await page.getByRole("button", { name: "取消本次模型选择" }).click();
    await expect(page.getByTestId("mounted-model")).toBeHidden();
    const input = page.getByPlaceholder("今天我能为你做些什么？");
    await input.fill("解释交叉验证");
    await input.press("Enter");
    await expect.poll(() => state.runs).toBe(1);
    expect(JSON.stringify(state.submitted)).not.toContain(
      attachment.model_path,
    );
  });
}

test("typing /models and clicking send retains previously uploaded prediction data", async ({
  page,
}) => {
  const state = await setup(page);
  await page
    .locator('input[type="file"]')
    .first()
    .setInputFiles({
      name: "new.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("x,y\n1,2\n"),
    });
  const input = page.getByPlaceholder(/how can i assist you/i);
  await input.fill("/models");
  await input.press("Escape"); // Exercise form submission rather than autocomplete selection.
  await page.locator('button[type="submit"]').click();
  const dialog = page.getByRole("dialog", { name: "Persistent model library" });
  await expect(dialog).toBeVisible();
  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .first()
    .click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText("new.csv", { exact: true })).toBeVisible();
  expect(state.runs).toBe(0);
  expect(state.uploads).toBe(0);
  await page.route("**/api/threads/*/uploads", (route) => {
    state.uploads++;
    return route.fulfill({
      json: {
        success: true,
        message: "Uploaded",
        files: [
          {
            filename: "new.csv",
            size: 8,
            path: "new.csv",
            virtual_path: "/mnt/user-data/uploads/new.csv",
          },
        ],
        skipped_files: [],
      },
    });
  });
  await input.fill("Predict using the selected model");
  await input.press("Enter");
  await expect.poll(() => state.runs).toBe(1);
  expect(state.uploads).toBe(1);
  expect(JSON.stringify(state.submitted)).toContain(attachment.model_path);
  expect(state.submitted).toMatchObject({
    input: {
      messages: [
        {
          additional_kwargs: {
            files: [{ path: "/mnt/user-data/uploads/new.csv" }],
          },
        },
      ],
    },
  });
});

test("closing the model picker cancels pending attachment", async ({
  page,
}) => {
  const state = await setup(page);
  let release!: () => void;
  const wait = new Promise<void>((resolve) => {
    release = resolve;
  });
  let started = false;
  await page.route(
    "**/api/nir/models/corn_pls/versions/v2/attach",
    async (route) => {
      started = true;
      await wait;
      await route.fulfill({ json: attachment }).catch(() => {
        // Closing the picker aborts the request before this response arrives.
      });
    },
  );
  const dialog = await openPicker(page);
  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .first()
    .click();
  await expect.poll(() => started).toBe(true);
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  release();
  const input = page.getByPlaceholder(/how can i assist you/i);
  await input.fill("Explain model validation");
  await input.press("Enter");
  await expect.poll(() => state.runs).toBe(1);
  expect(JSON.stringify(state.submitted)).not.toContain(attachment.model_path);
  await expect(page.getByTestId("mounted-model")).toBeHidden();
});
