import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

test("a model selection survives failed data upload and is sent only after retry", async ({
  page,
}) => {
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
          {
            id: "nmv_test",
            model_id: "corn",
            version: "v1",
            status: "ready",
            method: "pls",
            validation_scope: "independent_holdout_not_external",
          },
        ],
      },
    }),
  );
  const modelPath = "/mnt/user-data/outputs/models/corn/v1/model.pkl";
  await page.route("**/api/nir/models/corn/versions/v1/attach", (route) =>
    route.fulfill({
      json: {
        status: "attached",
        model_id: "corn",
        version: "v1",
        model_path: modelPath,
      },
    }),
  );
  let uploadAttempts = 0;
  await page.route("**/api/threads/*/uploads", (route) => {
    uploadAttempts++;
    return route.fulfill(
      uploadAttempts === 1
        ? { status: 503, json: { detail: "Upload unavailable" } }
        : {
            json: {
              success: true,
              files: [
                {
                  filename: "new.csv",
                  size: 8,
                  virtual_path: "/mnt/user-data/uploads/new.csv",
                },
              ],
              skipped_files: [],
            },
          },
    );
  });
  let runs = 0;
  let submitted: unknown;
  await page.route("**/runs/stream", (route) => {
    runs++;
    submitted = route.request().postDataJSON();
    return route.fallback();
  });
  await page.goto("/workspace/chats/new");
  const input = page.getByPlaceholder(/how can i assist you/i);
  await input.fill("/models");
  await input.press("Enter");
  const dialog = page.getByRole("dialog", { name: "Persistent model library" });
  await dialog
    .getByRole("button", { name: "Attach to chat", exact: true })
    .click();
  await expect(dialog).toBeHidden();
  await page
    .locator('input[type="file"]')
    .first()
    .setInputFiles({
      name: "new.csv",
      mimeType: "text/csv",
      buffer: Buffer.from("x,y\n1,2\n"),
    });
  await input.fill("Predict my new data");
  await input.press("Enter");
  await expect.poll(() => uploadAttempts).toBe(1);
  await expect(page.getByTestId("mounted-model")).toContainText("corn · v1");
  await expect(input).toHaveValue("Predict my new data");
  await expect(page.getByText("new.csv", { exact: true })).toBeVisible();
  expect(runs).toBe(0);
  await expect(page.locator('button[type="submit"]')).toBeEnabled();
  await input.press("Enter");
  await expect.poll(() => runs).toBe(1);
  expect(uploadAttempts).toBe(2);
  expect(JSON.stringify(submitted)).toContain(modelPath);
  await expect(page.getByTestId("mounted-model")).toBeHidden();
});
