import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

test("/datasets lists saved datasets and starts a new chat with the selected one", async ({
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
  await page.route("**/runs/stream", (route) => {
    runCalls += 1;
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
  await expect.poll(() => runCalls).toBe(1);
});
