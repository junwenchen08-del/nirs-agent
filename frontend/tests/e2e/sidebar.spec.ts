import { expect, test } from "@playwright/test";

import { methodCatalog } from "../fixtures/method-catalog";

import { mockLangGraphAPI } from "./utils/mock-api";

test.describe("Sidebar navigation", () => {
  test("shows methods and existing libraries without removed entries", async ({
    page,
  }) => {
    await mockLangGraphAPI(page).ready;
    await page.route("**/api/features", (route) =>
      route.fulfill({
        json: { nir_library: { enabled: true }, agents_api: { enabled: true } },
      }),
    );
    await page.goto("/workspace/chats/new");
    const sidebar = page.locator("[data-sidebar='sidebar']");
    for (const href of [
      "/workspace/chats",
      "/workspace/nir/datasets",
      "/workspace/nir/models",
      "/workspace/methods",
    ]) {
      await expect(sidebar.locator(`a[href='${href}']`)).toBeVisible();
    }
    await expect(sidebar.locator("a[href='/workspace/agents']")).toHaveCount(0);
    await expect(
      sidebar.getByRole("button", { name: "Agents", exact: true }),
    ).toHaveCount(0);
    await expect(
      sidebar.locator("a[href='/workspace/evaluations']"),
    ).toHaveCount(0);
  });

  test("method library is independent of asset-library feature flags", async ({
    page,
  }) => {
    await mockLangGraphAPI(page).ready;
    await page.route("**/api/features", (route) =>
      route.fulfill({ json: { nir_library: { enabled: false } } }),
    );
    await page.route("**/api/method-knowledge/cards", (route) =>
      route.fulfill({ json: methodCatalog }),
    );
    await page.goto("/workspace/chats/new");
    await page
      .locator("[data-sidebar='sidebar'] a[href='/workspace/methods']")
      .click();
    await expect(page).toHaveURL(/\/workspace\/methods/);
    await expect(
      page.getByRole("heading", { name: "Method library", exact: true }),
    ).toBeVisible();
    await expect(page.getByTestId("method-card")).toHaveCount(4);
  });

  test("mobile welcome and sidebar remain usable", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await mockLangGraphAPI(page).ready;
    await page.goto("/workspace/chats/new");
    const expectInsideViewport = async (
      locator: ReturnType<typeof page.locator>,
    ) => {
      await expect(locator).toBeVisible();
      const box = await locator.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(-1);
      expect(box!.x + box!.width).toBeLessThanOrEqual(391);
    };
    await expectInsideViewport(
      page.getByRole("heading", { name: "Start a spectral analysis" }),
    );
    await expectInsideViewport(page.getByRole("textbox").first());
    await expectInsideViewport(
      page.getByRole("button", { name: "Upload data", exact: true }),
    );
    await expectInsideViewport(
      page.getByRole("button", { name: "Choose a dataset", exact: true }),
    );
    await expectInsideViewport(
      page.getByRole("button", { name: "Use a saved model", exact: true }),
    );
    await page.locator("[data-sidebar='trigger']:visible").first().click();
    const sidebar = page.locator(
      "[data-mobile='true'][data-sidebar='sidebar']",
    );
    await expect(sidebar.locator("a[href='/workspace/chats']")).toBeVisible();
    await expect(sidebar.locator("a[href='/workspace/methods']")).toBeVisible();
    await expect(
      sidebar.locator(
        "a[href='/workspace/agents'], a[href='/workspace/evaluations']",
      ),
    ).toHaveCount(0);
  });
});
