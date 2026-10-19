import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

test.describe("Landing page", () => {
  test("renders the header and hero section", async ({ page }) => {
    await page.goto("/");

    // Header brand name
    await expect(
      page.locator("header h1", { hasText: "ChemAgent" }),
    ).toBeVisible();

    // Workspace call-to-action button in hero
    await expect(
      page
        .locator("main")
        .getByRole("link", { name: "进入工作台", exact: true }),
    ).toBeVisible();
  });

  test("Get Started link navigates to workspace", async ({ page }) => {
    mockLangGraphAPI(page, { locale: "zh-CN" });

    await page.goto("/");

    const getStarted = page.locator("main").getByRole("link", {
      name: "进入工作台",
      exact: true,
    });
    await getStarted.click();

    // Should redirect to /workspace/chats/new
    await page.waitForURL("**/workspace/chats/new");
    await expect(page).toHaveURL(/\/workspace\/chats\/new/);
  });
});
