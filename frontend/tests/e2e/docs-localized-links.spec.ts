import { expect, test } from "@playwright/test";

test.describe("Localized documentation links", () => {
  test("keeps English card navigation in the English docs", async ({
    page,
  }) => {
    await page.goto("/en/docs/introduction/core-concepts");

    const card = page.locator("a.nextra-card", { hasText: "Why ChemAgent" });
    await expect(card).toHaveAttribute(
      "href",
      "/en/docs/introduction/why-chemagent",
    );

    await card.click();
    await expect(page).toHaveURL(/\/en\/docs\/introduction\/why-chemagent$/);
    await expect(page.locator("main h1")).toContainText("Why ChemAgent");
  });

  test("keeps Chinese card navigation in the Chinese docs", async ({
    page,
  }) => {
    await page.goto("/zh/docs/introduction/core-concepts");

    const card = page.locator("a.nextra-card", { hasText: "会话与分析记录" });
    await expect(card).toHaveAttribute(
      "href",
      "/zh/docs/application/agents-and-threads",
    );

    await card.click();
    await expect(page).toHaveURL(
      /\/zh\/docs\/application\/agents-and-threads$/,
    );
    await expect(page.locator("main h1")).toContainText("会话与分析记录");
  });

  test("localizes regular Markdown links", async ({ page }) => {
    await page.goto("/en/docs/application/workspace-usage");

    const link = page
      .locator("main")
      .getByRole("link", { name: "Chats and analysis records" })
      .first();
    await expect(link).toHaveAttribute(
      "href",
      "/en/docs/application/agents-and-threads",
    );

    await link.click();
    await expect(page).toHaveURL(
      /\/en\/docs\/application\/agents-and-threads$/,
    );
    await expect(page.locator("main h1")).toContainText(
      "Chats and analysis records",
    );
  });
});
