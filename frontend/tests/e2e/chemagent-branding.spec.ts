import { expect, test, type Page } from "@playwright/test";

import { BRAND } from "../../src/core/branding";

import { mockLangGraphAPI } from "./utils/mock-api";

async function captureSurface(page: Page, name: string) {
  const directory = process.env.CHEMAGENT_BRANDING_QA_OUTPUT;
  if (directory) {
    await page.screenshot({
      path: `${directory}/${name}.png`,
      animations: "disabled",
    });
  }
}

async function expectChemAgentSurface(page: Page) {
  await expect(page.locator("body")).not.toContainText(/deer[\s_-]?flow|🦌/i);
  await expect(
    page.locator(
      'a[href*="deerflow"], a[href*="deer-flow"], a[href*="bytedance"]',
    ),
  ).toHaveCount(0);
  await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
    "href",
    "/favicon.svg",
  );
}

test("landing, documentation and articles present ChemAgent", async ({
  page,
}) => {
  for (const route of [
    "/",
    "/zh/docs",
    "/en/docs",
    "/blog/posts",
    "/blog/chemagent-workflow",
  ]) {
    await page.goto(route);
    await expect(page.locator("main").first()).toContainText("ChemAgent");
    await expectChemAgentSurface(page);
    if (route === "/blog/posts") {
      await expect(
        page.locator('main a[href="/blog/chemagent-workflow"]').first(),
      ).toBeVisible();
    }
    await captureSurface(page, route.replace(/\W+/g, "-") || "home");
  }
});

for (const locale of ["en-US", "zh-CN"] as const) {
  test(`workspace navigation and about use ChemAgent in ${locale}`, async ({
    page,
    context,
    baseURL,
  }) => {
    await mockLangGraphAPI(page, { locale }).ready;
    await context.addCookies([
      { name: "locale", value: locale, url: baseURL! },
      { name: "access_token", value: "chemagent-brand-qa", url: baseURL! },
    ]);
    await page.goto("/workspace/chats/new");
    await expect(page.locator("body")).toContainText("ChemAgent");
    await expectChemAgentSurface(page);
    await captureSurface(page, `workspace-${locale}`);
    const menuButton = page
      .locator("[data-sidebar='sidebar']")
      .getByRole("button", {
        name: locale === "zh-CN" ? "设置和更多" : "Settings and more",
      });
    await menuButton.click();
    const docsLink = page.getByRole("link", {
      name:
        locale === "zh-CN" ? "ChemAgent 使用文档" : "ChemAgent documentation",
    });
    await expect(docsLink).toHaveAttribute(
      "href",
      `/${locale.slice(0, 2)}/docs`,
    );
    await expect(
      page.locator(`a[href="${BRAND.repositoryUrl}"]`).last(),
    ).toBeVisible();
    await page
      .getByRole("menuitem", {
        name: locale === "zh-CN" ? "关于 ChemAgent" : "About ChemAgent",
      })
      .click();
    const dialog = page.getByRole("dialog");
    await expect(
      dialog.getByRole("heading", { name: /关于 ChemAgent|About ChemAgent/ }),
    ).toBeVisible();
    await expect(dialog.locator('a[href^="mailto:"]')).toHaveCount(0);
    await expectChemAgentSurface(page);
    await captureSurface(page, `about-${locale}`);
  });
}

test("sharing a chat keeps the current ChemAgent instance", async ({
  page,
  context,
  baseURL,
}) => {
  const threadId = "00000000-0000-0000-0000-000000000007";
  mockLangGraphAPI(page, {
    threads: [{ thread_id: threadId, title: "ChemAgent analysis" }],
  });
  await context.addCookies([
    { name: "locale", value: "en-US", url: baseURL! },
    { name: "access_token", value: "chemagent-brand-qa", url: baseURL! },
  ]);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/workspace/chats/new");
  const item = page.locator("[data-sidebar='menu-item']").filter({
    has: page.getByRole("link", { name: "ChemAgent analysis", exact: true }),
  });
  await item.hover();
  await item.getByRole("button", { name: "More", exact: true }).click();
  await page.getByRole("menuitem", { name: "Share", exact: true }).click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(`${new URL(baseURL!).origin}/workspace/chats/${threadId}`);
});

test("login and both setup forms use spectral ChemAgent branding", async ({
  page,
  context,
  baseURL,
}) => {
  test.skip(
    process.env.CHEMAGENT_BRANDING_AUTH_QA !== "1",
    "Requires the isolated authentication QA server.",
  );
  await context.clearCookies();
  const requestedAssets: string[] = [];
  page.on("request", (request) => requestedAssets.push(request.url()));
  for (const route of ["/login", "/setup"]) {
    await page.goto(route);
    await expect(
      page.getByRole("heading", { name: "ChemAgent", exact: true }),
    ).toBeVisible();
    await expect(page.locator("form")).toBeVisible();
    await expectChemAgentSurface(page);
    await expect(page.locator('[class*="nir-spectrum.svg"]')).toHaveCount(1);
    await captureSurface(page, route.slice(1));
  }
  await context.addCookies([
    { name: "access_token", value: "chemagent-setup-qa", url: baseURL! },
  ]);
  await page.goto("/setup");
  await expect(
    page.getByText("完成管理员账号设置", { exact: true }),
  ).toBeVisible();
  await expectChemAgentSurface(page);
  await captureSurface(page, "setup-change-password");
  expect(
    requestedAssets.some((url) => /deer\.svg|favicon\.ico/.test(url)),
  ).toBe(false);
});
