import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

test("new Chinese users get Chinese pages even with an English browser", async ({
  page,
  context,
  baseURL,
}) => {
  await context.clearCookies();
  await context.addCookies([
    { name: "access_token", value: "chemagent-brand-qa", url: baseURL! },
  ]);
  await page.goto("/");
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
  await expect(
    page.getByRole("link", { name: "文档", exact: true }),
  ).toHaveAttribute("href", "/zh/docs");
  await expect(
    page.getByText("近红外分析，从数据到模型", { exact: false }),
  ).toBeVisible();
  const mocks = mockLangGraphAPI(page, { locale: null });
  await mocks.ready;
  await page.goto("/workspace/chats/new");
  await expect(page.getByPlaceholder("今天我能为你做些什么？")).toBeVisible();
  await page
    .locator("[data-sidebar='sidebar']")
    .getByRole("button", { name: "设置和更多" })
    .click();
  await page.getByRole("menuitem", { name: "设置", exact: true }).click();
  const settings = page.getByRole("dialog");
  await expect(
    settings.getByRole("button", { name: "外观", exact: true }),
  ).toBeVisible();
  await settings.getByRole("combobox").click();
  await page.getByRole("option", { name: "English", exact: true }).click();
  await expect(
    settings.getByText("Adjust how ChemAgent looks and behaves for you."),
  ).toBeVisible();
  await page.reload();
  await expect(page.getByPlaceholder(/how can i assist you/i)).toBeVisible();
});

test("Chinese documentation and articles localize navigation, dates and help", async ({
  page,
  context,
}) => {
  await context.clearCookies();
  await page.goto("/zh/docs");
  await expect(page.getByText("反馈问题或建议", { exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText(
    /On This Page|Copy page|Edit this page|Scroll to top/,
  );
  await page.goto("/blog/posts");
  await expect(
    page.getByRole("heading", { name: "全部文章", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "近期文章", exact: true }),
  ).toBeVisible();
  await expect(page.locator("main")).toContainText("2026年10月17日");
  await page.locator('main a[href="/blog/chemagent-workflow"]').last().click();
  await expect(
    page.getByRole("heading", { name: "ChemAgent 分析流程", exact: true }),
  ).toBeVisible();
  await expect(page.locator("body")).not.toContainText(
    /All Posts|Recent Posts|Tags:|Language:|Last updated on/,
  );
});

test("first-time setup and authentication feedback are Chinese", async ({
  page,
  context,
  baseURL,
}) => {
  test.skip(
    process.env.CHEMAGENT_BRANDING_AUTH_QA !== "1",
    "Requires isolated authentication QA.",
  );
  await context.clearCookies();
  await page.goto("/setup");
  await expect(
    page.getByText("首次使用 ChemAgent，请先创建管理员账号。"),
  ).toBeVisible();
  await page.getByLabel("邮箱", { exact: true }).fill("qa@example.test");
  await page.getByLabel("密码", { exact: true }).fill("test-password-1");
  await page.getByLabel("确认新密码", { exact: true }).fill("test-password-2");
  await page
    .getByRole("button", { name: "创建管理员账号", exact: true })
    .click();
  await expect(page.getByText("两次输入的新密码不一致")).toBeVisible();
  await context.addCookies([
    { name: "access_token", value: "chemagent-setup-qa", url: baseURL! },
  ]);
  await page.goto("/setup");
  await expect(
    page.getByText("完成管理员账号设置", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "完成设置", exact: true }),
  ).toBeVisible();
});
