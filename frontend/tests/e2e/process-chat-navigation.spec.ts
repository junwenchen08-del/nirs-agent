import { expect, test, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const threadId = "00000000-0000-0000-0000-000000000081";
const otherThreadId = "00000000-0000-0000-0000-000000000082";
const chatPath = `/workspace/chats/${threadId}`;
const processPath = `/workspace/nir/process/${threadId}`;

async function setup(page: Page, locale: "zh-CN" | "en-US") {
  await mockLangGraphAPI(page, {
    locale,
    threads: [
      { thread_id: threadId, title: "Original modeling conversation" },
      { thread_id: otherThreadId, title: "Another conversation" },
    ],
  }).ready;
  // Navigation must work even when process evidence cannot be loaded.
  await page.route("**/api/threads/*/nir-process/**", (route) =>
    route.fulfill({ status: 403, json: { detail: "Evidence unavailable" } }),
  );
  const runs: string[] = [];
  await page.route("**/runs/stream", (route) => {
    runs.push(route.request().url());
    return route.abort();
  });
  return runs;
}

for (const locale of ["zh-CN", "en-US"] as const) {
  test(`process workspace breadcrumb returns to its own chat after direct entry (${locale})`, async ({
    page,
  }) => {
    const runs = await setup(page, locale);
    await page.goto(
      `${processPath}?nir_session=historical-session&nir_layout=graph`,
    );
    const workspace = page
      .getByRole("navigation", { name: "breadcrumb" })
      .getByRole("link", {
        name: locale === "zh-CN" ? "工作区" : "Workspace",
        exact: true,
      });
    await expect(workspace).toHaveAttribute("href", chatPath);
    await expect(workspace).toHaveAttribute(
      "title",
      locale === "zh-CN" ? "返回当前对话" : "Back to current chat",
    );
    await workspace.click();
    await expect(page).toHaveURL(new RegExp(`${chatPath}$`));
    await expect(page.getByRole("textbox").first()).toBeVisible();
    expect(runs).toEqual([]);
  });
}

test("process breadcrumb returns to the modeling chat instead of the previously visited chat", async ({
  page,
}) => {
  const runs = await setup(page, "zh-CN");
  await page.goto(`/workspace/chats/${otherThreadId}`);
  await expect(page.getByRole("textbox").first()).toBeVisible();
  await page.goto(processPath);
  await page
    .getByRole("navigation", { name: "breadcrumb" })
    .getByRole("link", { name: "工作区", exact: true })
    .click();
  await expect(page).toHaveURL(new RegExp(`${chatPath}$`));
  expect(runs).toEqual([]);
});

test("other workspace pages keep their existing workspace breadcrumb destination", async ({
  page,
}) => {
  await setup(page, "en-US");
  await page.goto("/workspace/methods");
  await expect(
    page
      .getByRole("navigation", { name: "breadcrumb" })
      .getByRole("link", { name: "Workspace", exact: true }),
  ).toHaveAttribute("href", "/workspace");
});
