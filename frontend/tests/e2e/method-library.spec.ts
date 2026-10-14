import { expect, test, type Page } from "@playwright/test";

import { methodCatalog } from "../fixtures/method-catalog";

import { mockLangGraphAPI, MOCK_THREAD_ID } from "./utils/mock-api";

async function mockCatalog(page: Page) {
  const catalog = structuredClone(methodCatalog);
  let writes = 0;
  await page.route("**/api/method-knowledge/**", async (route) => {
    const request = route.request();
    if (request.method() === "PUT") {
      writes++;
      const { expected_revision, changes } = request.postDataJSON();
      if (expected_revision !== catalog.revision)
        return route.fulfill({
          status: 409,
          json: { detail: "Catalog changed" },
        });
      const id = decodeURIComponent(
        new URL(request.url()).pathname.split("/").at(-1)!,
      );
      const index = catalog.cards.findIndex((card) => card.method_id === id);
      catalog.cards[index] = { ...catalog.cards[index]!, ...changes };
      catalog.revision++;
    }
    return route.fulfill({ json: catalog });
  });
  return { writes: () => writes };
}

test.describe("Method library", () => {
  test.beforeEach(async ({ page }) => {
    await mockLangGraphAPI(page).ready;
  });

  test("classifies preprocessing and opens read-only applicability and parameters", async ({
    page,
  }) => {
    const catalog = await mockCatalog(page);
    await page.goto("/workspace/methods");
    await expect(page.getByTestId("method-card")).toHaveCount(4);
    await page.getByRole("tab", { name: /^Preprocessing/ }).click();
    await expect(page.getByTestId("method-card")).toHaveCount(2);
    await page.getByRole("button", { name: /^Smoothing/ }).click();
    await expect(page.getByTestId("method-card")).toHaveCount(1);
    await page.getByTestId("method-card").click();
    const detail = page.getByRole("dialog", {
      name: "Savitzky-Golay 平滑",
      exact: true,
    });
    await expect(
      detail.getByRole("heading", { name: "Applicability", exact: true }),
    ).toBeVisible();
    await expect(detail).toContainText(
      "适合有高频噪声且需要保留吸收峰的光谱。",
    );
    await expect(detail).toContainText("过大的窗口可能削弱窄峰。");
    await expect(
      detail.getByRole("cell", { name: "window", exact: false }).first(),
    ).toBeVisible();
    await expect(
      detail.getByRole("cell", { name: "7", exact: true }),
    ).toBeVisible();
    await expect(
      detail.getByRole("link", { name: "View method source" }),
    ).toHaveAttribute("href", "https://chemotools.org/methods/");
    expect(catalog.writes()).toBe(0);
    if (process.env.METHOD_LIBRARY_QA_DIR)
      await page.screenshot({
        path: `${process.env.METHOD_LIBRARY_QA_DIR}/detail.png`,
        animations: "disabled",
      });
  });

  test("models and alignment have separate groups and applicability is searchable", async ({
    page,
  }) => {
    await mockCatalog(page);
    await page.goto("/workspace/methods");
    await page.getByRole("tab", { name: /^Modeling/ }).click();
    await expect(page.getByTestId("method-card")).toHaveCount(1);
    await expect(page.getByTestId("method-card")).toContainText("PLS 回归");
    await page.getByRole("tab", { name: /^Other/ }).click();
    await expect(page.getByTestId("method-card")).toContainText("光谱轴对齐");
    await page.getByRole("tab", { name: /^All methods/ }).click();
    await page.getByLabel("Search methods", { exact: true }).fill("颗粒大小");
    await expect(page.getByTestId("method-card")).toHaveCount(1);
    await expect(page.getByTestId("method-card")).toContainText("SNV 散射校正");
    await page
      .getByLabel("Search methods", { exact: true })
      .fill("no-such-method");
    await expect(
      page.getByText("No matching methods. Adjust the category or search."),
    ).toBeVisible();
  });

  test("settings still edit the same cards and refresh reads saved content", async ({
    page,
  }) => {
    const catalog = await mockCatalog(page);
    await page.route("**/api/knowledge/**", (route) =>
      route.fulfill({ status: 503, json: { detail: "Paper service offline" } }),
    );
    await page.goto("/workspace/methods");
    await page
      .locator("[data-sidebar='sidebar']")
      .getByRole("button", { name: /Settings and more/ })
      .click();
    await page.getByRole("menuitem", { name: "Settings", exact: true }).click();
    const settings = page.getByRole("dialog", {
      name: "Settings",
      exact: true,
    });
    await settings
      .getByRole("button", { name: "Knowledge", exact: true })
      .click();
    await settings
      .getByRole("tab", { name: "Method knowledge", exact: true })
      .click();
    const manager = settings.getByTestId("method-knowledge-manager");
    await expect(
      manager.getByText("Savitzky-Golay 平滑", { exact: true }),
    ).toBeVisible();
    await manager
      .getByRole("button", { name: "Edit", exact: true })
      .first()
      .click();
    const editor = page.getByRole("dialog", {
      name: "Edit method card",
      exact: true,
    });
    await editor
      .getByRole("textbox", { name: "Purpose and description", exact: true })
      .fill("Updated shared smoothing reference");
    await editor
      .getByRole("button", { name: "Save and publish", exact: true })
      .click();
    await expect(editor).not.toBeVisible();
    await settings.getByRole("button", { name: "Close", exact: true }).click();
    await page
      .getByTestId("method-library")
      .getByRole("button", { name: "Refresh", exact: true })
      .click();
    await expect(page.getByTestId("method-library")).toContainText(
      "Updated shared smoothing reference",
    );
    expect(catalog.writes()).toBe(1);
  });

  test("an API failure can be retried", async ({ page }) => {
    let failed = true;
    await page.route("**/api/method-knowledge/cards", (route) =>
      failed
        ? route.fulfill({
            status: 503,
            json: { detail: "Catalog temporarily unavailable" },
          })
        : route.fulfill({ json: methodCatalog }),
    );
    await page.goto("/workspace/methods");
    await expect(
      page.getByTestId("method-library").getByRole("alert"),
    ).toContainText("Catalog temporarily unavailable");
    failed = false;
    await page.getByRole("button", { name: "Refresh", exact: true }).click();
    await expect(page.getByTestId("method-card")).toHaveCount(4);
    await expect(
      page.getByTestId("method-library").getByRole("alert"),
    ).toHaveCount(0);
  });

  test("mobile library and details fit the viewport", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await mockCatalog(page);
    await page.goto("/workspace/methods");
    await expect(page.getByTestId("method-card")).toHaveCount(4);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth + 1,
      ),
    ).toBe(true);
    await page.getByTestId("method-card").first().click();
    const detail = page.getByRole("dialog", {
      name: "Savitzky-Golay 平滑",
      exact: true,
    });
    await expect(detail).toBeVisible();
    const box = await detail.boundingBox();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(391);
  });

  test("desktop library layout", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 960 });
    await mockCatalog(page);
    await page.goto("/workspace/methods");
    await expect(page.getByTestId("method-card")).toHaveCount(4);
    if (process.env.METHOD_LIBRARY_QA_DIR)
      await page.screenshot({
        path: `${process.env.METHOD_LIBRARY_QA_DIR}/library.png`,
        animations: "disabled",
      });
  });

  test("Chinese navigation and method details", async ({ page, baseURL }) => {
    await page
      .context()
      .addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
    await page.setViewportSize({ width: 1440, height: 960 });
    await mockCatalog(page);
    await page.goto("/workspace/methods");
    await expect(
      page.getByRole("heading", { name: "方法库", exact: true }),
    ).toBeVisible();
    await expect(
      page
        .locator("[data-sidebar='sidebar']")
        .getByRole("link", { name: "方法库", exact: true }),
    ).toBeVisible();
    await page.getByRole("tab", { name: /^预处理/ }).click();
    await page.getByRole("button", { name: /^散射校正/ }).click();
    await expect(page.getByTestId("method-card")).toHaveCount(1);
    await page.getByRole("tab", { name: /^全部方法/ }).click();
    if (process.env.METHOD_LIBRARY_QA_DIR)
      await page.screenshot({
        path: `${process.env.METHOD_LIBRARY_QA_DIR}/library-zh.png`,
        animations: "disabled",
      });
    await page.getByTestId("method-card").first().click();
    await expect(
      page
        .getByRole("dialog")
        .getByRole("heading", { name: "适用条件", exact: true }),
    ).toBeVisible();
    if (process.env.METHOD_LIBRARY_QA_DIR)
      await page.screenshot({
        path: `${process.env.METHOD_LIBRARY_QA_DIR}/detail-zh.png`,
        animations: "disabled",
      });
  });
});

test("historical agent links preserve chat and execution binding", async ({
  page,
}) => {
  await mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: MOCK_THREAD_ID,
        title: "Historical conversation",
        agent_name: "legacy-agent",
      },
    ],
  }).ready;
  let submitted: Record<string, unknown> | undefined;
  await page.route("**/runs/stream", (route) => {
    submitted = route.request().postDataJSON();
    return route.fallback();
  });
  await page.goto(`/workspace/agents/legacy-agent/chats/${MOCK_THREAD_ID}`);
  await expect(page).toHaveURL(`/workspace/chats/${MOCK_THREAD_ID}`);
  await expect(
    page.getByText("Response in thread Historical conversation"),
  ).toBeVisible();
  const input = page.getByPlaceholder(/how can i assist you/i);
  await input.fill("Continue this conversation");
  await input.press("Enter");
  await expect
    .poll(
      () =>
        (submitted?.context as Record<string, unknown> | undefined)?.agent_name,
    )
    .toBe("legacy-agent");
});

test("historical chat waits for metadata and supports retry before sending", async ({
  page,
}) => {
  await mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: MOCK_THREAD_ID,
        title: "Historical conversation",
        agent_name: "legacy-agent",
      },
    ],
  }).ready;
  let failed = true;
  await page.route(`**/api/langgraph/threads/${MOCK_THREAD_ID}`, (route) =>
    failed
      ? route.fulfill({
          status: 403,
          json: { detail: "Metadata request denied" },
        })
      : route.fallback(),
  );
  await page.goto(`/workspace/chats/${MOCK_THREAD_ID}`);
  const input = page.getByPlaceholder(/how can i assist you/i);
  await expect(page.locator('[data-slot="alert"]')).toContainText(
    "Could not load conversation details",
  );
  await expect(input).toBeDisabled();
  await expect(
    page.getByRole("button", { name: "Regenerate", exact: true }),
  ).toBeDisabled();
  failed = false;
  await page.getByRole("button", { name: "Retry", exact: true }).click();
  await expect(input).toBeEnabled();
  await expect(page.locator('[data-slot="alert"]')).toHaveCount(0);
});

test("removed management and evaluation URLs return to chats", async ({
  page,
}) => {
  await mockLangGraphAPI(page).ready;
  for (const path of [
    "/workspace/agents",
    "/workspace/agents/new",
    "/workspace/evaluations",
  ]) {
    await page.goto(path);
    await expect(page).toHaveURL("/workspace/chats");
  }
});
