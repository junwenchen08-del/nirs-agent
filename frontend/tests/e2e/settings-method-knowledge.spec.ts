import { expect, test, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const apiOrigin = process.env.NIR_METHOD_API_QA_URL;
test.skip(!apiOrigin, "Requires the isolated real method management API");

async function openMethods(page: Page) {
  mockLangGraphAPI(page);
  await page.route("**/api/knowledge/**", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Paper RAG is offline in this test" }),
    }),
  );
  await page.route("**/api/method-knowledge/**", async (route) => {
    const url = new URL(route.request().url());
    const response = await route.fetch({
      url: `${apiOrigin}${url.pathname}${url.search}`,
    });
    await route.fulfill({ response });
  });
  await page.goto("/workspace/chats/new");
  const sidebar = page.locator("[data-sidebar='sidebar']");
  await sidebar.getByRole("button", { name: /Settings and more/ }).click();
  await page.getByRole("menuitem", { name: "Settings", exact: true }).click();
  const settings = page.getByRole("dialog", { name: "Settings", exact: true });
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
  return { settings, manager };
}

async function catalog(page: Page) {
  return (
    await page.request.get(`${apiOrigin}/api/method-knowledge/cards`)
  ).json();
}

test.describe("Desktop method knowledge management", () => {
  test.use({ viewport: { width: 1560, height: 1050 } });
  test.beforeEach(async ({ page }) => {
    const current = await catalog(page);
    await page.request.post(
      `${apiOrigin}/api/method-knowledge/cards/sg_smooth/reset`,
      { data: { expected_revision: current.revision } },
    );
  });

  test("edits persist and search works while paper RAG is offline", async ({
    page,
  }) => {
    const { manager } = await openMethods(page);
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
      .fill("Browser-reviewed noise smoothing reference");
    await editor.getByLabel("candidate-1-window", { exact: true }).fill("13");
    if (process.env.NIR_METHOD_MANAGER_QA_SCREENSHOT)
      await page.screenshot({
        path: process.env.NIR_METHOD_MANAGER_QA_SCREENSHOT,
        animations: "disabled",
      });
    await editor
      .getByRole("button", { name: "Save and publish", exact: true })
      .click();
    await expect(editor).not.toBeVisible();
    let current = await catalog(page);
    let sg = current.cards.find(
      (card: { method_id: string }) => card.method_id === "sg_smooth",
    );
    expect(sg.candidate_params[0].window).toBe(13);
    await manager.getByLabel("Method search query").fill("noise smoothing");
    await manager.getByRole("button", { name: "Search", exact: true }).click();
    await expect(
      manager
        .getByTestId("method-search-results")
        .getByText("Browser-reviewed noise smoothing reference", {
          exact: true,
        }),
    ).toBeVisible();
    await page.reload();
    const reopened = await openMethods(page);
    await expect(
      reopened.manager.getByText("Browser-reviewed noise smoothing reference", {
        exact: true,
      }),
    ).toBeVisible();
    current = await catalog(page);
    sg = current.cards.find(
      (card: { method_id: string }) => card.method_id === "sg_smooth",
    );
    expect(sg.candidate_params[0].window).toBe(13);
  });

  test("illegal parameters and stale writes show errors without overwriting", async ({
    page,
  }) => {
    const { manager } = await openMethods(page);
    await manager
      .getByRole("button", { name: "Edit", exact: true })
      .first()
      .click();
    const editor = page.getByRole("dialog", {
      name: "Edit method card",
      exact: true,
    });
    await editor.getByLabel("candidate-1-window", { exact: true }).fill("10");
    await editor
      .getByRole("button", { name: "Save and publish", exact: true })
      .click();
    await expect(editor.getByRole("alert")).toContainText("window");
    let current = await catalog(page);
    const sg = current.cards.find(
      (card: { method_id: string }) => card.method_id === "sg_smooth",
    );
    expect(sg.candidate_params[0].window).toBe(7);
    await editor.getByLabel("candidate-1-window", { exact: true }).fill("13");
    await page.request.put(
      `${apiOrigin}/api/method-knowledge/cards/sg_smooth`,
      {
        data: {
          expected_revision: current.revision,
          changes: { title: "Saved in another window" },
        },
      },
    );
    await editor
      .getByRole("button", { name: "Save and publish", exact: true })
      .click();
    await expect(editor.getByRole("alert")).toContainText("Catalog changed");
    current = await catalog(page);
    expect(
      current.cards.find(
        (card: { method_id: string }) => card.method_id === "sg_smooth",
      ).title,
    ).toBe("Saved in another window");
  });

  test("retirement excludes a method and restore returns its packaged parameters", async ({
    page,
  }) => {
    const { manager } = await openMethods(page);
    const sg = manager.locator("article").first();
    await sg.getByRole("button", { name: "Retire", exact: true }).click();
    await expect(sg.getByText("Retired", { exact: true })).toBeVisible();
    await manager.getByLabel("Method search query").fill("sg_smooth");
    await manager.getByRole("button", { name: "Search", exact: true }).click();
    await expect(manager.getByTestId("method-search-results")).toBeVisible();
    await expect(
      manager
        .getByTestId("method-search-results")
        .getByText("Savitzky-Golay 平滑", { exact: true }),
    ).toHaveCount(0);
    await sg
      .getByRole("button", { name: "Restore default", exact: true })
      .click();
    await page
      .getByRole("dialog", { name: "Restore packaged reference" })
      .getByRole("button", { name: "Restore", exact: true })
      .click();
    await expect(sg.getByText("Published", { exact: true })).toBeVisible();
    const current = await catalog(page);
    expect(
      current.cards.find(
        (card: { method_id: string }) => card.method_id === "sg_smooth",
      ).candidate_params[0].window,
    ).toBe(7);
  });

  test("shows expanded model references and keeps model drafts out of retrieval", async ({
    page,
  }) => {
    const current = await catalog(page);
    expect(current.count).toBe(94);
    const { manager } = await openMethods(page);
    await expect(
      manager.getByRole("button", { name: "Add method card", exact: true }),
    ).toBeDisabled();
    await manager
      .getByLabel("Method type", { exact: true })
      .selectOption("modeling");
    await expect(manager.locator("article")).toHaveCount(4);
    const pls = manager
      .locator("article")
      .filter({ hasText: "PLS 偏最小二乘回归" });
    await pls
      .getByText("Applicability, limitations and parameters", { exact: true })
      .click();
    await expect(
      pls.getByText(/reference parameter edits do not change/),
    ).toBeVisible();
    await pls.getByRole("button", { name: "Edit", exact: true }).click();
    const editor = page.getByRole("dialog", {
      name: "Edit method card",
      exact: true,
    });
    await expect(
      editor.getByText(/These reference combinations/),
    ).toBeVisible();
    await editor
      .getByRole("textbox", { name: "Method title", exact: true })
      .fill("QA draft method reference");
    await editor
      .getByRole("textbox", { name: "Purpose and description", exact: true })
      .fill("Synthetic draft for method management QA");
    await editor
      .getByLabel("Source URL", { exact: true })
      .fill(
        "https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html",
      );
    await editor.getByLabel("Source section", { exact: true }).fill("Methods");
    await editor
      .getByLabel("Search keywords", { exact: true })
      .fill("qa_newmethod_card_test");
    await editor
      .getByRole("button", { name: "Save draft", exact: true })
      .click();
    await expect(editor).not.toBeVisible();
    const updated = await catalog(page);
    const created = updated.cards.find(
      (item: { method_id: string }) => item.method_id === "model_pls",
    );
    expect(created.review_status).toBe("draft");
    const result = await (
      await page.request.post(`${apiOrigin}/api/method-knowledge/search`, {
        data: { query: "qa_newmethod_card_test" },
      })
    ).json();
    expect(result.results).toEqual([]);
    await page.request.post(
      `${apiOrigin}/api/method-knowledge/cards/model_pls/reset`,
      { data: { expected_revision: updated.revision } },
    );
  });

  test("covers official methods and renders required MCP inputs separately from candidate settings", async ({
    page,
  }) => {
    const { manager } = await openMethods(page);
    await expect(manager.getByTestId("method-coverage")).toContainText("70/70");
    await expect(manager.getByTestId("method-coverage")).toContainText("84/84");
    await manager
      .getByLabel("Method category", { exact: true })
      .selectOption("feature_selection");
    await expect(manager.locator("article")).toHaveCount(4);
    await manager
      .getByLabel("Filter method list", { exact: true })
      .fill("VIPSelector");
    await expect(manager.locator("article")).toHaveCount(1);
    const vip = manager.locator("article");
    await vip
      .getByText("Applicability, limitations and parameters", { exact: true })
      .click();
    const reference = vip.getByTestId("mcp-parameter-reference");
    await expect(reference).toContainText("model");
    await expect(reference).toContainText("required");
    await expect(reference).toContainText("1");
    const expandedWidth = await manager.evaluate((element) => ({
      width: element.clientWidth,
      scroll: element.scrollWidth,
    }));
    expect(expandedWidth.scroll).toBeLessThanOrEqual(expandedWidth.width + 1);
    if (process.env.NIR_METHOD_MCP_QA_SCREENSHOT)
      await page.screenshot({
        path: process.env.NIR_METHOD_MCP_QA_SCREENSHOT,
        animations: "disabled",
      });
    await vip.getByRole("button", { name: "Edit", exact: true }).click();
    const editor = page.getByRole("dialog", {
      name: "Edit method card",
      exact: true,
    });
    await expect(editor.getByTestId("mcp-parameter-reference")).toBeVisible();
    await expect(editor.locator("[aria-label^='candidate-']")).toHaveCount(0);
    await editor
      .getByRole("textbox", { name: "Purpose and description", exact: true })
      .fill("Synthetic VIP review for required model input");
    await editor
      .getByRole("button", { name: "Save and publish", exact: true })
      .click();
    await expect(editor).not.toBeVisible();
    const current = await catalog(page);
    const saved = current.cards.find(
      (card: { method_id: string }) =>
        card.method_id === "chemotools.feature_selection.VIPSelector",
    );
    expect(saved.mcp_parameters.required).toContain("model");
    expect(saved.candidate_params).toEqual([{}]);
    await page.request.post(
      `${apiOrigin}/api/method-knowledge/cards/${encodeURIComponent(saved.method_id)}/reset`,
      { data: { expected_revision: current.revision } },
    );
    await manager
      .getByLabel("Method category", { exact: true })
      .selectOption("adaptation");
    await manager
      .getByLabel("Filter method list", { exact: true })
      .fill("check_metadata_function");
    await expect(manager.locator("article")).toHaveCount(1);
    await expect(manager.locator("article")).toContainText(
      "Documentation only / MCP unavailable",
    );
    const dimensions = await manager.evaluate((element) => ({
      width: element.clientWidth,
      scroll: element.scrollWidth,
    }));
    expect(dimensions.scroll).toBeLessThanOrEqual(dimensions.width + 1);
  });
});
