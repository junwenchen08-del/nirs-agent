import { expect, test, type Locator, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const DOCUMENT = {
  doc_id: "knowledge-layout-test",
  title:
    "Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat",
  source:
    "Linearization and Scatter-Correction for Near-Infrared Reflectance Spectra of Meat.pdf",
  year: 2024,
  chunk_count: 24,
  review_status: "published",
  authors: ["Example Author"],
  doi: "10.1000/example",
  language: "en",
  domains: ["NIR"],
  quality_tier: "A",
  publication_readiness: {
    ready: true,
    missing_fields: [],
    message: "",
  },
};

async function openKnowledgeSettings(page: Page) {
  mockLangGraphAPI(page);
  await page.route("**/api/knowledge/documents", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ documents: [DOCUMENT] }),
    }),
  );
  await page.route("**/api/knowledge/stats", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ documents: 1, total_chunks: 24 }),
    }),
  );

  await page.goto("/workspace/chats/new");
  const sidebar = page.locator("[data-sidebar='sidebar']");
  await sidebar.getByRole("button", { name: /Settings and more/ }).click();
  await page.getByRole("menuitem", { name: "Settings" }).click();

  const dialog = page.getByRole("dialog", { name: "Settings" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Knowledge" }).click();
  await expect(
    dialog.getByText("Knowledge Base", { exact: true }),
  ).toBeVisible();
  return dialog;
}

async function expectFullyVisibleInSettingsContent(locator: Locator) {
  await expect(locator).toBeVisible();
  const bounds = await locator.evaluate((element) => {
    const elementRect = element.getBoundingClientRect();
    const viewport = element.closest('[data-slot="scroll-area-viewport"]');
    const viewportRect = viewport?.getBoundingClientRect();

    return {
      withinWindow:
        elementRect.left >= 0 &&
        elementRect.right <= window.innerWidth &&
        elementRect.top >= 0 &&
        elementRect.bottom <= window.innerHeight,
      withinSettingsContent:
        viewportRect !== undefined &&
        elementRect.left >= viewportRect.left &&
        elementRect.right <= viewportRect.right,
    };
  });

  expect(bounds).toEqual({
    withinWindow: true,
    withinSettingsContent: true,
  });
}

test.describe("Knowledge settings layout", () => {
  test.use({ viewport: { width: 1106, height: 733 } });

  test("keeps toolbar and document actions visible at compact desktop widths", async ({
    page,
  }) => {
    const dialog = await openKnowledgeSettings(page);

    await expectFullyVisibleInSettingsContent(
      dialog.getByRole("button", { name: "Refresh" }),
    );
    await expectFullyVisibleInSettingsContent(
      dialog.getByRole("button", { name: "Upload Document" }),
    );
    await expectFullyVisibleInSettingsContent(
      dialog.getByRole("button", { name: "Edit" }),
    );
    await expectFullyVisibleInSettingsContent(
      dialog.getByRole("button", { name: "Delete" }),
    );
  });
});
