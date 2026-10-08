import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

const threadId = "00000000-0000-0000-0000-00000000d101";
const root = "/mnt/user-data/outputs/nir_analysis/";

test("new desktop delivery leads with the full report and one package", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  const files = [
    "report.html",
    "delivery.zip",
    "report.md",
    "model.pkl",
    "metrics.json",
    "raw_spectra.png",
  ].map((name) => root + name);
  mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: threadId,
        title: "NIR package",
        artifacts: files,
        messages: [
          {
            type: "human",
            id: "package-human",
            content: "Deliver the modeling result",
          },
          {
            type: "ai",
            id: "package-ai",
            content: "Results ready",
            tool_calls: [
              {
                id: "package-present",
                name: "present_files",
                args: { filepaths: files },
              },
            ],
          },
        ],
      },
    ],
  });
  await page.route(
    "**/artifacts/mnt/user-data/outputs/nir_analysis/report.html",
    (route) =>
      route.fulfill({
        contentType: "text/html",
        body: process.env.NIR_DELIVERY_QA_HTML
          ? readFileSync(process.env.NIR_DELIVERY_QA_HTML, "utf8")
          : "<html><body><h1>近红外光谱建模报告</h1></body></html>",
      }),
  );
  await page.goto(`/workspace/chats/${threadId}`);
  await expect(
    page.getByText("report.html", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText(/完整建模交付包|Complete modeling package/).first(),
  ).toBeVisible();
  await expect(
    page.getByText("report.md", { exact: true }).first(),
  ).not.toBeVisible();
  await page.getByText("report.html", { exact: true }).first().click();
  const preview = page.locator('#artifacts iframe[title="Artifact preview"]');
  await expect(preview).toBeVisible();
  await expect(
    preview
      .contentFrame()
      .getByRole("heading", { name: "近红外光谱建模报告", exact: true }),
  ).toBeVisible();
});

test("desktop delivery folds supporting files and renders relative report figures", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  const files = [
    "report.md",
    "model.pkl",
    "metrics.json",
    "raw_spectra.png",
  ].map((name) => root + name);
  mockLangGraphAPI(page, {
    threads: [
      {
        thread_id: threadId,
        title: "NIR delivery",
        artifacts: files,
        messages: [
          {
            type: "human",
            id: "delivery-human",
            content: "Deliver the modeling result",
          },
          {
            type: "ai",
            id: "delivery-ai",
            content: "Results ready",
            tool_calls: [
              {
                id: "delivery-present",
                name: "present_files",
                args: { filepaths: files },
              },
            ],
          },
        ],
      },
    ],
  });
  await page.route(
    "**/artifacts/mnt/user-data/outputs/nir_analysis/report.md",
    (route) =>
      route.fulfill({
        contentType: "text/markdown",
        body: "# 建模完整报告\n\n![原始光谱](raw_spectra.png)\n\n## 验证范围\n内部独立留出，不是外部验证",
      }),
  );
  await page.route(
    "**/artifacts/mnt/user-data/outputs/nir_analysis/raw_spectra.png",
    (route) =>
      route.fulfill({
        contentType: "image/png",
        body: Buffer.from(
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aV1sAAAAASUVORK5CYII=",
          "base64",
        ),
      }),
  );
  await page.goto(`/workspace/chats/${threadId}`);
  await expect(
    page.getByText("report.md", { exact: true }).first(),
  ).toBeVisible();
  await expect(
    page.getByText("model.pkl", { exact: true }).first(),
  ).not.toBeVisible();
  await page
    .locator("summary")
    .filter({ hasText: /辅助文件|supporting files/ })
    .first()
    .click();
  await expect(
    page.getByText("model.pkl", { exact: true }).first(),
  ).toBeVisible();
  await page.getByText("report.md", { exact: true }).first().click();
  const image = page
    .locator("#artifacts")
    .getByRole("img", { name: "原始光谱", exact: true });
  await expect(image).toBeVisible();
  await expect(image).toHaveJSProperty("naturalWidth", 1);
});

test("actual generated HTML displays offline figures and desktop layout", async ({
  page,
}) => {
  const fixture = process.env.NIR_DELIVERY_QA_HTML;
  test.skip(
    !fixture,
    "Set NIR_DELIVERY_QA_HTML to an actual trainer-generated report",
  );
  await page.setViewportSize({ width: 1440, height: 960 });
  await page.setContent(readFileSync(fixture!, "utf8"));
  await expect(
    page.getByRole("heading", { name: "1. 结论摘要" }),
  ).toBeVisible();
  const images = page.locator("figure img");
  expect(await images.count()).toBeGreaterThanOrEqual(3);
  for (const image of await images.all()) {
    await expect(image).toHaveAttribute("src", /^data:image\/png;base64,/);
    expect(
      await image.evaluate((element: HTMLImageElement) => element.naturalWidth),
    ).toBeGreaterThan(0);
  }
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  if (process.env.NIR_DELIVERY_QA_SCREENSHOT)
    await page.screenshot({
      path: process.env.NIR_DELIVERY_QA_SCREENSHOT,
      fullPage: true,
    });
});
