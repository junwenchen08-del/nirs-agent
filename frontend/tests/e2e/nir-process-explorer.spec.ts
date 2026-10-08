import fs from "node:fs";
import path from "node:path";

import { expect, test } from "@playwright/test";

import {
  type CandidatePage,
  type ProcessChart,
  type ProcessStep,
  type ProcessSummary,
  type RunList,
} from "@/core/nir/process-contract";

import { mockLangGraphAPI } from "./utils/mock-api";

const evidencePath = process.env.NIR_PROCESS_QA_FIXTURE;
test.skip(
  !evidencePath,
  "Set NIR_PROCESS_QA_FIXTURE to an export of the real training test store.",
);
const evidence = evidencePath
  ? (JSON.parse(fs.readFileSync(evidencePath, "utf-8")) as {
      runs: RunList;
      summaries: Record<string, ProcessSummary>;
      steps: Record<string, ProcessStep>;
      charts: Record<string, ProcessChart>;
      candidates: Record<string, CandidatePage>;
    })
  : null;
const threadId = "process-qa-thread";

test.beforeEach(async ({ page, context }) => {
  await context.addCookies([
    { name: "locale", value: "zh-CN", domain: "localhost", path: "/" },
  ]);
  mockLangGraphAPI(page, { threads: [] });
  await page.route("**/api/threads/*/nir-process/**", async (route) => {
    const url = new URL(route.request().url());
    const segments = url.pathname.split("/nir-process/")[1]!.split("/");
    const rid = segments[1];
    let payload: unknown = evidence!.runs;
    if (rid) payload = evidence!.summaries[rid];
    if (segments[2] === "steps")
      payload =
        segments[4] === "candidates"
          ? evidence!.candidates[segments[3]!]
          : evidence!.steps[segments[3]!];
    if (segments[2] === "charts") payload = evidence!.charts[segments[3]!];
    await route.fulfill({
      status: payload ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(payload ?? {}),
    });
  });
});

test("shows method references from actual calibration training with clickable official sources", async ({
  page,
}) => {
  const step = Object.values(evidence!.steps).find(
    (item) => item.step_key === "preprocessing" && item.facts.method_knowledge,
  );
  test.skip(
    !step,
    "The fixture must include a retrieval-guided real training attempt.",
  );
  const run = Object.values(evidence!.summaries).find((item) =>
    item.attempts.some((attempt) => attempt.attempt_id === step!.attempt_id),
  )!;
  await page.setViewportSize({ width: 1560, height: 1000 });
  await page.goto(
    `/workspace/nir/process/${threadId}?run=${run.run_id}&attempt=${step!.attempt_id}&step=preprocessing`,
  );
  const card = page.getByTestId("method-knowledge");
  await expect(
    card.getByRole("heading", { name: "数据诊断与算法知识依据" }),
  ).toBeVisible();
  await expect(card).toContainText("诊断与检索只使用校准数据");
  await expect(card).toContainText("原始光谱保留为同批对照");
  await expect(
    card.getByRole("link", { name: "官方说明" }).first(),
  ).toHaveAttribute("href", /^https:\/\/chemotools\.org\/methods\//);
  const output = process.env.NIR_METHOD_QA_SCREENSHOT;
  if (output) {
    fs.mkdirSync(path.dirname(output), { recursive: true });
    await page.screenshot({ path: output, fullPage: true });
  }
});

test("explores actual computed spectra, candidates and linked diagnostics", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1560, height: 1000 });
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(
    page.getByRole("heading", { name: "建模过程探索" }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "验证与诊断", exact: true }),
  ).toBeVisible();
  const scatter = page.getByRole("img", {
    name: "实测值与预测值",
    exact: true,
  });
  await expect(scatter).toBeVisible();
  await scatter.locator("circle").first().focus();
  await page.keyboard.press("Enter");
  await expect(
    page.getByText("当前显示的所选点: 1", { exact: false }),
  ).toHaveCount(2);
  const residual = page.getByRole("img", { name: "残差诊断", exact: true });
  await expect(residual.locator('circle[fill="#d97706"]')).toHaveCount(1);
  await page
    .getByRole("navigation", { name: "建模步骤" })
    .getByRole("button", { name: /预处理/ })
    .click();
  await expect(
    page.getByRole("button", { name: "前后对照", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "光谱曲线", exact: true }),
  ).toHaveCount(2);
  await page.getByLabel("区间起点", { exact: true }).first().fill("25");
  await expect(page.getByLabel("区间起点", { exact: true }).nth(1)).toHaveValue(
    "25",
  );
  await page.getByRole("button", { name: "处理后", exact: true }).click();
  await expect(
    page.getByRole("img", { name: "光谱曲线", exact: true }),
  ).toHaveCount(1);
  await page
    .getByRole("navigation", { name: "建模步骤" })
    .getByRole("button", { name: /模型与调参比较/ })
    .click();
  await page.getByRole("button", { name: /pls.*调参集.*RMSE/i }).click();
  await expect(page.getByTestId("candidate-detail")).toBeVisible();
  const attempts = Object.values(evidence!.summaries)[0]!.attempts;
  if (attempts.length > 1) {
    await page.getByLabel("选择建模尝试").selectOption(attempts[0]!.attempt_id);
    await expect(page).toHaveURL(new RegExp(attempts[0]!.attempt_id));
    await expect(
      page.getByRole("heading", { name: "模型与调参比较", exact: true }),
    ).toBeVisible();
  }
  const output = process.env.NIR_PROCESS_QA_SCREENSHOT;
  if (output) {
    fs.mkdirSync(path.dirname(output), { recursive: true });
    await page.screenshot({ path: output, fullPage: true });
  }
});

test("keeps a slow previous step response from replacing the selected step", async ({
  page,
}) => {
  const audit = Object.values(evidence!.steps)
    .filter((step) => step.step_key === "audit")
    .at(-1)!;
  await page.route(`**/steps/${audit.step_execution_id}`, async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(audit),
    });
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const nav = page.getByRole("navigation", { name: "建模步骤" });
  await nav.getByRole("button", { name: /数据解释与审查/ }).click();
  await nav.getByRole("button", { name: /数据划分/ }).click();
  await expect(
    page.getByRole("heading", { name: "数据划分", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "数据解释与审查", exact: true }),
  ).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "数据划分", exact: true }),
  ).toBeVisible();
});

test("shows recorded Chinese pipeline names and keeps IDs in collapsed technical details", async ({
  page,
}) => {
  const step = Object.values(evidence!.steps).find(
    (item) => item.step_key === "preprocessing" && item.facts.method_knowledge,
  )!;
  const run = Object.values(evidence!.summaries).find((item) =>
    item.attempts.some((attempt) => attempt.attempt_id === step.attempt_id),
  )!;
  await page.setViewportSize({ width: 1560, height: 1050 });
  await page.goto(
    `/workspace/nir/process/${threadId}?run=${run.run_id}&attempt=${step.attempt_id}&step=preprocessing`,
  );
  const heading = page.getByRole("heading", {
    name: "候选比较 · 点击查看实际参数与依据",
    exact: true,
  });
  const comparison = heading.locator("..");
  const sg = comparison
    .getByRole("button", { name: /光谱平滑.*窗口：7/ })
    .first();
  await expect(sg).toBeVisible();
  await sg.click();
  const detail = page.getByTestId("candidate-detail");
  await expect(detail).toContainText("最佳潜变量数");
  await expect(detail).toContainText("交叉验证决定系数（R²）");
  await expect(detail).toContainText("训练内交叉验证");
  await expect(detail.getByText("cv_r2", { exact: true })).toHaveCount(0);
  const candidate = evidence!.candidates[step.step_execution_id]!.data[0]!;
  const trace = detail
    .getByTestId("process-technical-record")
    .filter({ hasText: candidate.candidate_id });
  await expect(trace.locator("pre")).not.toBeVisible();
  await trace.getByText("查看技术记录（含内部编号）", { exact: true }).click();
  await expect(trace.locator("pre")).toContainText(candidate.candidate_id);
  await expect(trace).toContainText("不代表排名或质量");
  await trace.getByText("查看技术记录（含内部编号）", { exact: true }).click();
  const width = await page.evaluate(
    () => document.documentElement.scrollWidth <= window.innerWidth,
  );
  expect(width).toBe(true);
  if (process.env.NIR_PROCESS_CHINESE_QA_SCREENSHOT)
    await page.screenshot({
      path: process.env.NIR_PROCESS_CHINESE_QA_SCREENSHOT,
      fullPage: true,
    });
});

test("supports compact desktop windows and honest empty process state", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1100, height: 900 });
  await page.goto(`/workspace/nir/process/${threadId}?step=preprocessing`);
  await expect(
    page.getByRole("heading", { name: "预处理", exact: true }),
  ).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > window.innerWidth,
  );
  expect(overflow).toBe(false);
  await page.route("**/nir-process/runs", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ data: [], next_offset: null }),
    }),
  );
  await page.goto(`/workspace/nir/process/old-chat`);
  await expect(page.getByText(/此会话没有新版过程证据/)).toBeVisible();
});

test("hides cached process evidence when access is revoked", async ({
  page,
}) => {
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(
    page.getByRole("img", { name: "实测值与预测值", exact: true }),
  ).toBeVisible();
  await page.route("**/nir-process/runs/*", (route) =>
    route.fulfill({
      status: 404,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Process evidence not found" }),
    }),
  );
  await page.getByRole("button", { name: "刷新进展" }).click();
  await expect(
    page.getByText("该运行不存在或没有访问权限。", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "实测值与预测值", exact: true }),
  ).toHaveCount(0);
});

test("measures 30 browser renders of bounded scale evidence", async ({
  page,
}) => {
  test.skip(
    !process.env.NIR_PROCESS_QA_PERFORMANCE,
    "Optional scale benchmark.",
  );
  test.setTimeout(180000);
  await page.setViewportSize({ width: 1560, height: 1000 });
  const destination = `/workspace/nir/process/${threadId}?step=audit`;
  await page.goto(destination);
  await expect(
    page.getByRole("img", { name: "光谱曲线", exact: true }),
  ).toBeVisible();
  const firstChart: number[] = [];
  const cachedSwitch: number[] = [];
  const hover: number[] = [];
  for (let index = 0; index < 30; index++) {
    let start = Date.now();
    await page.reload();
    const spectra = page.getByRole("img", { name: "光谱曲线", exact: true });
    await expect(spectra).toBeVisible();
    firstChart.push(Date.now() - start);
    const hoverElapsed = await spectra.evaluate(async (element) => {
      const rect = element.getBoundingClientRect();
      const started = performance.now();
      element.dispatchEvent(
        new PointerEvent("pointermove", {
          bubbles: true,
          clientX: rect.left + 140,
          clientY: rect.top + 100,
        }),
      );
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
      );
      return performance.now() - started;
    });
    await expect(page.getByText(/光谱数值:/).first()).toBeVisible();
    hover.push(hoverElapsed);
    const nav = page.getByRole("navigation", { name: "建模步骤" });
    await nav.getByRole("button", { name: /数据划分/ }).click();
    await expect(
      page.getByRole("img", { name: "目标值分布", exact: true }),
    ).toHaveCount(3);
    start = Date.now();
    await nav.getByRole("button", { name: /数据解释与审查/ }).click();
    await expect(
      page.getByRole("img", { name: "光谱曲线", exact: true }),
    ).toBeVisible();
    cachedSwitch.push(Date.now() - start);
  }
  const p95 = (values: number[]) =>
    [...values].sort((a, b) => a - b)[Math.ceil(values.length * 0.95) - 1];
  fs.writeFileSync(
    process.env.NIR_PROCESS_QA_PERFORMANCE!,
    JSON.stringify(
      {
        environment: await page.evaluate(() => ({
          userAgent: navigator.userAgent,
          cores: navigator.hardwareConcurrency,
        })),
        measurements: 30,
        mode: `${process.env.NIR_PROCESS_QA_SERVER_MODE ?? "Next development server"}, warm compilation, HTTP interception of exported computed evidence; browser measures exclude backend latency`,
        first_chart_p95_ms: p95(firstChart),
        cached_switch_p95_ms: p95(cachedSwitch),
        hover_p95_ms: p95(hover),
        firstChart,
        cachedSwitch,
        hover,
      },
      null,
      2,
    ),
  );
});
