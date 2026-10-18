import fs from "node:fs";

import { expect, test, type Page } from "@playwright/test";

import { mockLangGraphAPI } from "./utils/mock-api";

type Row = {
  decision_id: string;
  experiment_id: string;
  action: string;
  round_index: number;
};
type Evidence = {
  summary: Record<string, unknown> & { session_id: string };
  decisions: { data: Row[]; total: number };
  details: Record<string, unknown>;
  diagnostics: Record<string, unknown>;
  spectra: Record<string, unknown>;
  final_test: unknown;
  initial: {
    summary: Record<string, unknown>;
    decisions: { data: Row[]; total: number };
    details: Record<string, unknown>;
  };
};
const evidencePath = process.env.NIR_ROUND_QA_FIXTURE;
const evidence = evidencePath
  ? (JSON.parse(fs.readFileSync(evidencePath, "utf8")) as Evidence)
  : null;
test.skip(
  !evidence,
  "Set NIR_ROUND_QA_FIXTURE to the real training regression export.",
);
const threadId = "round-process-qa";

async function setup(page: Page, initial = false, delayedDecision?: string) {
  const requests: string[] = [];
  let live = initial;
  mockLangGraphAPI(page, { threads: [{ thread_id: threadId, values: {} }] });
  await page.route("**/api/threads/*/nir-process/**", async (route) => {
    const url = new URL(route.request().url());
    const resource = url.pathname.split("/nir-process/")[1]!;
    requests.push(resource);
    const segments = resource.split("/");
    const snapshot = live ? evidence!.initial : evidence!;
    let payload: unknown;
    if (resource === "autonomous")
      payload = { data: [{ session_id: evidence!.summary.session_id }] };
    else if (segments.length === 2) payload = snapshot.summary;
    else if (segments[2] === "decisions" && segments[3]) {
      if (segments[3] === delayedDecision)
        await new Promise((resolve) => setTimeout(resolve, 600));
      payload = snapshot.details[segments[3]];
    } else if (segments[2] === "decisions")
      payload = { ...snapshot.decisions, next_offset: null };
    else if (segments[2] === "diagnostics")
      payload = evidence!.diagnostics[segments[3]!];
    else if (segments[2] === "spectra")
      payload = evidence!.spectra[segments[3]!];
    else if (segments[2] === "charts") payload = evidence!.final_test;
    await route.fulfill({
      status: payload ? 200 : 404,
      contentType: "application/json",
      body: JSON.stringify(payload ?? {}),
    });
  });
  return {
    requests,
    finish: () => {
      live = false;
    },
  };
}

test.beforeEach(async ({ context }) => {
  await context.addCookies([
    { name: "locale", value: "zh-CN", domain: "localhost", path: "/" },
  ]);
});

test.afterEach(async ({ page }, info) => {
  if (info.status === "passed")
    await page.screenshot({
      path: info.outputPath("round-process.png"),
      fullPage: true,
    });
});

test("real rounds expose executed parameters, retention links and refreshable selection", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1100 });
  const { requests } = await setup(page);
  const [first, second, stop] = evidence!.decisions.data;
  await page.goto(`/workspace/nir/process/${threadId}`);
  const explorer = page.getByTestId("autonomous-round-explorer");
  await expect(explorer.getByTestId("round-node")).toHaveCount(3);
  await explorer.locator(`[data-decision-id="${second!.decision_id}"]`).click();
  const detail = page.getByTestId("round-detail");
  await detail.getByRole("tab", { name: "参数" }).click();
  await expect(detail).toContainText("实际执行配置");
  await expect(detail).toContainText("3");
  await detail.getByRole("tab", { name: "诊断图", exact: true }).click();
  await detail.getByRole("button", { name: "参数搜索", exact: true }).click();
  await expect(detail.getByRole("table")).toContainText("3");
  await detail.getByRole("button", { name: "光谱对照", exact: true }).click();
  await expect(detail).toContainText("Cal 原始与预处理光谱");
  await expect(detail.locator("svg[aria-label]")).toHaveCount(2);
  await explorer
    .getByRole("button", { name: "方案关系图", exact: true })
    .click();
  await expect(
    page.getByTestId("round-graph").locator(".react-flow__edge"),
  ).toHaveCount(3);
  await page
    .getByTestId("round-graph")
    .locator(`[data-decision-id="${stop!.decision_id}"]`)
    .click();
  await expect(detail).toHaveAttribute("data-decision-id", stop!.decision_id);
  await expect(detail).toContainText("主动停止");
  await page.reload();
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    stop!.decision_id,
  );
  await expect(
    page.getByTestId("round-graph").getByTestId("round-node"),
  ).toHaveCount(3);
  expect(requests.some((resource) => resource.includes("/charts/"))).toBe(
    false,
  );
  expect(
    requests.some((resource) =>
      resource.includes(`/diagnostics/${first!.experiment_id}`),
    ),
  ).toBe(true);
});

test("scatter and residual selection stay linked and stale responses cannot replace a selected round", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const [first, second] = evidence!.decisions.data;
  await setup(page, false, second!.decision_id);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const nav = page.getByRole("navigation", { name: "建模轮次节点" });
  await nav.locator(`[data-decision-id="${second!.decision_id}"]`).click();
  await nav.locator(`[data-decision-id="${first!.decision_id}"]`).click();
  const detail = page.getByTestId("round-detail");
  await expect(detail).toHaveAttribute("data-decision-id", first!.decision_id);
  await expect(detail.getByTestId("round-linked-plots")).toBeVisible();
  const charts = detail.getByTestId("round-linked-plots");
  await charts.locator("svg circle").first().click();
  await expect(charts).toContainText("当前显示的所选点");
  await detail.getByLabel("评价范围").selectOption("calibration_cv");
  await expect(detail).toContainText("校准交叉验证");
  await page.waitForTimeout(700);
  await expect(detail).toHaveAttribute("data-decision-id", first!.decision_id);
  await expect(
    detail.locator('[data-testid="round-linked-plots"]'),
  ).toBeVisible();
  await page.setViewportSize({ width: 1100, height: 900 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
});

test("the next recommendation exposes the model's pre-execution plan and links to its execution", async ({
  page,
}) => {
  await setup(page);
  const [first, second] = evidence!.decisions.data;
  await page.goto(`/workspace/nir/process/${threadId}`);
  await page
    .getByRole("navigation", { name: "建模轮次节点" })
    .locator(`[data-decision-id="${first!.decision_id}"]`)
    .click();
  await page
    .getByTestId("round-detail")
    .getByRole("tab", { name: "LLM 建议", exact: true })
    .click();
  const recommendation = page.getByTestId("round-next-recommendation");
  await expect(recommendation).toContainText("大模型的下一轮建议");
  await expect(recommendation).toContainText("建议方案");
  await expect(recommendation).toContainText("建议模型参数");
  await expect(recommendation).toContainText("参数搜索计划");
  await expect(recommendation).toContainText("预期改善");
  await expect(recommendation).toContainText("后续执行状态");
  await recommendation
    .getByRole("button", { name: "查看下一轮执行记录" })
    .click();
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    second!.decision_id,
  );
  await expect(recommendation).toContainText("主动停止");
  await expect(recommendation).not.toContainText("建议模型参数");
});

test("new evidence updates nodes without stealing a manually selected historical round", async ({
  page,
}) => {
  const state = await setup(page, true);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const explorer = page.getByTestId("autonomous-round-explorer");
  await expect(explorer.getByTestId("round-node")).toHaveCount(1);
  const first = evidence!.decisions.data[0]!;
  await explorer.getByTestId("round-node").click();
  state.finish();
  await explorer.getByRole("button", { name: "刷新轮次", exact: true }).click();
  await expect(explorer.getByTestId("round-node")).toHaveCount(3);
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    first.decision_id,
  );
});

test("a historical missing chart preserves metrics and does not trigger a modeling request", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/nir-process/autonomous/*/diagnostics/*", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ evidence_status: "not_recorded" }),
    }),
  );
  const mutations: string[] = [];
  page.on("request", (request) => {
    // The sidebar's read-only thread search uses POST.
    if (
      request.method() !== "GET" &&
      !new URL(request.url()).pathname.endsWith("/threads/search")
    )
      mutations.push(request.method());
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const detail = page.getByTestId("round-detail");
  await expect(detail).toContainText("本轮当时未记录可用的诊断图");
  await expect(detail).toContainText("RMSE");
  await detail.getByRole("button", { name: "重新读取图表" }).click();
  await expect(detail).toContainText("本轮当时未记录可用的诊断图");
  expect(mutations.filter((method) => method !== "HEAD")).toEqual([]);
});

test("a persisted same-session question displays a waiting state without resuming execution", async ({
  page,
}) => {
  await setup(page, true);
  await page.route(`**/api/threads/${threadId}/state`, (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        values: {
          nir_workflow: {
            project_id: "delivery",
            task_type: "calibration",
            stage: "execution",
            revision: 3,
            search_session_id: evidence!.summary.session_id,
            user_question: {
              status: "waiting",
              question: "请选择：1. 保留当前方案；2. 补充数据。",
            },
          },
        },
      }),
    }),
  );
  const mutations: string[] = [];
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      !new URL(request.url()).pathname.endsWith("/threads/search")
    )
      mutations.push(request.url());
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const explorer = page.getByTestId("autonomous-round-explorer");
  await expect(explorer).toContainText("等待你的回复");
  await expect(explorer).toContainText(
    "请选择：1. 保留当前方案；2. 补充数据。",
  );
  await explorer.getByRole("button", { name: "刷新轮次" }).click();
  await expect(explorer).toContainText("等待你的回复");
  expect(mutations).toEqual([]);
});

test("full lifecycle panes stay aligned and final Test remains lazy while chart selection and enlargement work", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const { requests } = await setup(page);
  const mutations: string[] = [];
  page.on("request", (request) => {
    if (
      !["GET", "HEAD"].includes(request.method()) &&
      !new URL(request.url()).pathname.endsWith("/threads/search")
    )
      mutations.push(request.url());
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const explorer = page.getByTestId("autonomous-round-explorer");
  await expect(explorer.getByTestId("lifecycle-node")).toHaveCount(8);
  await expect(explorer.getByTestId("round-node")).toHaveCount(3);
  const first = evidence!.decisions.data[0]!;
  await explorer
    .locator(
      `[data-testid="round-node"][data-decision-id="${first.decision_id}"]`,
    )
    .click();
  const detail = page.getByTestId("round-detail");
  await expect(detail.getByTestId("round-linked-plots")).toBeVisible();
  const panes = explorer.locator(".process-workbench");
  const sizes = await panes.evaluate((node) => {
    const left = node
      .querySelector(".process-node-pane")!
      .getBoundingClientRect();
    const right = node
      .querySelector(".process-detail-pane")!
      .getBoundingClientRect();
    return {
      top: Math.abs(left.top - right.top),
      bottom: Math.abs(left.bottom - right.bottom),
    };
  });
  expect(sizes.top).toBeLessThan(1);
  expect(sizes.bottom).toBeLessThan(1);
  await panes.screenshot({ path: "../.tmp/ui-redesign/process-workbench.png" });
  const selected = detail.getByLabel("选中样本", { exact: true });
  const value = await selected
    .first()
    .locator("option")
    .nth(1)
    .getAttribute("value");
  await selected.first().selectOption(value);
  await expect(selected.nth(1)).toHaveValue(value!);
  await detail
    .getByRole("button", { name: "放大：实测值与预测值", exact: true })
    .click();
  await expect(
    page
      .getByRole("dialog")
      .getByRole("img", { name: "实测值与预测值", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click();
  await detail.getByRole("button", { name: "光谱对照", exact: true }).click();
  await expect(
    detail.getByRole("img", { name: "光谱曲线", exact: true }),
  ).toHaveCount(2);
  await detail.getByRole("tab", { name: "LLM 建议", exact: true }).click();
  await expect(detail.getByTestId("round-next-recommendation")).toContainText(
    "建议模型参数",
  );
  const after = await panes.evaluate((node) =>
    Array.from(node.children).map(
      (child) => child.getBoundingClientRect().height,
    ),
  );
  expect(after[0]).toBe(after[1]);
  expect(
    requests.some((resource) => resource.includes("/charts/final_test")),
  ).toBe(false);
  await explorer
    .locator('[data-testid="lifecycle-node"][data-node-id="prepare"]')
    .click();
  const lifecycle = page.getByTestId("lifecycle-detail");
  await lifecycle.getByRole("tab", { name: "诊断图", exact: true }).click();
  await expect(
    lifecycle.getByRole("img", { name: "样本用途数量", exact: true }),
  ).toBeVisible();
  await explorer
    .locator('[data-testid="lifecycle-node"][data-node-id="test"]')
    .click();
  await expect(
    lifecycle.getByRole("img", { name: "实测值与预测值", exact: true }),
  ).toBeVisible();
  expect(
    requests.some((resource) => resource.includes("/charts/final_test")),
  ).toBe(true);
  await page.screenshot({
    path: "../.tmp/ui-redesign/process-final-test.png",
    fullPage: true,
  });
  expect(mutations).toEqual([]);
});

test("lifecycle selection and explicit detail tabs survive reload and compact screens stay contained", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await page
    .getByTestId("autonomous-round-explorer")
    .locator('[data-node-id="approval"]')
    .click();
  await page
    .getByTestId("lifecycle-detail")
    .getByRole("tab", { name: "LLM 建议", exact: true })
    .click();
  await page.reload();
  const detail = page.getByTestId("lifecycle-detail");
  await expect(detail).toHaveAttribute("data-node-id", "approval");
  await expect(
    detail.getByRole("tab", { name: "LLM 建议", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await detail
    .getByRole("tab", { name: "LLM 建议", exact: true })
    .press("Home");
  await expect(
    detail.getByRole("tab", { name: "概览", exact: true }),
  ).toBeFocused();
  for (const width of [1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBe(true);
  }
  await page.screenshot({
    path: "../.tmp/ui-redesign/process-narrow.png",
    fullPage: true,
  });
});

test("dark process panes keep visible focus, enlarged charts and contained compact layouts", async ({
  page,
}) => {
  await page.addInitScript(() => localStorage.setItem("theme", "dark"));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(page.locator("html")).toHaveClass(/dark/);
  const detail = page.getByTestId("round-detail");
  await expect(detail.getByTestId("round-linked-plots")).toBeVisible();
  await detail
    .getByRole("tab", { name: "诊断图", exact: true })
    .press("ArrowRight");
  await expect(
    detail.getByRole("tab", { name: "LLM 建议", exact: true }),
  ).toBeFocused();
  await detail.getByRole("tab", { name: "诊断图", exact: true }).click();
  await page
    .getByTestId("process-workbench")
    .screenshot({ path: "../.tmp/ui-redesign/process-dark.png" });
  await page.setViewportSize({ width: 720, height: 450 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
  await expect(
    detail.getByRole("tab", { name: "参数", exact: true }),
  ).toBeVisible();
});
