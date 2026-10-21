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
  const mocks = mockLangGraphAPI(page, {
    locale: "zh-CN",
    threads: [{ thread_id: threadId, values: {} }],
  });
  await mocks.ready;
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
    else if (segments[2] === "rejections")
      payload = { data: [], next_offset: null };
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

test.beforeEach(async ({ context, baseURL }) => {
  await context.addCookies([{ name: "locale", value: "zh-CN", url: baseURL! }]);
});

test.afterEach(async ({ page }, info) => {
  if (info.status === "passed")
    await page.screenshot({
      path: info.outputPath("round-process.png"),
      fullPage: true,
    });
});

const stageButton = (page: Page, name: string) =>
  page
    .getByRole("navigation", { name: "建模阶段" })
    .getByRole("button", { name: new RegExp(name) });
const roundButton = (page: Page, id: string) =>
  page.locator(`[data-testid="round-node"][data-decision-id="${id}"]`);
const roundRow = (page: Page, id: string) =>
  page.locator(`[data-testid="pipeline-row"][data-decision-id="${id}"]`);
async function selectRound(page: Page, index: number) {
  const id = evidence!.decisions.data[index]!.decision_id;
  await page
    .locator(`[data-testid="round-nav-node"][data-decision-id="${id}"]`)
    .click();
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    id,
  );
  return id;
}

test("left vertical navigation selects exactly one round beside compact chart previews", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const sidebar = page.getByTestId("process-round-sidebar");
  await expect(sidebar.getByTestId("round-nav-node")).toHaveCount(3);
  await expect(page.getByTestId("pipeline-row")).toHaveCount(1);
  const side = await sidebar.boundingBox(),
    content = await page.getByTestId("process-selected-content").boundingBox();
  expect(side!.width).toBeLessThanOrEqual(240);
  expect(side!.x + side!.width).toBeLessThan(content!.x);
  const headers = await sidebar
    .locator(".process-stage > button")
    .evaluateAll((items) =>
      items.map((item) => item.getBoundingClientRect().y),
    );
  expect(headers).toHaveLength(4);
  expect(headers.every((y, i) => i === 0 || y > headers[i - 1]!)).toBe(true);
  await expect(page.getByTestId("round-trend-disclosure")).not.toHaveAttribute(
    "open",
  );
  expect(requests.some((p) => p.includes("/charts/"))).toBe(false);
  await selectRound(page, 1);
  const row = roundRow(page, evidence!.decisions.data[1]!.decision_id);
  await expect(row).toContainText("参考第 1 轮");
  await expect(row).toContainText("调参");
  await row.locator('[data-step-id="preprocess-0"]').click();
  const plots = page
    .getByTestId("round-detail")
    .getByRole("img", { name: "光谱曲线", exact: true });
  await expect(plots).toHaveCount(1);
  const box = await plots.boundingBox();
  expect(box!.height).toBeGreaterThanOrEqual(180);
  expect(box!.height).toBeLessThanOrEqual(220);
  await expect(page.getByTestId("round-detail").getByRole("tab")).toHaveCount(
    0,
  );
  await selectRound(page, 0);
  await expect(page.getByTestId("pipeline-row")).toHaveCount(1);
  await expect(
    roundRow(page, evidence!.decisions.data[0]!.decision_id),
  ).toContainText("原始光谱 · 无预处理");
  await roundButton(page, evidence!.decisions.data[0]!.decision_id).click();
  await expect(page.getByTestId("round-detail")).toHaveCount(0);
  await selectRound(page, 0);
  await expect(page.getByTestId("round-detail")).toBeVisible();
});

test("step selection opens exact fitted parameters and survives reload", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const second = evidence!.decisions.data[1]!;
  await selectRound(page, 1);
  await roundRow(page, second.decision_id)
    .locator('[data-step-id="model"]')
    .click();
  const detail = page.getByTestId("round-detail");
  await expect(detail).toHaveAttribute("data-focus-step", "model");
  await detail.getByText("查看实际模型参数", { exact: true }).click();
  await expect(detail.getByTestId("round-parameters").first()).toContainText(
    "alpha",
  );
  await expect(page).toHaveURL(/nir_step=model/);
  await page.reload();
  await expect(detail).toHaveAttribute("data-focus-step", "model");
  await detail
    .getByTestId("model-parameter-search")
    .locator(":scope > summary")
    .click();
  await expect(detail.getByRole("table")).toContainText("3");
  expect(requests.some((p) => p.includes("/charts/"))).toBe(false);
});

test("every cumulative step has a plot, linked zoom and samples, and before-after comparison", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const id = await selectRound(page, 1);
  const row = roundRow(page, id),
    detail = page.getByTestId("round-detail");
  for (const step of ["raw", "preprocess-0", "preprocess-1"]) {
    await row.locator(`[data-step-id="${step}"]`).click();
    await expect(detail.getByTestId("step-spectra")).toHaveCount(1);
    await expect(
      detail.getByRole("img", { name: "光谱曲线", exact: true }),
    ).toHaveCount(1);
  }
  await expect(
    detail.getByRole("group", { name: "图表分组", exact: true }),
  ).toHaveCount(0);
  await detail.getByText("样本显隐", { exact: true }).click();
  const sample = detail.getByRole("button", { name: "样本 1", exact: true });
  await sample.click();
  await expect(sample).toHaveAttribute("aria-pressed", "false");
  await detail.getByLabel("区间起点", { exact: true }).fill("20");
  await detail
    .getByLabel("聚焦样本", { exact: true })
    .selectOption({ label: "样本 2" });
  await expect(detail.locator('svg polyline[stroke-width="3"]')).toHaveCount(1);
  const selectedSample = await detail
    .getByLabel("聚焦样本", { exact: true })
    .inputValue();
  await row.locator('[data-step-id="preprocess-0"]').click();
  await expect(detail.getByLabel("区间起点", { exact: true })).toHaveValue(
    "20",
  );
  await expect(detail.getByLabel("聚焦样本", { exact: true })).toHaveValue(
    selectedSample,
  );
  await row.locator('[data-step-id="preprocess-1"]').click();
  await detail.getByLabel("统一纵轴范围").check();
  await detail.getByRole("button", { name: "对照上一步", exact: true }).click();
  await expect(
    page
      .getByTestId("step-spectra-comparison")
      .getByRole("img", { name: "光谱曲线", exact: true }),
  ).toHaveCount(2);
  await expect(page.getByRole("dialog")).toContainText("上一步输出");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click();
});

test("legacy caches keep raw and complete output but never invent the middle step", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/nir-process/autonomous/*/spectra/*", (route) => {
    const id = new URL(route.request().url()).pathname.split("/").at(-1)!;
    const cache = structuredClone(evidence!.spectra[id]) as Record<
      string,
      unknown
    >;
    cache.schema_version = 1;
    delete cache.steps;
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(cache),
    });
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  await selectRound(page, 1);
  const row = roundRow(page, evidence!.decisions.data[1]!.decision_id),
    detail = page.getByTestId("round-detail");
  await row.locator('[data-step-id="preprocess-0"]').click();
  await expect(
    detail.locator('[data-spectra-step="preprocess-0"]'),
  ).toContainText("该次运行未保存此步骤光谱");
  await expect(
    detail.getByRole("img", { name: "光谱曲线", exact: true }),
  ).toHaveCount(0);
  for (const step of ["preprocess-1", "raw"]) {
    await row.locator(`[data-step-id="${step}"]`).click();
    await expect(
      detail.getByRole("img", { name: "光谱曲线", exact: true }),
    ).toHaveCount(1);
  }
});

test("trend point keyboard selection, metric switch and diagnostic sample links work", async ({
  page,
}) => {
  await setup(page, false, evidence!.decisions.data[1]!.decision_id);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await selectRound(page, 1);
  await page.getByTestId("round-trend-disclosure").locator("summary").click();
  const trend = page.getByTestId("optimization-trend");
  await trend.getByRole("button", { name: /查看第 1 轮/ }).press("Enter");
  const detail = page.getByTestId("round-detail");
  await expect(detail).toHaveAttribute(
    "data-decision-id",
    evidence!.decisions.data[0]!.decision_id,
  );
  const charts = detail.getByTestId("round-linked-plots");
  await expect(charts).toBeVisible();
  await charts.locator("svg circle").first().click();
  await expect(charts).toContainText("当前显示的所选点");
  await detail.getByLabel("评价范围").selectOption("calibration_cv");
  await expect(detail).toContainText("校准交叉验证");
  await trend.getByLabel("优化趋势指标").selectOption("cal_cv:RMSE");
  await expect(
    trend.getByRole("group", { name: "Cal CV RMSE", exact: true }),
  ).toBeVisible();
  await page.waitForTimeout(700);
  await expect(detail).toHaveAttribute(
    "data-decision-id",
    evidence!.decisions.data[0]!.decision_id,
  );
});

test("recommendations have one location and the recorded successor remains linked", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const id = await selectRound(page, 0),
    detail = page.getByTestId("round-detail");
  for (const step of ["raw", "model", "validation"]) {
    await roundRow(page, id).locator(`[data-step-id="${step}"]`).click();
    await expect(page.getByTestId("round-next-recommendation")).toHaveCount(0);
  }
  const decisionNode = () =>
    page
      .getByTestId("round-pipeline-strip")
      .locator('[data-step-id="decision"]');
  await decisionNode().click();
  await expect(detail).toHaveAttribute("data-focus-step", "decision");
  await expect(decisionNode()).toHaveAttribute("aria-pressed", "true");
  await expect(page).toHaveURL(/nir_step=decision/);
  await page.reload();
  await expect(decisionNode()).toHaveAttribute("aria-pressed", "true");
  const advice = page.getByTestId("round-next-recommendation");
  await expect(advice).toHaveCount(1);
  await expect(advice).toContainText("后续行动");
  await expect(advice).toContainText("执行状态");
  await expect(advice).not.toContainText("建议模型参数");
  await advice.getByRole("button", { name: "查看下一轮执行记录" }).click();
  await expect(detail).toHaveAttribute(
    "data-decision-id",
    evidence!.decisions.data[1]!.decision_id,
  );
  await expect(detail).toHaveAttribute("data-focus-step", "validation");
  await decisionNode().click();
  await expect(advice).toContainText("主动停止");
  await stageButton(page, "交付审批").click();
  await expect(advice).toHaveCount(0);
});

test("data stage previews original saved Cal evidence and preserves lifecycle links", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await stageButton(page, "数据准备").click();
  await expect(page.getByTestId("data-nav-node")).toHaveCount(3);
  await expect(page.getByTestId("lifecycle-node")).toHaveCount(0);
  const detail = page.getByTestId("lifecycle-detail");
  await expect(detail.getByTestId("process-data-overview")).toContainText("36");
  await expect(detail).toContainText("来源记录未提供");
  await detail.getByRole("tab", { name: "原始光谱", exact: true }).click();
  await expect(detail.getByTestId("cal-spectra")).toContainText(
    "原始光谱 · Cal",
  );
  await page.reload();
  await expect(detail).toHaveAttribute("data-node-id", "input");
  await expect(detail.getByTestId("cal-spectra")).toBeVisible();
  await page
    .locator('[data-testid="data-nav-node"][data-node-id="prepare"]')
    .click();
  await detail.getByRole("tab", { name: "分区图", exact: true }).click();
  await expect(
    detail.getByRole("img", { name: "样本用途数量", exact: true }),
  ).toBeVisible();
  expect(requests.some((p) => p.includes("/charts/"))).toBe(false);
});

test("final Test stays lazy until final stage and delivery remains separate", async ({
  page,
}) => {
  const { requests } = await setup(page);
  const mutations: string[] = [];
  page.on("request", (r) => {
    if (
      !["GET", "HEAD"].includes(r.method()) &&
      !new URL(r.url()).pathname.endsWith("/threads/search")
    )
      mutations.push(r.url());
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(page.getByTestId("pipeline-row")).toHaveCount(1);
  expect(requests.some((p) => p.includes("/charts/"))).toBe(false);
  await stageButton(page, "最终评价").click();
  await expect(
    page.getByRole("heading", { name: "最终保留模型", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: "实测值与预测值", exact: true }),
  ).toBeVisible();
  expect(requests.some((p) => p.includes("/charts/final_test"))).toBe(true);
  await stageButton(page, "交付审批").click();
  await expect(
    page.getByRole("heading", { name: "交付、审批与入库", exact: true }),
  ).toBeVisible();
  await expect(page.getByTestId("lifecycle-node")).toHaveCount(3);
  expect(mutations).toEqual([]);
});

test("stop event is never shown as another fitted pipeline and restores selection", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  const stop = evidence!.decisions.data[2]!;
  await selectRound(page, 2);
  const detail = page.getByTestId("round-detail");
  await expect(detail).toContainText("主动停止");
  await expect(roundRow(page, stop.decision_id)).toHaveCount(0);
  await page.reload();
  await expect(detail).toHaveAttribute("data-decision-id", stop.decision_id);
});

test("new evidence never steals a manually selected historical round", async ({
  page,
}) => {
  const state = await setup(page, true);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(page.getByTestId("round-nav-node")).toHaveCount(1);
  const first = evidence!.decisions.data[0]!;
  await roundRow(page, first.decision_id)
    .locator('[data-step-id="model"]')
    .click();
  state.finish();
  await page.getByRole("button", { name: "刷新轮次", exact: true }).click();
  await expect(page.getByTestId("round-nav-node")).toHaveCount(3);
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    first.decision_id,
  );
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-focus-step",
    "model",
  );
});

test("missing historical diagnostics stay missing and never trigger training", async ({
  page,
}) => {
  await setup(page);
  await page.route("**/nir-process/autonomous/*/diagnostics/*", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ evidence_status: "not_recorded" }),
    }),
  );
  const mutations: string[] = [];
  page.on("request", (r) => {
    if (
      !["GET", "HEAD"].includes(r.method()) &&
      !new URL(r.url()).pathname.endsWith("/threads/search")
    )
      mutations.push(r.url());
  });
  await page.goto(`/workspace/nir/process/${threadId}`);
  await selectRound(page, 0);
  const detail = page.getByTestId("round-detail");
  await expect(detail).toContainText("本轮当时未记录可用的诊断图");
  await detail.getByRole("button", { name: "重新读取图表" }).click();
  await expect(detail).toContainText("本轮当时未记录可用的诊断图");
  expect(mutations).toEqual([]);
});

test("persisted waiting question never resumes modeling", async ({ page }) => {
  await setup(page, true);
  await page.route(`**/api/threads/${threadId}/state`, (r) =>
    r.fulfill({
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
  await page.goto(`/workspace/nir/process/${threadId}`);
  const explorer = page.getByTestId("autonomous-round-explorer");
  await expect(explorer).toContainText(
    "请选择：1. 保留当前方案；2. 补充数据。",
  );
  await explorer.getByRole("button", { name: "刷新轮次" }).click();
  await expect(explorer).toContainText("等待你的回复");
});

test("stage tabs restore explicit lifecycle detail and compact screens stay contained", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await stageButton(page, "交付审批").click();
  await page
    .locator('[data-testid="lifecycle-node"][data-node-id="approval"]')
    .click();
  const detail = page.getByTestId("lifecycle-detail");
  await detail.getByRole("tab", { name: "记录详情", exact: true }).click();
  await page.reload();
  await expect(detail).toHaveAttribute("data-node-id", "approval");
  await expect(
    detail.getByRole("tab", { name: "记录详情", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await detail
    .getByRole("tab", { name: "记录详情", exact: true })
    .press("Home");
  await expect(
    detail.getByRole("tab", { name: "阶段概览", exact: true }),
  ).toBeFocused();
  await stageButton(page, "逐轮建模").click();
  for (const width of [1024, 768, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth + 1,
      ),
    ).toBe(true);
    await expect(page.getByTestId("pipeline-row").first()).toBeVisible();
  }
});

test("dark pipeline view keeps chart enlargement and keyboard navigation", async ({
  page,
}) => {
  await page.addInitScript(() => localStorage.setItem("theme", "dark"));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(page.locator("html")).toHaveClass(/dark/);
  await selectRound(page, 0);
  const detail = page.getByTestId("round-detail");
  await expect(detail.getByTestId("round-linked-plots")).toBeVisible();
  const strip = page.getByTestId("round-pipeline-strip");
  await strip.locator('[data-step-id="validation"]').focus();
  await strip.locator('[data-step-id="validation"]').press("Tab");
  await expect(strip.locator('[data-step-id="decision"]')).toBeFocused();
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
  await page.setViewportSize({ width: 720, height: 450 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth + 1,
    ),
  ).toBe(true);
});
test("empty rejection history is hidden and actual rejected proposals remain inspectable", async ({
  page,
}) => {
  await setup(page);
  await page.goto(`/workspace/nir/process/${threadId}`);
  await expect(page.getByTestId("pipeline-row")).toHaveCount(1);
  await expect(
    page.getByText("提案校验记录（未执行的方案）", { exact: true }),
  ).toHaveCount(0);
  await page.route("**/nir-process/autonomous/*/rejections?*", (r) =>
    r.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        data: [
          {
            errors: ["模型参数范围不合法，修正后才可执行。"],
            model_executed: false,
          },
        ],
        next_offset: null,
      }),
    }),
  );
  await page.reload();
  await page.getByText("提案校验记录（未执行的方案）", { exact: true }).click();
  await expect(
    page.getByText("模型参数范围不合法，修正后才可执行。", { exact: true }),
  ).toBeVisible();
});

test("recorded claim objects display readable observations without exposing JSON by default", async ({
  page,
}) => {
  await setup(page);
  const second = evidence!.decisions.data[1]!;
  await page.route(
    `**/nir-process/autonomous/*/decisions/${second.decision_id}`,
    (route) => {
      const detail = structuredClone(
        evidence!.details[second.decision_id],
      ) as Record<string, unknown>;
      const proposal = (detail.proposal ?? {}) as Record<string, unknown>;
      proposal.observations = [
        {
          claim: "记录的观察：本轮需要比较处理后的验证误差。",
          evidence_refs: ["profile:calibration"],
        },
        { evidence_refs: ["internal_only"] },
      ];
      detail.proposal = proposal;
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detail),
      });
    },
  );
  await page.goto(`/workspace/nir/process/${threadId}`);
  const id = await selectRound(page, 1);
  await roundRow(page, id).locator('[data-step-id="decision"]').click();
  const detail = page.getByTestId("round-detail");
  await expect(detail.getByRole("tab")).toHaveCount(0);
  await detail
    .getByTestId("recorded-pre-execution-reasons")
    .locator(":scope > summary")
    .click();
  await expect(
    detail.getByText("记录的观察：本轮需要比较处理后的验证误差。", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    detail.getByText("查看技术记录（含内部编号）", { exact: true }),
  ).toHaveCount(0);
  await expect(detail.getByTestId("round-run-record")).not.toHaveAttribute(
    "open",
  );
  await expect(
    detail.getByTestId("round-run-record").locator("pre"),
  ).toBeHidden();
  await expect(detail.getByText("internal_only", { exact: true })).toHaveCount(
    0,
  );
});

test("mismatched comparison evidence does not claim an improvement", async ({
  page,
}) => {
  await setup(page);
  const second = evidence!.decisions.data[1]!;
  await page.route(
    `**/nir-process/autonomous/*/decisions/${second.decision_id}`,
    (route) => {
      const detail = structuredClone(
        evidence!.details[second.decision_id],
      ) as Record<string, unknown>;
      const feedback = detail.feedback as Record<string, unknown>;
      feedback.cv_settings = { manifest_id: "different_split" };
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(detail),
      });
    },
  );
  await page.goto(`/workspace/nir/process/${threadId}`);
  await selectRound(page, 1);
  const comparison = page.getByTestId("round-comparison");
  await expect(comparison).toContainText("不计算改善幅度");
  await expect(comparison.getByTestId("metric-change")).toHaveCount(0);
});

test("left stage pipeline expands data and round steps without losing the selected detail", async ({
  page,
}) => {
  const { requests } = await setup(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`/workspace/nir/process/${threadId}`);
  const data = stageButton(page, "数据准备"),
    search = stageButton(page, "逐轮建模");
  await expect(search).toHaveAttribute("aria-expanded", "true");
  await expect(data).toHaveAttribute("aria-expanded", "false");
  await data.click();
  await expect(data).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("data-nav-node")).toHaveCount(3);
  await expect(page.getByTestId("lifecycle-node")).toHaveCount(0);
  const inspect = page.locator(
    '[data-testid="data-nav-node"][data-node-id="inspect"]',
  );
  await inspect.click();
  const detail = page.getByTestId("lifecycle-detail");
  await expect(detail).toHaveAttribute("data-node-id", "inspect");
  await expect(inspect).toHaveAttribute("aria-pressed", "true");
  await page.reload();
  await expect(data).toHaveAttribute("aria-expanded", "true");
  await expect(detail).toHaveAttribute("data-node-id", "inspect");
  await data.click();
  await expect(data).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("data-nav-node")).toHaveCount(0);
  await expect(detail).toHaveAttribute("data-node-id", "inspect");
  await expect(page).toHaveURL(/nir_node=inspect/);
  await data.press("Enter");
  await expect(page.getByTestId("data-nav-node")).toHaveCount(3);
  await search.click();
  await expect(search).toHaveAttribute("aria-expanded", "true");
  const selected = await page
    .getByTestId("round-detail")
    .getAttribute("data-decision-id");
  await search.click();
  await expect(search).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("round-nav-node")).toHaveCount(0);
  await expect(page.getByTestId("round-detail")).toHaveAttribute(
    "data-decision-id",
    selected!,
  );
  await search.press("Enter");
  await expect(page.getByTestId("round-nav-node")).toHaveCount(3);
  await selectRound(page, 1);
  expect(requests.some((p) => p.includes("/charts/"))).toBe(false);
});
