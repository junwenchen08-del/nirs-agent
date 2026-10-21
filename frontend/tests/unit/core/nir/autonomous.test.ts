import { expect, test, rs } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/i18n/hooks", () => ({ useI18n: () => ({ locale: "zh-CN" }) }));
rs.mock("@/core/artifacts/utils", () => ({ urlOfArtifact: () => "/download" }));
rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));
rs.mock("@/core/nir/round-hooks", () => ({
  useRoundHistory: rs.fn(() => ({ rows: [], total: 0, refetch: rs.fn() })),
}));

import { AutonomousProcess } from "@/components/workspace/nir/autonomous-process";
import {
  actionLabel,
  autonomousDecisionPageSize,
  autonomousIntakeStatus,
  autonomousLifecycleFacts,
  autonomousPollingInterval,
  autonomousPublicationPollingLimitMs,
  autonomousTestStatusLabel,
  budgetSelection,
  executionLabel,
  experimentMarkers,
  isAutonomousTerminal,
  isCustomBudgetValid,
  metricRows,
  modelingChatContext,
  parameterTrials,
  parseDecision,
  parseDecisionPage,
  parseFinalChart,
  parseRejectionPage,
  parseSummary,
  stopExplanation,
  validationSeries,
} from "@/core/nir/autonomous";
import { useProcessResource } from "@/core/nir/process-hooks";
import { useRoundHistory } from "@/core/nir/round-hooks";

const summary = {
  schema_version: 1,
  session_id: "search_a",
  stage: "searching",
  test_status: "sealed",
  trends: [
    {
      experiment_id: "exp_a",
      round_index: 1,
      method: "ridge",
      metrics: { val: { R2: 0.6, RMSE: 2, scope: "validation" } },
    },
    {
      experiment_id: "exp_b",
      round_index: 3,
      method: "svr",
      metrics: { val: { R2: 0.8, RMSE: 1, scope: "validation" } },
    },
  ],
  retained_experiment_id: "exp_a",
  best_val_experiment_id: "exp_b",
};

test("unsealed intake distinguishes waiting, blocked and confirmed ingestion failure", () => {
  expect(autonomousIntakeStatus({ stage: "data_audit" }, true).failed).toBe(
    false,
  );
  expect(autonomousIntakeStatus({ stage: "blocked" }, true).message).toContain(
    "任务已被阻塞",
  );
  const state = {
    stage: "blocked",
    ingestion_evidence: {
      source_tool: "nir_prepare_search",
      status: "failed",
      code: "nir_ingestion_numeric_integrity",
      error: "C:\\private\\file",
    },
  };
  const result = autonomousIntakeStatus(state, true);
  expect(result.failed).toBe(true);
  expect(result.code).toBe("nir_ingestion_numeric_integrity");
  expect(result.message).toContain("数据摄取失败");
  expect(JSON.stringify(result)).not.toContain("private");
});
const decision = {
  decision_id: "d_a",
  status: "accepted",
  action: "EXPLORE",
  selected: { pipeline: { model_method: "svr", preprocessing: [] } },
  proposal: {
    rationale: "comparison",
    observations: [{ claim: "cal", evidence_refs: ["cal"] }],
  },
};

test("web mode is a separate structured request", () => {
  expect(budgetSelection("development")).toEqual({
    enabled: true,
    profile: "development",
  });
});
test("custom budget exposes only the four user controls", () => {
  expect(budgetSelection("custom", { max_fits: 8 }).budget).toEqual({
    max_fits: 8,
  });
});
test("Raw remains an empty preprocessing choice", () => {
  expect(parseDecision(decision)?.pipelineLabel).toBe("Raw → SVR");
});
test("absent or unnamed preprocessing never becomes Raw", () => {
  expect(
    parseDecision({
      ...decision,
      selected: { pipeline: { model_method: "svr" } },
    })?.pipelineLabel,
  ).toBe("— → SVR");
  expect(
    parseDecision({
      ...decision,
      selected: { pipeline: { model_method: "svr", preprocessing: [{}] } },
    })?.pipelineLabel,
  ).toBe("— → SVR");
});
test("preprocessing order is shown as recorded", () => {
  expect(
    parseDecision({
      ...decision,
      selected: {
        pipeline: {
          model_method: "pls",
          preprocessing: [{ method: "sg_smooth" }, { method: "snv" }],
        },
      },
    })?.pipelineLabel,
  ).toBe("sg_smooth → snv → PLS");
});
test("accepted proposal is not shown as trained", () => {
  expect(executionLabel("accepted", true)).toBe("已接受，待执行");
});
test("failed action has a distinct status", () => {
  expect(executionLabel("failed", true)).toBe("执行失败");
});
test("STOP and RETAIN are distinct actions", () => {
  expect(actionLabel("STOP", true)).not.toBe(actionLabel("RETAIN", true));
});
test("budget exhaustion does not pretend to be an LLM STOP", () => {
  expect(
    stopExplanation(
      { stop_source: "budget", retained_selection_source: "llm" },
      true,
    ),
  ).toContain("预算");
  expect(stopExplanation({ stop_source: "llm" }, true)).toContain("主动停止");
});
test("only validation scoped executed metrics enter the trend", () => {
  const parsed = parseSummary({
    ...summary,
    trends: [
      ...summary.trends,
      {
        experiment_id: "exp_bad",
        metrics: { val: { RMSE: 0.01, scope: "final_test" } },
      },
    ],
  })!;
  expect(validationSeries(parsed, "RMSE").map((x) => x.experiment_id)).toEqual([
    "exp_a",
    "exp_b",
  ]);
});
test("best validation and LLM retained identities stay independent", () => {
  const parsed = parseSummary(summary)!;
  expect(parsed.best_val_experiment_id).toBe("exp_b");
  expect(parsed.retained_experiment_id).toBe("exp_a");
});
test("finalized delivery and quality never imply user approval or registration", () => {
  const parsed = parseSummary({
    ...summary,
    stage: "finalized",
    quality: { passed: true },
    export_status: "succeeded",
  })!;
  expect(autonomousLifecycleFacts(parsed, undefined)).toEqual({
    quality: { passed: true },
    user_approval_status: "not_recorded_in_session",
    registered_model_id: null,
    lifecycle_at_delivery: {},
  });
  expect(
    autonomousLifecycleFacts(parsed, {
      search_session_id: "search_a",
      approval_status: "approved",
    }),
  ).toMatchObject({
    user_approval_status: "approved",
    registered_model_id: null,
  });
  expect(
    autonomousLifecycleFacts(parsed, {
      search_session_id: "search_a",
      approval_status: "pending",
      registered_model_id: "registered_1",
    }),
  ).toMatchObject({
    user_approval_status: "pending",
    registered_model_id: "registered_1",
  });
  expect(
    autonomousLifecycleFacts(parsed, {
      search_session_id: "foreign",
      approval_status: "approved",
      registered_model_id: "foreign_model",
    }),
  ).toMatchObject({
    user_approval_status: "not_recorded_in_session",
    registered_model_id: null,
  });
});
test("sealed Test metrics are never displayed", () => {
  expect(
    metricRows({
      ...summary,
      metrics: { test: { R2: 0.99, scope: "final_test" } },
    }),
  ).toEqual([]);
});
test("final Test metrics require finalized state and scope", () => {
  expect(
    metricRows({
      ...summary,
      stage: "finalized",
      metrics: { test: { R2: 0.9, scope: "final_test" } },
    })[0]?.label,
  ).toBe("Test");
});
test("empty and legacy evidence creates no experiments", () => {
  expect(parseSummary({ attempts: [] })).toBeNull();
  expect(
    validationSeries(parseSummary({ ...summary, trends: [] })!, "R2"),
  ).toEqual([]);
});
test("projection excludes paths and confidential extra fields", () => {
  const data = parseSummary({
    ...summary,
    private_path: "C:\\secret",
    api_key: "credential",
    downloads: [{ name: "unsafe", path: "../holdout.npz" }],
  })!;
  expect(JSON.stringify(data)).not.toContain("credential");
  expect(data.downloads).toEqual([]);
});
test("read-only reasons remain separate from actual feedback", () => {
  const data = parseDecision({
    ...decision,
    rationale: "original",
    feedback: {
      actual_parameters: { C: 2 },
      expected_improvement: { status: "not_achieved" },
    },
  })!;
  expect(data.rationale).toBe("original");
  expect(data.feedback?.expected_improvement).toEqual({
    status: "not_achieved",
  });
});
test("source audit distinguishes Fake from a real provider", () => {
  expect(
    parseDecision({ ...decision, origin: { verification_kind: "fake_model" } })
      ?.origin.verification_kind,
  ).toBe("fake_model");
});

test("all four metric ranges retain their recorded identities", () => {
  const rows = metricRows({
    stage: "finalized",
    metrics: {
      cal: { RMSE: 4, scope: "calibration" },
      cal_cv: { RMSE: 3, scope: "calibration_cv" },
      val: { RMSE: 2, scope: "validation" },
      test: { RMSE: 1, scope: "final_test" },
    },
  });
  expect(rows.map((row) => [row.label, row.values.RMSE])).toEqual([
    ["Cal", 4],
    ["Cal CV", 3],
    ["Val", 2],
    ["Test", 1],
  ]);
});

test("invalid metrics stay absent instead of becoming perfect zero scores", () => {
  const parsed = parseSummary({
    ...summary,
    metrics: {
      val: {
        scope: "validation",
        R2: NaN,
        RMSE: null,
        RPD: Infinity,
        n: "10",
        bias: 0,
      },
    },
    trends: [
      {
        experiment_id: "nan",
        round_index: 1,
        metrics: { val: { RMSE: NaN, scope: "validation" } },
      },
      {
        experiment_id: "string",
        round_index: 2,
        metrics: { val: { RMSE: "0.1", scope: "validation" } },
      },
      {
        experiment_id: "missing",
        metrics: { val: { RMSE: 0.1, scope: "validation" } },
      },
      ...summary.trends,
    ],
  })!;
  expect(parsed.metrics.val).toEqual({ scope: "validation", bias: 0 });
  expect(parsed.trends[2]?.round_index).toBeNull();
  expect(
    validationSeries(parsed, "RMSE").map((item) => item.experiment_id),
  ).toEqual(["exp_a", "exp_b"]);
  expect(parseDecision(decision)?.round_index).toBeNull();
});

test("non-finite budget and actual-fit facts never become zero", () => {
  expect(
    parseSummary({ ...summary, budget: { fits_used: NaN } })?.budget.fits_used,
  ).toBeNull();
  expect(
    parseDecision({ ...decision, feedback: { fits: Infinity } })?.feedback
      ?.fits,
  ).toBeNull();
});

test("failed and pending executions do not project performance metrics", () => {
  for (const status of ["failed", "accepted", "executing", "stopped"]) {
    expect(
      parseDecision({
        ...decision,
        status,
        feedback: {
          metrics: { val: { RMSE: 0.1, scope: "validation" } },
          fits: 2,
        },
      })?.feedback?.metrics,
    ).toEqual({});
  }
  expect(
    parseDecision({
      ...decision,
      status: "succeeded",
      feedback: { metrics: { val: { RMSE: 0.1, scope: "validation" } } },
    })?.feedback?.metrics?.val,
  ).toEqual({ RMSE: 0.1, scope: "validation" });
});

test("best Val and LLM retention render separate markers without reselecting", () => {
  const parsed = parseSummary({
    ...summary,
    retained_selection_source: "llm",
  })!;
  expect(experimentMarkers(parsed, "exp_a", true)).toEqual(["LLM 已保留"]);
  expect(experimentMarkers(parsed, "exp_b", true)).toEqual([
    "当前最佳 Val RMSE",
  ]);
  expect(parsed.retained_experiment_id).toBe("exp_a");
  expect(experimentMarkers(parseSummary(summary)!, "exp_a", true)).toEqual([
    "已记录保留",
  ]);
});

test("budget safety termination does not claim an LLM retained model", () => {
  expect(
    stopExplanation({ stage: "safety_closed", stop_source: "budget" }, true),
  ).toContain("尚未记录");
  expect(
    stopExplanation({ stage: "budget_exhausted", stop_source: "budget" }, true),
  ).toContain("等待 LLM");
});

test("real terminal stages stop polling but budget retention and finalization stay active", () => {
  for (const stage of [
    "finalized",
    "cancelled",
    "safety_closed",
    "provider_failed",
    "finalization_unavailable",
    "final_test_failed",
    "no_deliverable_model",
    "stopped_pending_confirmation",
    "decision_invalid",
  ])
    expect(isAutonomousTerminal(stage)).toBe(true);
  for (const stage of [
    "searching",
    "budget_exhausted",
    "stopped",
    "frozen",
    "testing",
  ])
    expect(isAutonomousTerminal(stage)).toBe(false);
});

test("final Test completion keeps bounded publication polling until export settles", () => {
  const pending = { stage: "finalized", autonomous_delivery: true };
  expect(autonomousPollingInterval(pending)).toBe(5000);
  expect(
    autonomousPollingInterval(
      { ...pending, export_status: "building" },
      autonomousPublicationPollingLimitMs - 1,
    ),
  ).toBe(5000);
  expect(
    autonomousPollingInterval(pending, autonomousPublicationPollingLimitMs),
  ).toBe(false);
  for (const export_status of ["succeeded", "failed"])
    expect(autonomousPollingInterval({ ...pending, export_status })).toBe(
      false,
    );
  expect(
    autonomousPollingInterval({
      stage: "finalized",
      autonomous_delivery: false,
    }),
  ).toBe(false);
  expect(autonomousPollingInterval({ stage: "searching" })).toBeNull();
});

test("long decision history is read as bounded ten-record pages beyond twenty rounds", () => {
  const decisions = Array.from({ length: 25 }, (_, index) => ({
    ...decision,
    decision_id: `d_${index + 1}`,
    round_index: index + 1,
  }));
  const first = parseDecisionPage(
    { data: decisions.slice(0, 10), total: 25, next_offset: 10 },
    0,
  );
  const second = parseDecisionPage(
    { data: decisions.slice(10, 20), total: 25, next_offset: 20 },
    10,
  );
  const third = parseDecisionPage(
    { data: decisions.slice(20), total: 25, next_offset: null },
    20,
  );
  expect([first.next_offset, second.next_offset, third.next_offset]).toEqual([
    10,
    20,
    null,
  ]);
  expect(third.data.map((item) => item.round_index)).toEqual([
    21, 22, 23, 24, 25,
  ]);
  expect(
    parseDecisionPage({ data: decisions, total: 25 }, 0).data,
  ).toHaveLength(autonomousDecisionPageSize);
});

test("malformed pages do not fabricate totals or permit repeated offsets", () => {
  expect(
    parseDecisionPage({ data: {}, total: NaN, next_offset: 10 }, 10),
  ).toEqual({ data: [], total: null, next_offset: null });
});
test("rejection pages are bounded and omit confidential fields", () => {
  const page = parseRejectionPage(
    {
      data: Array.from({ length: 15 }, () => ({
        failure_reason: "invalid proposal",
        api_key: "private",
        system_prompt: "private",
      })),
      next_offset: 10,
    },
    0,
  );
  expect(page.data).toHaveLength(10);
  expect(page.data[0]).toEqual({ failure_reason: "invalid proposal" });
  expect(parseRejectionPage({ data: {}, next_offset: 10 }, 10)).toEqual({
    data: [],
    next_offset: null,
  });
});
test("missing Test and freeze evidence stay unrecorded", () => {
  const parsed = parseSummary({
    ...summary,
    test_status: undefined,
    freeze: {},
  })!;
  expect(autonomousTestStatusLabel(parsed, true)).toBe("未记录");
  expect(parsed.freeze).toBeNull();
  expect(
    autonomousTestStatusLabel(
      parseSummary({
        ...summary,
        stage: "final_test_failed",
        test_status: "consumed",
      })!,
      true,
    ),
  ).toContain("失败");
});

test("parameter trials keep actual Cal CV order, values and failed numerical gaps", () => {
  const trials = parameterTrials([
    {
      parameters: { alpha: 2 },
      RMSECV: 1.2,
      fold_rmse: [1, 1.4],
      scope: "calibration_cv",
    },
    {
      parameters: { alpha: 3 },
      RMSECV: NaN,
      fold_rmse: [Infinity],
      scope: "calibration_cv",
    },
    { parameters: { alpha: 4 }, RMSECV: 0.01, scope: "final_test" },
  ]);
  expect(trials).toEqual([
    {
      index: 1,
      parameters: { alpha: 2 },
      RMSECV: 1.2,
      fold_rmse: [1, 1.4],
      scope: "calibration_cv",
    },
    {
      index: 2,
      parameters: { alpha: 3 },
      RMSECV: null,
      fold_rmse: [null],
      scope: "calibration_cv",
    },
  ]);
});

const finalChart = {
  scope: "final_test",
  evidence_status: "available",
  observed: [1, 2],
  predicted: [1.1, 2.2],
  residual: [0.1, 0.2],
  sample_ids: ["test_0", "test_1"],
  n_total: 2,
  n_returned: 2,
  sampling: "full_population",
};
test("Test plots require finalization plus finite aligned persisted evidence", () => {
  expect(parseFinalChart(finalChart, "frozen")).toBeNull();
  expect(
    parseFinalChart({ ...finalChart, predicted: [NaN, 2] }, "finalized"),
  ).toBeNull();
  expect(
    parseFinalChart({ ...finalChart, sample_ids: ["test_0"] }, "finalized"),
  ).toBeNull();
  expect(
    parseFinalChart({ ...finalChart, n_total: 1 }, "finalized"),
  ).toBeNull();
  expect(parseFinalChart(finalChart, "finalized")?.residual).toEqual([
    0.1, 0.2,
  ]);
});

test("custom controls reject empty, decimal, non-finite and out-of-range values", () => {
  expect(isCustomBudgetValid({})).toBe(true);
  expect(isCustomBudgetValid({ max_fits: 8 })).toBe(true);
  for (const max_fits of [0, -1, 1.5, 1001, NaN, Infinity])
    expect(isCustomBudgetValid({ max_fits })).toBe(false);
});

test("runtime custom requests still allowlist the four user budget fields", () => {
  expect(
    budgetSelection("custom", { max_fits: 8, max_llm_calls: 999 } as Parameters<
      typeof budgetSelection
    >[1]).budget,
  ).toEqual({ max_fits: 8 });
});

test("default vertical process opens one round detail while keeping Test charts lazy", () => {
  rs.mocked(useRoundHistory).mockReturnValue({
    rows: [parseDecision({ ...decision, round_index: 1 })!],
    total: 25,
    refetch: rs.fn(),
  } as unknown as ReturnType<typeof useRoundHistory>);
  const mocked = rs.mocked(useProcessResource);
  mocked.mockReset();
  mocked.mockImplementation(
    (_thread, resource) =>
      ({
        data:
          resource === "autonomous/search_a"
            ? { ...summary, stage: "finalized" }
            : {
                data: [{ ...decision, round_index: 1 }],
                total: 25,
                next_offset: 10,
              },
        isError: false,
        isPending: false,
        refetch: rs.fn(),
      }) as unknown as ReturnType<typeof useProcessResource>,
  );
  const html = renderToStaticMarkup(
    createElement(AutonomousProcess, {
      threadId: "thread_a",
      sessionId: "search_a",
      workflow: { approval_status: "pending" },
    }),
  );
  expect(mocked.mock.calls.map((call) => call[1])).toEqual([
    null,
    "autonomous/search_a",
    null,
    "autonomous/search_a/decisions/d_a",
    null,
    null,
    "autonomous/search_a/rejections?offset=0",
  ]);
  expect(mocked.mock.calls.at(-1)?.[2]).toBe(false);
  expect(
    mocked.mock.calls.every((call) => !String(call[1]).includes("/charts/")),
  ).toBe(true);
  expect(html).toContain("已加载 1 / 25 次决策");
  expect(html).not.toContain("已注册");
});

test("ordinary chat context drops obsolete mode fields without mutating settings", () => {
  const old = {
    model_name: "deepseek",
    thinking_enabled: true,
    nir_autonomous: { enabled: true },
    nir_web_request: {},
    nir_unified_entry: {},
  };
  expect(modelingChatContext(old)).toEqual({
    model_name: "deepseek",
    thinking_enabled: true,
  });
  expect(old.nir_autonomous.enabled).toBe(true);
});

test("deterministic publication state is displayed without a final chat message", () => {
  const mocked = rs.mocked(useProcessResource);
  mocked.mockReset();
  mocked.mockImplementation(
    (_thread, resource) =>
      ({
        data:
          resource === "autonomous/search_a"
            ? {
                ...summary,
                stage: "finalized",
                automatic_delivery: { status: "delivered" },
                export_status: "succeeded",
              }
            : { data: [], total: 0 },
        isError: false,
        isPending: false,
        refetch: rs.fn(),
      }) as unknown as ReturnType<typeof useProcessResource>,
  );
  const html = renderToStaticMarkup(
    createElement(AutonomousProcess, {
      threadId: "thread_a",
      sessionId: "search_a",
      workflow: { approval_status: "pending" },
    }),
  );
  expect(html).toContain("已完成交付，等待用户审批");
  expect(html).not.toContain("已注册");
});
