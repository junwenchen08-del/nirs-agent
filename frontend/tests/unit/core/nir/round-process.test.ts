import { expect, test } from "@rstest/core";

import { parseDecision, parseSummary } from "@/core/nir/autonomous";
import {
  parseRoundDiagnostics,
  roundGraph,
  roundGraphFocus,
  roundConfiguration,
  roundWorkflowPollingInterval,
  parseRoundPage,
} from "@/core/nir/round-process";

const make = (id: string, extra = {}) =>
  parseDecision({
    decision_id: `d_${id}`,
    experiment_id: `exp_${id}`,
    status: "succeeded",
    action: "EXPLORE",
    round_index: Number(id),
    selected: {
      pipeline: {
        preprocessing: [],
        model_method: "ridge",
        fixed_params: { alpha: 2 },
      },
    },
    ...extra,
  })!;

test("graph uses exact stored parents and retention, including an earlier retained round", () => {
  const rows = [
    make("1"),
    make("2", { base_experiment_id: "exp_1" }),
    make("3", { base_experiment_id: "exp_1" }),
    make("4", {
      action: "STOP",
      status: "accepted",
      base_experiment_id: "exp_3",
      retained_experiment_id: "exp_1",
    }),
  ];
  const graph = roundGraph(rows);
  expect(
    graph.edges.map((edge) => [edge.source, edge.target, edge.data?.relation]),
  ).toEqual([
    ["d_1", "d_2", "feedback"],
    ["d_1", "d_3", "feedback"],
    ["d_3", "d_4", "feedback"],
    ["d_1", "d_4", "retained"],
  ]);
  expect(graph.nodes[1]!.position.x).not.toBe(graph.nodes[2]!.position.x);
  expect(
    graph.edges.find((edge) => edge.data?.relation === "retained"),
  ).toMatchObject({
    sourceHandle: "retention-source",
    targetHandle: "retention-target",
    type: "smoothstep",
  });
});

test("unknown, self, later or cross-page parent references never fabricate edges", () => {
  const rows = [
    make("1", { base_experiment_id: "exp_2" }),
    make("2", { base_experiment_id: "exp_2" }),
    make("3", { base_experiment_id: "exp_missing" }),
  ];
  expect(roundGraph(rows).edges).toEqual([]);
});

test("graph focus keeps the selected round with its recorded neighbours", () => {
  const rows = [
    make("1"),
    make("2", { base_experiment_id: "exp_1" }),
    make("3", { base_experiment_id: "exp_2" }),
  ];
  const graph = roundGraph(rows);
  expect(roundGraphFocus(graph, "d_2").map((node) => node.id)).toEqual([
    "d_1",
    "d_2",
    "d_3",
  ]);
  expect(roundGraphFocus(graph, "d_1").map((node) => node.id)).toEqual([
    "d_1",
    "d_2",
  ]);
  expect(roundGraphFocus(graph, "d_3").map((node) => node.id)).toEqual([
    "d_2",
    "d_3",
  ]);
  expect(roundGraphFocus(graph, "").map((node) => node.id)).toEqual([
    "d_1",
    "d_2",
    "d_3",
  ]);
  expect(roundGraphFocus(roundGraph([]), "d_1")).toEqual([]);
});

test("actual configuration stays distinct from a proposed pipeline", () => {
  const selected = make("1");
  expect(roundConfiguration(selected).actual).toBe(false);
  expect(
    roundConfiguration(
      make("2", {
        feedback: {
          actual_configuration: { method: "svr", steps: [{ method: "snv" }] },
          actual_parameters: { C: 10 },
        },
      }),
    ),
  ).toMatchObject({
    actual: true,
    method: "svr",
    steps: [{ method: "snv" }],
    parameters: { C: 10 },
  });
});

const chart = {
  scope: "validation",
  evidence_status: "available",
  observed: [1, 2],
  predicted: [1.2, 1.9],
  residual: [0.2, -0.1],
  sample_ids: ["0123456789abcdef", "1123456789abcdef"],
  n_total: 2,
  n_returned: 2,
  sampling: "full_population",
};
const cache = {
  schema_version: 1,
  evidence_status: "available",
  session_id: "s_a",
  experiment_id: "exp_1",
  decision_id: "d_1",
  comparison_group_id: "g_a",
  charts: { validation: chart },
};

test("round charts require matching session, decision, experiment and comparison context", () => {
  expect(
    parseRoundDiagnostics(cache, "s_a", "exp_1", "d_1", "g_a")?.validation
      ?.n_total,
  ).toBe(2);
  for (const [field, value] of [
    ["session_id", "s_b"],
    ["experiment_id", "exp_2"],
    ["decision_id", "d_2"],
    ["comparison_group_id", "g_b"],
  ])
    expect(
      parseRoundDiagnostics(
        { ...cache, [field!]: value },
        "s_a",
        "exp_1",
        "d_1",
        "g_a",
      ),
    ).toBeNull();
});

test("Test scopes, inconsistent residuals, duplicate samples and partial arrays stay unavailable", () => {
  for (const patch of [
    { scope: "final_test" },
    { predicted: [1] },
    { residual: [0, 0] },
    { sample_ids: ["0123456789abcdef", "0123456789abcdef"] },
    { n_total: 1 },
  ])
    expect(
      parseRoundDiagnostics(
        { ...cache, charts: { validation: { ...chart, ...patch } } },
        "s_a",
        "exp_1",
        "d_1",
        "g_a",
      ),
    ).toBeNull();
  expect(
    parseRoundDiagnostics(
      { ...cache, charts: { final_test: chart } },
      "s_a",
      "exp_1",
      "d_1",
      "g_a",
    ),
  ).toBeNull();
});

test("next actual decision survives projection without recursively importing history", () => {
  const value = make("1", {
    next_decision: {
      decision_id: "d_next",
      next_decision: { decision_id: "d_leak" },
    },
  });
  expect(value.next_decision?.decision_id).toBe("d_next");
  expect(value.next_decision?.next_decision).toBeUndefined();
  expect(
    parseSummary({
      schema_version: 1,
      session_id: "s_a",
      comparison_group_id: "g_a",
      data_context: { unit: "%" },
    })?.data_context,
  ).toEqual({ unit: "%" });
});

test("round history pages retain twenty records and reject repeated cursors", () => {
  const data = Array.from({ length: 25 }, (_, index) => ({
    decision_id: `d_${index}`,
  }));
  expect(
    parseRoundPage({ data, total: 25, next_offset: 20 }, 0).data,
  ).toHaveLength(20);
  expect(
    parseRoundPage({ data, total: 25, next_offset: 20 }, 20).next_offset,
  ).toBeNull();
});

test("standalone process polls active user waits but stops at recorded workflow outcomes", () => {
  expect(
    roundWorkflowPollingInterval({
      stage: "execution",
      user_question: { status: "waiting" },
    }),
  ).toBe(3000);
  for (const stage of [
    "review",
    "approved",
    "registered",
    "completed",
    "blocked",
  ])
    expect(roundWorkflowPollingInterval({ stage })).toBe(false);
  expect(roundWorkflowPollingInterval(null)).toBe(false);
});
