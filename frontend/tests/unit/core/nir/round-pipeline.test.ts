import { expect, test } from "@rstest/core";

import { parseDecision, record } from "@/core/nir/autonomous";
import { roundPipeline } from "@/core/nir/round-pipeline";

function row(
  id: string,
  method = "pls",
  steps: unknown[] | undefined = [],
  params = {},
) {
  return parseDecision({
    decision_id: id,
    experiment_id: `e_${id}`,
    status: "succeeded",
    action: "EXPLORE",
    selected: {
      pipeline: {
        model_method: "ridge",
        preprocessing: [{ method: "detrend" }],
      },
    },
    feedback: {
      actual_configuration: { method, ...(steps ? { steps } : {}) },
      actual_parameters: params,
    },
  })!;
}
test("pipeline uses executed configuration and distinguishes missing from explicit raw preprocessing", () => {
  expect(roundPipeline(row("a")).steps).toEqual([]);
  expect(roundPipeline(row("a")).stepsRecorded).toBe(true);
  const missing = row("a");
  delete record(missing.feedback!.actual_configuration).steps;
  expect(roundPipeline(missing).stepsRecorded).toBe(false);
  expect(roundPipeline(missing).method).toBe("pls");
});
test("only the recorded base supplies change labels, with order and parameter evidence", () => {
  const base = row("a", "pls", [{ method: "snv" }, { method: "center" }], {
    n_components: 3,
  });
  const next = {
    ...row("b", "svr", [
      { method: "snv" },
      { method: "smooth", params: { window: 7 } },
    ]),
    base_experiment_id: base.experiment_id!,
  };
  const pipeline = roundPipeline(next, base);
  expect(pipeline.steps.map((s) => s.change)).toEqual(["same", "replaced"]);
  expect(pipeline.modelChange).toBe("replaced");
  expect(pipeline.removed).toEqual(["center"]);
  expect(roundPipeline(next, row("wrong")).steps[0]!.change).toBe("unknown");
});
test("changed parameters and reordered steps are visible without inventing execution", () => {
  const base = row(
    "a",
    "pls",
    [{ method: "snv" }, { method: "smooth", params: { window: 5 } }],
    { n_components: 3 },
  );
  const next = {
    ...row(
      "b",
      "pls",
      [{ method: "smooth", params: { window: 7 } }, { method: "snv" }],
      { n_components: 5 },
    ),
    base_experiment_id: base.experiment_id!,
  };
  expect(roundPipeline(next, base).steps.map((s) => s.change)).toEqual([
    "tuned",
    "moved",
  ]);
  expect(roundPipeline(next, base).modelChange).toBe("tuned");
  next.status = "accepted";
  expect(roundPipeline(next, base).actual).toBe(false);
  expect(
    roundPipeline(next, base).steps.every((s) => s.change === "unknown"),
  ).toBe(true);
});
