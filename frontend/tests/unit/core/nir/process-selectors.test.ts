import { expect, test } from "@rstest/core";

import { selectNIRProcessView } from "@/core/nir/process-selectors";

const workflow = {
  project_id: "p1",
  task_type: "calibration",
  stage: "blocked",
  revision: 9,
  attempt: 3,
  max_attempts: 3,
  data_path: "/mnt/private/data.csv",
  audit_evidence: {
    n_samples: 80,
    n_wavelengths: 120,
    data_path: "/mnt/private/data.csv",
  },
  attempts: [
    {
      attempt: 1,
      method: "pls",
      pipeline_steps: ["snv"],
      protocol: "p",
      validation_scope: "independent_holdout_not_external",
      metrics_summary: { RMSEP: 2 },
      decision_facts: {
        preprocessing_selection: {
          selected_candidate_id: "snv",
          reason_code: "lowest_cv",
          candidates: [
            {
              candidate_id: "snv",
              cv_rmse: 1.5,
              selected: true,
              steps: ["snv"],
            },
            { candidate_id: "raw", cv_rmse: 1.8 },
          ],
        },
      },
    },
    {
      attempt: 2,
      method: "svr",
      pipeline_steps: ["snv", "sg"],
      protocol: "p",
      validation_scope: "independent_holdout_not_external",
      metrics_summary: { RMSEP: 1.2 },
    },
    {
      attempt: 3,
      method: "pls",
      pipeline_steps: ["sg"],
      protocol: "external",
      validation_scope: "independent_external_validation",
      metrics_summary: { RMSEP: 1.1 },
    },
  ],
  reflections: [
    {
      source_attempt: 1,
      should_retry: true,
      reason: "CV error",
      diagnostics: { high_bias: true },
    },
    {
      source_attempt: 3,
      should_retry: false,
      stop_reason: "retry_budget_exhausted",
    },
  ],
  retry_plans: [
    {
      source_attempt: 1,
      target_attempt: 2,
      method: "svr",
      pipeline_steps: ["snv", "sg"],
      rationale: "try nonlinear model",
    },
  ],
};

test("projects historical decisions, story and bounded audit without paths", () => {
  const view = selectNIRProcessView(workflow)!;
  expect(view.audit.samples).toBe(80);
  expect(view.attempts[0]?.preprocessing.candidates).toHaveLength(2);
  expect(view.attempts[0]?.reflection?.shouldRetry).toBe(true);
  expect(view.attempts[0]?.retry?.method).toBe("svr");
  expect(view.attempts[1]?.pipelineChanges).toEqual([
    "+ sg",
    "model: pls → svr",
  ]);
  expect(view.stopReason).toBe("retry_budget_exhausted");
  expect(JSON.stringify(view)).not.toContain("/mnt/");
});

test("does not rank legacy summaries without actual partition provenance", () => {
  const view = selectNIRProcessView(workflow)!;
  expect(view.comparableAttempts).toEqual([]);
  expect(view.trend).toEqual([]);
  expect(view.bestAttempt).toBeNull();
});

test("legacy and invalid workflows degrade without invented evidence", () => {
  expect(selectNIRProcessView({ stage: "review" })).toBeNull();
  const view = selectNIRProcessView({
    project_id: "old",
    task_type: "calibration",
    stage: "execution",
    revision: 1,
    attempt: 1,
    attempt_evidence: { method: "pls", metrics_summary: { RMSECV: 2 } },
  });
  expect(view?.attempts[0]?.model.selected).toBeNull();
  expect(view?.finalReady).toBe(false);
});

test("shows the selected automatic pipeline instead of the requested auto method", () => {
  const view = selectNIRProcessView({
    project_id: "auto",
    task_type: "calibration",
    stage: "review",
    revision: 3,
    attempt: 1,
    attempts: [
      {
        attempt: 1,
        method: "auto",
        selected_method: "ridge",
        pipeline_steps: [],
        decision_facts: {
          preprocessing_selection: {
            selected_pipeline: [{ method: "snv" }, { method: "sg" }],
          },
          model_selection_decision: { selected_method: "ridge" },
        },
      },
    ],
  });
  expect(view?.attempts[0]?.method).toBe("ridge");
  expect(view?.attempts[0]?.pipeline).toEqual(["snv", "sg"]);
});
