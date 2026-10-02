import { expect, test } from "@rstest/core";

import {
  selectLatestNIRWorkflow,
  selectNIRWorkflow,
  selectNIRWorkflowView,
} from "@/core/nir/selectors";

test("returns null when a thread has no recognizable NIR workflow", () => {
  expect(selectNIRWorkflow(undefined)).toBeNull();
  expect(selectNIRWorkflow({ stage: "planning" })).toBeNull();
  expect(selectNIRWorkflow({ task_type: "calibration" })).toBeNull();
});

test("projects bounded regression evidence without exposing artifact paths", () => {
  const view = selectNIRWorkflowView({
    project_id: "nir-project-one",
    task_type: "calibration",
    stage: "review",
    revision: 8,
    attempt: 2,
    max_attempts: 3,
    validation_goal: "external_validation",
    approval_status: "pending",
    dataset_id: "ds_one",
    dataset_profile_id: "dsp_one",
    dataset_sha256: "a".repeat(64),
    data_path: "/mnt/user-data/uploads/private.csv",
    model_path: "/mnt/user-data/outputs/private.pkl",
    attempt_evidence: {
      tool_name: "nir_train_partitioned_model",
      method: "pls",
      pipeline_steps: [
        { method: "snv", params: {} },
        { method: "derivative1", params: { window: 11 } },
      ],
      validation_scope: "independent_external_validation",
      metrics_summary: {
        RMSEP: 0.12345,
        R2_val: 0.93456,
        RPD: 3.456,
        grade: "A",
        passed: true,
      },
      metrics_path: "/mnt/user-data/outputs/private-metrics.json",
    },
  });

  expect(view).toMatchObject({
    projectId: "nir-project-one",
    taskType: "calibration",
    stage: "review",
    revision: 8,
    attempt: 2,
    maxAttempts: 3,
    validationGoal: "external_validation",
    validationScope: "independent_external_validation",
    approvalStatus: "pending",
    method: "pls",
    preprocessing: ["snv", "derivative1"],
    datasetId: "ds_one",
    datasetProfileId: "dsp_one",
    datasetHashShort: "aaaaaaaaaaaa",
  });
  expect(view?.metrics).toEqual([
    { key: "R2_val", value: "0.9346" },
    { key: "RPD", value: "3.456" },
    { key: "RMSEP", value: "0.1235" },
    { key: "grade", value: "A" },
    { key: "passed", value: "yes" },
  ]);
  expect(JSON.stringify(view)).not.toContain("/mnt/");
});

test("selects classification holdout metrics and preserves internal validation scope", () => {
  const view = selectNIRWorkflowView({
    project_id: "classification-one",
    task_type: "classification",
    stage: "evaluation",
    revision: 4,
    attempt: 1,
    max_attempts: 3,
    validation_goal: "internal_holdout",
    approval_status: "not_required",
    attempts: [
      {
        tool_name: "nir_train_classifier",
        validation_scope: "independent_holdout_not_external",
        pipeline_steps: ["snv"],
        metrics_summary: {
          holdout: {
            balanced_accuracy: 0.91,
            macro_f1: 0.89,
            mcc: 0.84,
          },
        },
      },
    ],
  });

  expect(view?.validationScope).toBe("independent_holdout_not_external");
  expect(view?.metrics).toEqual([
    { key: "balanced_accuracy", value: "0.91" },
    { key: "macro_f1", value: "0.89" },
    { key: "mcc", value: "0.84" },
  ]);
});

test("uses the latest attempt and tolerates legacy workflow fields", () => {
  const view = selectNIRWorkflowView({
    project_id: "legacy-one",
    task_type: "calibration",
    stage: "execution",
    revision: 2,
    attempts: [
      { method: "pls", metrics_summary: { RMSECV: 2 } },
      { method: "svr", metrics_summary: { RMSECV: 1.25 } },
    ],
  });

  expect(view).toMatchObject({
    method: "svr",
    attempt: 0,
    maxAttempts: 0,
    validationScope: null,
    approvalStatus: null,
    datasetId: null,
  });
  expect(view?.metrics).toEqual([{ key: "RMSECV", value: "1.25" }]);
});

test("blocks arbitrary metric objects and caps the visible metric list", () => {
  const view = selectNIRWorkflowView({
    project_id: "bounded-one",
    task_type: "multi_modeling",
    stage: "completed",
    revision: 9,
    attempt_evidence: {
      metrics_summary: {
        RMSEC: 1,
        RMSECV: 2,
        RMSEP: 3,
        R2_val: 0.8,
        RPD: 2.5,
        balanced_accuracy: 0.9,
        macro_f1: 0.88,
        mcc: 0.8,
        secret_blob: { raw: [1, 2, 3] },
      },
    },
  });

  expect(view?.metrics).toHaveLength(8);
  expect(JSON.stringify(view)).not.toContain("secret_blob");
  expect(JSON.stringify(view)).not.toContain("raw");
});

test("bounds user-visible workflow strings from legacy checkpoints", () => {
  const view = selectNIRWorkflowView({
    project_id: `project-${"x".repeat(500)}`,
    task_type: `task-${"x".repeat(500)}`,
    stage: `stage-${"x".repeat(500)}`,
    revision: 1,
    next_action: `next-${"x".repeat(500)}`,
    dataset_id: `dataset-${"x".repeat(500)}`,
    attempt_evidence: {
      method: `method-${"x".repeat(500)}`,
      pipeline_steps: [`step-${"x".repeat(500)}`],
    },
  });

  expect(view).not.toBeNull();
  expect(view?.projectId.length).toBeLessThanOrEqual(128);
  expect(view?.taskType.length).toBeLessThanOrEqual(128);
  expect(view?.stage.length).toBeLessThanOrEqual(128);
  expect(view?.nextAction?.length).toBeLessThanOrEqual(128);
  expect(view?.datasetId?.length).toBeLessThanOrEqual(128);
  expect(view?.method?.length).toBeLessThanOrEqual(128);
  expect(view?.preprocessing[0]?.length).toBeLessThanOrEqual(128);
});

test("prefers newer fallback revisions but never replaces a live project with another project", () => {
  const live = {
    project_id: "live-project",
    task_type: "analysis",
    stage: "execution",
    revision: 2,
  };
  const newerSameProject = {
    ...live,
    stage: "review",
    revision: 3,
  };
  const unrelatedFallback = {
    ...newerSameProject,
    project_id: "old-project",
    revision: 99,
  };

  expect(selectLatestNIRWorkflow(live, newerSameProject)?.revision).toBe(3);
  expect(selectLatestNIRWorkflow(live, unrelatedFallback)?.project_id).toBe(
    "live-project",
  );
  expect(selectLatestNIRWorkflow(undefined, newerSameProject)?.revision).toBe(
    3,
  );
});
