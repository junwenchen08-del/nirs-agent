import { expect, test } from "@rstest/core";

import { selectNIRMilestones } from "@/core/nir/progress";

test("shows the current modeling phase without claiming a percent complete", () => {
  expect(
    selectNIRMilestones({ taskType: "calibration", stage: "evaluation" }),
  ).toEqual([
    { key: "data", status: "complete" },
    { key: "plan", status: "complete" },
    { key: "model", status: "current" },
    { key: "review", status: "upcoming" },
  ]);
});

test("marks review active until the workflow is completed", () => {
  expect(
    selectNIRMilestones({ taskType: "classification", stage: "approved" })?.map(
      (item) => item.status,
    ),
  ).toEqual(["complete", "complete", "complete", "current"]);
  expect(
    selectNIRMilestones({
      taskType: "classification",
      stage: "completed",
    })?.map((item) => item.status),
  ).toEqual(["complete", "complete", "complete", "complete"]);
});

test("does not imply modeling progress for blocked or inspection workflows", () => {
  expect(
    selectNIRMilestones({ taskType: "calibration", stage: "blocked" }),
  ).toBeNull();
  expect(
    selectNIRMilestones({ taskType: "inspection", stage: "data_audit" }),
  ).toBeNull();
});
