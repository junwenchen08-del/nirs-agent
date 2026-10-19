import { type NIRWorkflowView } from "./types";

export const NIR_MILESTONES = ["data", "plan", "model", "review"] as const;

export type NIRMilestone = (typeof NIR_MILESTONES)[number];
export type NIRMilestoneStatus = "complete" | "current" | "upcoming";

const MODELING_TASKS = new Set([
  "analysis",
  "calibration",
  "classification",
  "multi_modeling",
  "compare",
]);

const STAGE_MILESTONE: Record<string, number> = {
  intake: 0,
  data_audit: 0,
  clarification: 0,
  planning: 1,
  execution: 2,
  knowledge: 2,
  evaluation: 2,
  review: 3,
  approved: 3,
  registered: 3,
  completed: 4,
};

export function selectNIRMilestones(
  view: Pick<NIRWorkflowView, "taskType" | "stage">,
): { key: NIRMilestone; status: NIRMilestoneStatus }[] | null {
  if (!MODELING_TASKS.has(view.taskType)) {
    return null;
  }
  const current = STAGE_MILESTONE[view.stage];
  if (current === undefined) {
    return null;
  }
  return NIR_MILESTONES.map((key, index) => ({
    key,
    status:
      index < current ? "complete" : index === current ? "current" : "upcoming",
  }));
}
