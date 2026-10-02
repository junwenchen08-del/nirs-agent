import type { NIRMetricView, NIRWorkflowState, NIRWorkflowView } from "./types";

const METRIC_KEYS = [
  "R2_val",
  "RPD",
  "RMSEC",
  "RMSECV",
  "RMSEP",
  "balanced_accuracy",
  "macro_f1",
  "mcc",
  "accuracy",
  "f1",
  "bias",
  "slope",
  "grade",
  "passed",
] as const;
const METRIC_CONTAINERS = ["external", "holdout", "validation", "test"];
const MAX_VISIBLE_METRICS = 8;
const MAX_PIPELINE_STEPS = 12;
const MAX_VISIBLE_STRING_LENGTH = 128;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function optionalString(value: unknown): string | null {
  return typeof value === "string" && value.trim()
    ? value.trim().slice(0, MAX_VISIBLE_STRING_LENGTH)
    : null;
}

function boundedNonNegativeInteger(value: unknown): number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0
    ? Math.min(value, 10_000)
    : 0;
}

function attemptEvidence(workflow: NIRWorkflowState): Record<string, unknown> {
  if (isRecord(workflow.attempt_evidence)) {
    return workflow.attempt_evidence;
  }
  if (!Array.isArray(workflow.attempts)) {
    return {};
  }
  for (let index = workflow.attempts.length - 1; index >= 0; index -= 1) {
    const candidate = workflow.attempts[index];
    if (isRecord(candidate)) {
      return candidate;
    }
  }
  return {};
}

function selectMethod(evidence: Record<string, unknown>): string | null {
  const direct = optionalString(evidence.method);
  if (direct) {
    return direct;
  }
  const facts = isRecord(evidence.result_facts) ? evidence.result_facts : null;
  return optionalString(facts?.method);
}

function selectPreprocessing(evidence: Record<string, unknown>): string[] {
  if (!Array.isArray(evidence.pipeline_steps)) {
    return [];
  }
  const methods: string[] = [];
  for (const step of evidence.pipeline_steps.slice(0, MAX_PIPELINE_STEPS)) {
    const method =
      optionalString(step) ??
      (isRecord(step)
        ? (optionalString(step.method) ?? optionalString(step.name))
        : null);
    if (method && !methods.includes(method)) {
      methods.push(method);
    }
  }
  return methods;
}

function findMetric(summary: Record<string, unknown>, key: string): unknown {
  if (summary[key] !== undefined) {
    return summary[key];
  }
  for (const container of METRIC_CONTAINERS) {
    const nested = summary[container];
    if (isRecord(nested) && nested[key] !== undefined) {
      return nested[key];
    }
  }
  return undefined;
}

function formatMetric(value: unknown): string | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return Number(value.toPrecision(4)).toString();
  }
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }
  if (typeof value === "string" && value.trim()) {
    return value.trim().slice(0, 32);
  }
  return null;
}

function selectMetrics(evidence: Record<string, unknown>): NIRMetricView[] {
  const summary = isRecord(evidence.metrics_summary)
    ? evidence.metrics_summary
    : {};
  const metrics: NIRMetricView[] = [];
  for (const key of METRIC_KEYS) {
    const value = formatMetric(findMetric(summary, key));
    if (value !== null) {
      metrics.push({ key, value });
    }
    if (metrics.length === MAX_VISIBLE_METRICS) {
      break;
    }
  }
  return metrics;
}

export function selectNIRWorkflow(value: unknown): NIRWorkflowState | null {
  if (!isRecord(value)) {
    return null;
  }
  const projectId = optionalString(value.project_id);
  const taskType = optionalString(value.task_type);
  const stage = optionalString(value.stage);
  if (
    !projectId ||
    !taskType ||
    !stage ||
    typeof value.revision !== "number" ||
    !Number.isInteger(value.revision) ||
    value.revision < 0
  ) {
    return null;
  }
  return {
    ...value,
    project_id: projectId,
    task_type: taskType,
    stage,
  } as NIRWorkflowState;
}

export function selectLatestNIRWorkflow(
  primary: unknown,
  fallback: unknown,
): NIRWorkflowState | null {
  const liveWorkflow = selectNIRWorkflow(primary);
  const recoveredWorkflow = selectNIRWorkflow(fallback);

  if (!liveWorkflow) {
    return recoveredWorkflow;
  }
  if (!recoveredWorkflow) {
    return liveWorkflow;
  }
  if (liveWorkflow.project_id !== recoveredWorkflow.project_id) {
    return liveWorkflow;
  }
  return recoveredWorkflow.revision > liveWorkflow.revision
    ? recoveredWorkflow
    : liveWorkflow;
}

export function selectNIRWorkflowView(value: unknown): NIRWorkflowView | null {
  const workflow = selectNIRWorkflow(value);
  if (!workflow) {
    return null;
  }
  const evidence = attemptEvidence(workflow);
  const digest = optionalString(workflow.dataset_sha256)?.toLowerCase();
  const datasetHashShort =
    digest && /^[a-f0-9]{64}$/.test(digest) ? digest.slice(0, 12) : null;

  return {
    projectId: workflow.project_id,
    taskType: workflow.task_type,
    stage: workflow.stage,
    revision: workflow.revision,
    attempt: boundedNonNegativeInteger(workflow.attempt),
    maxAttempts: boundedNonNegativeInteger(workflow.max_attempts),
    validationGoal: optionalString(workflow.validation_goal),
    validationScope: optionalString(evidence.validation_scope),
    approvalStatus: optionalString(workflow.approval_status),
    nextAction: optionalString(workflow.next_action),
    method: selectMethod(evidence),
    preprocessing: selectPreprocessing(evidence),
    metrics: selectMetrics(evidence),
    datasetId: optionalString(workflow.dataset_id),
    datasetProfileId: optionalString(workflow.dataset_profile_id),
    datasetHashShort,
  };
}
