export const PROCESS_STEPS = [
  "audit",
  "split",
  "preprocessing",
  "wavelength",
  "model",
  "validation",
  "reflection",
  "registration",
] as const;
export type StepKey = (typeof PROCESS_STEPS)[number];
export interface StepSummary {
  step_execution_id: string;
  step_key: StepKey | "workflow";
  execution_status: string;
  evidence_status: string;
  comparison_group_id?: string | null;
  comparison_reason?: string | null;
  score?: number | null;
}
export interface ProcessAttempt {
  attempt_id: string;
  number: number;
  source_tool: string;
  execution_status: string;
  started_at: string;
  ended_at: string | null;
  steps: StepSummary[];
}
export interface ProcessSummary {
  schema_version: 1;
  run_id: string;
  revision: number;
  stage: string;
  updated_at: string;
  capabilities: {
    can_view_raw_spectra: boolean;
    can_view_sample_points: boolean;
  };
  attempts: ProcessAttempt[];
  shared_steps: StepSummary[];
  next_offset: number | null;
  offset?: number;
}
export interface ChartRef {
  chart_id: string;
  version: string;
  type: string;
  evidence_status: string;
}
export interface ProcessStep extends StepSummary {
  attempt_id?: string | null;
  facts: Record<string, unknown>;
  charts: ChartRef[];
  candidate_count: number;
  metrics?: Record<string, Record<string, number>>;
  evaluation_context?: Record<string, unknown>;
}
export interface ProcessCandidate {
  candidate_id: string;
  method?: string;
  score?: number | null;
  evaluation_kind?: string;
  selected?: boolean;
  error?: string;
  [key: string]: unknown;
}
export interface ProcessChart {
  chart_schema_version: 1;
  chart_id: string;
  run_id: string;
  attempt_id: string;
  step_execution_id: string;
  evidence_status: string;
  reason?: string;
  type: string;
  scope?: string;
  x_label?: string;
  y_label?: string;
  x_unit?: string | null;
  y_unit?: string | null;
  x?: number[];
  y?: number[];
  series?: { id: string; values: number[] }[];
  sample_ids?: string[];
  ranges?: { start: number; end: number; count: number }[];
  n_total: number;
  n_returned: number;
  axis_total?: number;
  axis_bounds?: [number, number];
  axis_returned?: number;
  sampling: string;
}
export interface RunList {
  data: { run_id: string; stage: string; revision: number; updated: string }[];
  next_offset: number | null;
}
export interface CandidatePage {
  data: ProcessCandidate[];
  total: number;
  next_offset: number | null;
}

export function stableJSON(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJSON).join(",")}]`;
  if (value !== null && typeof value === "object")
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${stableJSON(item)}`)
      .join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

export function pipelineDifference(before: unknown, after: unknown): boolean {
  return stableJSON(before) !== stableJSON(after);
}

export function comparableScores(
  attempts: ProcessAttempt[],
  reference: ProcessAttempt,
) {
  const model = reference.steps.find((step) => step.step_key === "model");
  if (!model?.comparison_group_id) return [];
  return attempts.flatMap((attempt) => {
    const step = attempt.steps.find((item) => item.step_key === "model");
    return step &&
      step.comparison_group_id === model.comparison_group_id &&
      typeof step.score === "number" &&
      Number.isFinite(step.score)
      ? [{ attempt, value: step.score }]
      : [];
  });
}

export function displayedPointSummary(chart: ProcessChart, selected: string[]) {
  const indices = (chart.sample_ids ?? []).flatMap((id, index) =>
    selected.includes(id) ? [index] : [],
  );
  const values = indices
    .map((index) => chart.y?.[index])
    .filter(
      (value): value is number =>
        typeof value === "number" && Number.isFinite(value),
    );
  return {
    count: indices.length,
    mean: values.length
      ? values.reduce((sum, value) => sum + value, 0) / values.length
      : null,
  };
}
