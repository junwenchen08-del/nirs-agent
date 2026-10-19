import { selectNIRWorkflow, selectNIRWorkflowView } from "./selectors";

type Row = Record<string, unknown>;
const LIMIT = 10;
const TEXT = 160;

function row(value: unknown): Row {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Row)
    : {};
}

function safeText(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text && !/\\|\/mnt\/|[a-z]:\//i.test(text)
    ? text.slice(0, TEXT)
    : null;
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function integer(value: unknown): number | null {
  const n = number(value);
  return n !== null && Number.isInteger(n) && n >= 0 ? n : null;
}

function rows(value: unknown): Row[] {
  return Array.isArray(value)
    ? value
        .filter((v) => Object.keys(row(v)).length > 0)
        .slice(-LIMIT)
        .map(row)
    : [];
}

function steps(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .slice(0, 8)
        .map(
          (v) =>
            safeText(v) ?? safeText(row(v).method) ?? safeText(row(v).name),
        )
        .filter((v): v is string => !!v)
    : [];
}

function metric(summary: Row, key: string): number | null {
  for (const source of [
    summary,
    row(summary.external),
    row(summary.holdout),
    row(summary.validation),
    row(summary.test),
  ]) {
    const found = number(source[key]);
    if (found !== null) return found;
  }
  return null;
}

const CLASS_METRICS = ["mcc", "balanced_accuracy", "macro_f1"];
const REGRESSION_METRICS = ["RMSEP", "RMSECV", "RMSEC"];

export interface CandidateView {
  name: string;
  steps: string[];
  score: number | null;
  scoreKey: string | null;
  selected: boolean;
  reason: string | null;
}

export interface SelectionView {
  selected: string | null;
  reasonCode: string | null;
  comparisonReasonCode: string | null;
  reason: string | null;
  rule: string | null;
  improvement: number | null;
  minImprovement: number | null;
  candidates: CandidateView[];
}

export interface AttemptView {
  number: number;
  method: string | null;
  pipeline: string[];
  passed: boolean | null;
  grade: string | null;
  protocol: string | null;
  validationScope: string | null;
  metricKey: string | null;
  metricValue: number | null;
  metrics: { key: string; value: number }[];
  preprocessing: SelectionView;
  wavelength: SelectionView;
  model: SelectionView;
  reflection: {
    reason: string | null;
    stopReason: string | null;
    shouldRetry: boolean | null;
    diagnostics: { key: string; value: string }[];
  } | null;
  retry: {
    method: string | null;
    pipeline: string[];
    rationale: string | null;
    expectedImprovement: string | null;
  } | null;
  pipelineChanges: string[];
}

export interface ProcessView {
  base: NonNullable<ReturnType<typeof selectNIRWorkflowView>>;
  unit: string | null;
  audit: {
    samples: number | null;
    wavelengths: number | null;
    range: string | null;
    hasNaN: boolean | null;
    constantWavelengths: number | null;
    usableWavelengths: number | null;
  };
  attempts: AttemptView[];
  bestAttempt: number | null;
  comparableAttempts: number[];
  trend: { attempt: number; value: number; best: number }[];
  stopReason: string | null;
  finalReady: boolean;
  registered: boolean;
  modelLibrary: boolean | null;
}

function selection(
  facts: Row,
  kind: "preprocessing" | "wavelength" | "model",
): SelectionView {
  const data = row(facts[`${kind}_selection`]);
  const decision = row(facts[`${kind}_selection_decision`]);
  const adoption = row(decision.adoption);
  const rawCandidates =
    kind === "preprocessing" ? data.candidates : facts[`${kind}_candidates`];
  const selected =
    safeText(
      kind === "preprocessing"
        ? data.selected_candidate_id
        : decision.selected_method,
    ) ??
    safeText(
      kind === "wavelength" ? row(facts.wavelength_selection).method : null,
    );
  const candidates = rows(rawCandidates)
    .slice(0, 8)
    .map((candidate) => {
      const scoreKeys =
        kind === "preprocessing" ? ["cv_rmse"] : ["RMSE_tuning", "RMSECV"];
      const scoreKey =
        scoreKeys.find((key) => number(candidate[key]) !== null) ?? null;
      const name =
        safeText(candidate.candidate_id) ?? safeText(candidate.method) ?? "—";
      return {
        name,
        steps: steps(candidate.steps),
        score: scoreKey ? number(candidate[scoreKey]) : null,
        scoreKey,
        selected: candidate.selected === true || name === selected,
        reason: safeText(
          Array.isArray(candidate.reasons)
            ? candidate.reasons[0]
            : candidate.reason,
        ),
      };
    });
  return {
    selected,
    reasonCode:
      safeText(data.reason_code) ??
      safeText(adoption.reason_code) ??
      safeText(decision.reason_code),
    comparisonReasonCode: safeText(decision.reason_code),
    reason: safeText(decision.reason),
    rule: safeText(data.selection_rule),
    improvement: number(adoption.relative_RMSE_improvement),
    minImprovement: number(adoption.minimum_required_improvement),
    candidates,
  };
}

function range(value: unknown): string | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const low = number(value[0]);
  const high = number(value[1]);
  return low !== null && high !== null ? `${low}–${high}` : null;
}

export function selectNIRProcessView(value: unknown): ProcessView | null {
  const workflow = selectNIRWorkflow(value);
  const base = selectNIRWorkflowView(value);
  if (!workflow || !base) return null;
  const audit = row(workflow.audit_evidence);
  const reflections = rows(workflow.reflections);
  if (!reflections.length && Object.keys(row(workflow.reflection)).length)
    reflections.push(row(workflow.reflection));
  const plans = rows(workflow.retry_plans);
  if (!plans.length && Object.keys(row(workflow.retry_plan)).length)
    plans.push(row(workflow.retry_plan));
  const latestEvidence = row(workflow.attempt_evidence);
  const rawAttempts = rows(workflow.attempts);
  const attempts: AttemptView[] = rawAttempts.map((raw, index) => {
    const attemptNumber = integer(raw.attempt) ?? index + 1;
    const facts = row(raw.decision_facts);
    // Legacy workflows only have selection facts on their latest attempt.
    const latest =
      attemptNumber === base.attempt ? row(latestEvidence.result_facts) : {};
    const evidence = { ...latest, ...facts };
    const summary = row(raw.metrics_summary);
    const keys =
      base.taskType === "classification" ? CLASS_METRICS : REGRESSION_METRICS;
    const metricKey = keys.find((key) => metric(summary, key) !== null) ?? null;
    const reflection = reflections.find(
      (item) => integer(item.source_attempt) === attemptNumber,
    );
    const retry = plans.find(
      (item) => integer(item.source_attempt) === attemptNumber,
    );
    const pipeline = steps(raw.pipeline_steps).length
      ? steps(raw.pipeline_steps)
      : steps(row(evidence.preprocessing_selection).selected_pipeline);
    const previous = index ? steps(rawAttempts[index - 1]?.pipeline_steps) : [];
    const pipelineChanges =
      index === 0
        ? []
        : [
            ...pipeline
              .filter((s) => !previous.includes(s))
              .map((s) => `+ ${s}`),
            ...previous
              .filter((s) => !pipeline.includes(s))
              .map((s) => `− ${s}`),
            ...(safeText(raw.method) !==
            safeText(rawAttempts[index - 1]?.method)
              ? [
                  `model: ${safeText(rawAttempts[index - 1]?.method) ?? "—"} → ${safeText(raw.method) ?? "—"}`,
                ]
              : []),
          ];
    const diagnostics = row(reflection?.diagnostics);
    return {
      number: attemptNumber,
      method:
        safeText(raw.selected_method) ??
        safeText(row(evidence.model_selection_decision).selected_method) ??
        safeText(raw.method),
      pipeline,
      passed: typeof raw.passed === "boolean" ? raw.passed : null,
      grade: safeText(raw.grade),
      protocol: safeText(raw.protocol),
      validationScope: safeText(raw.validation_scope),
      metricKey,
      metricValue: metricKey ? metric(summary, metricKey) : null,
      metrics: [...CLASS_METRICS, ...REGRESSION_METRICS, "R2_val", "RPD"]
        .filter((key) => metric(summary, key) !== null)
        .slice(0, 8)
        .map((key) => ({ key, value: metric(summary, key)! })),
      preprocessing: selection(evidence, "preprocessing"),
      wavelength: selection(evidence, "wavelength"),
      model: selection(evidence, "model"),
      reflection: reflection
        ? {
            reason: safeText(reflection.reason),
            stopReason: safeText(reflection.stop_reason),
            shouldRetry:
              typeof reflection.should_retry === "boolean"
                ? reflection.should_retry
                : null,
            diagnostics: Object.keys(diagnostics)
              .slice(0, 5)
              .filter((key) => /^[a-z_]{1,40}$/.test(key))
              .map((key) => ({
                key,
                value:
                  safeText(diagnostics[key]) ??
                  number(diagnostics[key])?.toString() ??
                  (typeof diagnostics[key] === "boolean"
                    ? String(diagnostics[key])
                    : "—"),
              })),
          }
        : null,
      retry: retry
        ? {
            method: safeText(retry.method),
            pipeline: steps(retry.pipeline_steps),
            rationale: safeText(retry.rationale),
            expectedImprovement: safeText(retry.expected_improvement),
          }
        : null,
      pipelineChanges,
    };
  });
  // A pre-history workflow still has one live attempt; expose it without inventing its number or score.
  if (!attempts.length && base.attempt > 0) {
    const facts = row(latestEvidence.result_facts);
    const summary = row(latestEvidence.metrics_summary);
    const keys =
      base.taskType === "classification" ? CLASS_METRICS : REGRESSION_METRICS;
    const metricKey = keys.find((key) => metric(summary, key) !== null) ?? null;
    attempts.push({
      number: base.attempt,
      method:
        safeText(row(facts.model_selection_decision).selected_method) ??
        safeText(facts.method) ??
        base.method,
      pipeline: steps(row(facts.preprocessing_selection).selected_pipeline)
        .length
        ? steps(row(facts.preprocessing_selection).selected_pipeline)
        : base.preprocessing,
      passed: typeof summary.passed === "boolean" ? summary.passed : null,
      grade: safeText(summary.grade),
      protocol: safeText(latestEvidence.protocol),
      validationScope: base.validationScope,
      metricKey,
      metricValue: metricKey ? metric(summary, metricKey) : null,
      metrics: keys
        .filter((key) => metric(summary, key) !== null)
        .map((key) => ({ key, value: metric(summary, key)! })),
      preprocessing: selection(facts, "preprocessing"),
      wavelength: selection(facts, "wavelength"),
      model: selection(facts, "model"),
      reflection: null,
      retry: null,
      pipelineChanges: [],
    });
  }
  const withMetric = attempts.filter(
    (attempt) => attempt.metricKey && attempt.metricValue !== null,
  );
  const reference = withMetric.at(-1);
  // Legacy checkpoint summaries omit actual split/fold and metric provenance.
  // Do not rank these scores: authoritative comparisons live in process API.
  const comparable: AttemptView[] = [];
  const lowerBetter = reference
    ? reference.metricKey?.startsWith("RMSE")
    : true;
  let best: AttemptView | null = null;
  const trend = comparable.map((attempt) => {
    if (
      !best ||
      (lowerBetter
        ? attempt.metricValue! < best.metricValue!
        : attempt.metricValue! > best.metricValue!)
    )
      best = attempt;
    return {
      attempt: attempt.number,
      value: attempt.metricValue!,
      best: best.metricValue!,
    };
  });
  const lastReflection = attempts.at(-1)?.reflection;
  return {
    base,
    unit: safeText(workflow.unit),
    audit: {
      samples: integer(audit.n_samples),
      wavelengths: integer(audit.n_wavelengths),
      range:
        range(audit.usable_wavelength_range) ??
        range(audit.raw_wavelength_range),
      hasNaN: typeof audit.has_nan === "boolean" ? audit.has_nan : null,
      constantWavelengths: integer(audit.constant_wavelength_count),
      usableWavelengths: integer(audit.usable_wavelength_count),
    },
    attempts,
    bestAttempt: best ? (best as AttemptView).number : null,
    comparableAttempts: comparable.map((attempt) => attempt.number),
    trend,
    stopReason: lastReflection?.stopReason ?? null,
    finalReady:
      ["review", "approved", "registered", "completed", "blocked"].includes(
        base.stage,
      ) && attempts.length > 0,
    registered:
      base.stage === "registered" ||
      rows(workflow.history).some((event) => event.action === "registered"),
    modelLibrary: null,
  };
}
