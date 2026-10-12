/** Allowlisted projection of the owner-checked, read-only autonomous API. */
import { methodLabel } from "./presentation";
import { type ProcessChart } from "./process-contract";

export interface AutonomousSelection {
  enabled: true;
  profile: "development" | "autonomous" | "custom";
  budget?: Partial<
    Record<
      "max_fits" | "max_experiments" | "max_decisions" | "max_wall_seconds",
      number
    >
  >;
}
export const autonomousBudgetLimits = {
  max_experiments: [5, 40],
  max_fits: [50, 1000],
  max_decisions: [15, 100],
  max_wall_seconds: [600, 7200],
} as const;

export function isCustomBudgetValid(budget: AutonomousSelection["budget"]) {
  return Object.entries(budget ?? {}).every(([key, value]) => {
    const limit =
      autonomousBudgetLimits[key as keyof typeof autonomousBudgetLimits];
    return (
      limit &&
      typeof value === "number" &&
      Number.isInteger(value) &&
      value >= 1 &&
      value <= limit[1]
    );
  });
}
type Facts = Record<string, unknown>;

export function decisionPipelineLabel(selected: Facts, zh: boolean): string {
  const pipeline = record(selected.pipeline);
  const method = text(pipeline.model_method, 50);
  if (!method) return zh ? "方案未记录" : "Pipeline not recorded";
  const methods = Array.isArray(pipeline.preprocessing)
    ? pipeline.preprocessing.map((step) => text(record(step).method, 60) || "—")
    : null;
  return [
    ...(methods === null
      ? [zh ? "预处理未记录" : "Preprocessing not recorded"]
      : methods.length
        ? methods.map((step) => methodLabel(step, zh))
        : [methodLabel("raw", zh)]),
    methodLabel(method, zh),
  ].join(" → ");
}
export interface DecisionView {
  decision_id: string;
  context_id: string;
  experiment_id: string | null;
  round_index: number | null;
  action: string;
  status: string;
  decision_time: string;
  rationale: string;
  expected_outcome: string;
  pipelineLabel: string;
  selected: Facts;
  origin: Facts;
  proposal?: Facts;
  accepted_spec?: Facts;
  feedback?: Facts & { metrics?: Facts };
}
export interface AutonomousSummary {
  schema_version: 1;
  session_id: string;
  revision: string;
  stage: string;
  test_status: string;
  autonomous_delivery: boolean;
  decision_count: number | null;
  budget: Facts;
  task: Facts;
  stop_source: string;
  stop_reason: unknown;
  retained_selection_source: string;
  retained_experiment_id: string | null;
  best_val_experiment_id: string | null;
  terminal_failure_source: string;
  terminal_failure_reason: unknown;
  freeze: Facts | null;
  metrics: Facts;
  quality: Facts;
  lifecycle_at_delivery: Facts;
  export_status: string;
  automatic_delivery?: Facts;
  downloads: { name: string; path: string }[];
  trends: {
    experiment_id: string;
    decision_id: string;
    round_index: number | null;
    method: string;
    metrics: Facts;
  }[];
  chart_scopes: string[];
  spectra_experiments: {
    experiment_id: string;
    decision_id: string;
    round_index: number | null;
    method: string;
  }[];
}
export const record = (value: unknown): Facts =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Facts)
    : {};
export const text = (value: unknown, limit = 2000): string =>
  typeof value === "string" && !/\/mnt\/|[A-Za-z]:[\\/]|\\\\/.test(value)
    ? value.slice(0, limit)
    : "";
const number = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;
const facts = (value: unknown, depth = 0): unknown => {
  if (depth > 8) return null;
  if (Array.isArray(value))
    return value.slice(0, 100).map((item) => facts(item, depth + 1));
  if (value && typeof value === "object")
    return Object.fromEntries(
      Object.entries(record(value))
        .slice(0, 100)
        .filter(
          ([key]) =>
            !/path|password|secret|api_key|system_prompt|raw_spectra/i.test(
              key,
            ),
        )
        .map(([key, item]) => [key, facts(item, depth + 1)]),
    );
  return typeof value === "string"
    ? text(value)
    : typeof value === "number"
      ? number(value)
      : typeof value === "boolean"
        ? value
        : null;
};
const safeFacts = (value: unknown) => record(facts(value));
export function budgetSelection(
  profile: AutonomousSelection["profile"],
  budget?: AutonomousSelection["budget"],
): AutonomousSelection {
  return {
    enabled: true,
    profile,
    ...(profile === "custom"
      ? {
          budget: Object.fromEntries(
            Object.entries(budget ?? {}).filter(([key]) =>
              Object.hasOwn(autonomousBudgetLimits, key),
            ),
          ),
        }
      : {}),
  };
}

export function modelingChatContext<T extends object>(context: T): T {
  // Old browser settings are read without silently resubmitting mode switches.
  // The Gateway owns task recognition and all scientific/resource policy.
  return Object.fromEntries(
    Object.entries(context).filter(
      ([key]) =>
        !["nir_autonomous", "nir_web_request", "nir_unified_entry"].includes(
          key,
        ),
    ),
  ) as T;
}
export function parseDecision(raw: unknown): DecisionView | null {
  const value = record(raw);
  if (!text(value.decision_id)) return null;
  const selected = safeFacts(value.selected),
    pipeline = record(selected.pipeline);
  const methods = Array.isArray(pipeline.preprocessing)
    ? pipeline.preprocessing.map((step) => text(record(step).method, 60) || "—")
    : null;
  const model = text(pipeline.model_method, 50).toUpperCase();
  return {
    decision_id: text(value.decision_id),
    context_id: text(value.context_id),
    experiment_id: text(value.experiment_id) || null,
    round_index: number(value.round_index),
    action: text(value.action),
    status: text(value.status),
    decision_time: text(value.decision_time),
    rationale: text(value.rationale),
    expected_outcome: text(value.expected_outcome),
    pipelineLabel: model
      ? [
          ...(methods === null ? ["—"] : methods.length ? methods : ["Raw"]),
          model,
        ].join(" → ")
      : "—",
    selected,
    origin: safeFacts(value.origin),
    ...(value.proposal ? { proposal: safeFacts(value.proposal) } : {}),
    ...(value.accepted_spec
      ? { accepted_spec: safeFacts(value.accepted_spec) }
      : {}),
    ...(value.feedback
      ? {
          feedback: {
            ...safeFacts(value.feedback),
            metrics:
              value.status === "succeeded"
                ? scoped(record(value.feedback).metrics, false)
                : {},
          },
        }
      : {}),
  };
}
export const autonomousDecisionPageSize = 10;
export function parseDecisionPage(raw: unknown, offset: number) {
  const value = record(raw);
  const total = number(value.total);
  const next = number(value.next_offset);
  return {
    data: Array.isArray(value.data)
      ? value.data
          .slice(0, autonomousDecisionPageSize)
          .map(parseDecision)
          .filter((item): item is DecisionView => item !== null)
      : [],
    total:
      total !== null && Number.isInteger(total) && total >= 0 ? total : null,
    next_offset:
      next !== null && Number.isInteger(next) && next > offset ? next : null,
  };
}
export function parseRejectionPage(raw: unknown, offset: number) {
  const page = record(raw),
    next = number(page.next_offset);
  return {
    data: Array.isArray(page.data)
      ? page.data.slice(0, autonomousDecisionPageSize).map(safeFacts)
      : [],
    next_offset:
      next !== null && Number.isInteger(next) && next > offset ? next : null,
  };
}
export function parameterTrials(raw: unknown) {
  return Array.isArray(raw)
    ? raw.slice(0, 100).flatMap((value, index) => {
        const trial = record(value);
        return trial.scope === "calibration_cv"
          ? [
              {
                index: index + 1,
                parameters: safeFacts(trial.parameters),
                RMSECV: number(trial.RMSECV),
                fold_rmse: Array.isArray(trial.fold_rmse)
                  ? trial.fold_rmse.slice(0, 100).map(number)
                  : [],
                scope: "calibration_cv" as const,
              },
            ]
          : [];
      })
    : [];
}

export function parseFinalChart(raw: unknown, stage: string) {
  const value = record(raw);
  const arrays = [value.observed, value.predicted, value.residual];
  if (
    stage !== "finalized" ||
    value.scope !== "final_test" ||
    value.evidence_status !== "available" ||
    !arrays.every(
      (item) =>
        Array.isArray(item) &&
        item.length > 0 &&
        item.length <= 2000 &&
        item.every(
          (point) => typeof point === "number" && Number.isFinite(point),
        ),
    )
  )
    return null;
  const observed = value.observed as number[],
    predicted = value.predicted as number[],
    residual = value.residual as number[];
  if (
    observed.length !== predicted.length ||
    observed.length !== residual.length ||
    !Array.isArray(value.sample_ids) ||
    value.sample_ids.length !== observed.length ||
    !value.sample_ids.every((id) => text(id, 200) === id && Boolean(id)) ||
    !Number.isInteger(value.n_total) ||
    Number(value.n_total) < observed.length ||
    value.n_returned !== observed.length
  )
    return null;
  return {
    observed,
    predicted,
    residual,
    sample_ids: value.sample_ids as string[],
    n_total: value.n_total as number,
    sampling: text(value.sampling),
  };
}
function scoped(value: unknown, final: boolean): Facts {
  const metrics = record(value),
    result: Facts = {};
  for (const [key, scope] of [
    ["cal", "calibration"],
    ["cal_cv", "calibration_cv"],
    ["val", "validation"],
    ["test", "final_test"],
  ]) {
    const item = record(metrics[key!]);
    if (item.scope === scope && (key !== "test" || final))
      result[key!] = Object.fromEntries(
        ["R2", "RMSE", "MAE", "RPD", "bias", "n", "scope"]
          .filter((name) => name === "scope" || number(item[name]) !== null)
          .map((name) => [name, name === "scope" ? scope : number(item[name])]),
      );
  }
  return result;
}
export function parseSummary(raw: unknown): AutonomousSummary | null {
  const value = record(raw);
  if (value.schema_version !== 1 || !text(value.session_id)) return null;
  const final = value.stage === "finalized";
  return {
    schema_version: 1,
    session_id: text(value.session_id),
    revision: text(value.revision),
    stage: text(value.stage),
    test_status: text(value.test_status),
    autonomous_delivery: value.autonomous_delivery === true,
    decision_count: number(value.decision_count),
    task: safeFacts(value.task),
    budget: safeFacts(value.budget),
    stop_source: text(value.stop_source),
    stop_reason: facts(value.stop_reason),
    retained_selection_source: text(value.retained_selection_source),
    retained_experiment_id: text(value.retained_experiment_id) || null,
    best_val_experiment_id: text(value.best_val_experiment_id) || null,
    terminal_failure_source: text(value.terminal_failure_source),
    terminal_failure_reason: facts(value.terminal_failure_reason),
    freeze:
      Object.keys(safeFacts(value.freeze)).length > 0
        ? safeFacts(value.freeze)
        : null,
    metrics: scoped(value.metrics, final),
    quality: safeFacts(value.quality),
    lifecycle_at_delivery: safeFacts(value.lifecycle_at_delivery),
    export_status: text(value.export_status),
    automatic_delivery: safeFacts(value.automatic_delivery),
    downloads:
      final && Array.isArray(value.downloads)
        ? value.downloads.flatMap((raw) => {
            const item = record(raw),
              path = typeof item.path === "string" ? item.path : "";
            return /^mnt\/user-data\/outputs\/[A-Za-z0-9_./-]+$/.test(path) &&
              !path.split("/").includes("..") &&
              [
                "model.joblib",
                "report.md",
                "report.html",
                "delivery.zip",
                "metrics.json",
                "decisions.json",
              ].includes(String(item.name))
              ? [{ name: String(item.name), path }]
              : [];
          })
        : [],
    trends: Array.isArray(value.trends)
      ? value.trends.slice(0, 1000).map((raw) => {
          const item = record(raw);
          return {
            experiment_id: text(item.experiment_id),
            decision_id: text(item.decision_id),
            round_index: number(item.round_index),
            method: text(item.method),
            metrics: scoped(item.metrics, false),
          };
        })
      : [],
    chart_scopes:
      final && Array.isArray(value.chart_scopes)
        ? value.chart_scopes.filter((item) => item === "final_test")
        : [],
    spectra_experiments: Array.isArray(value.spectra_experiments)
      ? value.spectra_experiments.slice(0, 100).flatMap((raw) => {
          const item = record(raw);
          return /^[A-Za-z0-9_-]{1,128}$/.test(text(item.experiment_id))
            ? [
                {
                  experiment_id: text(item.experiment_id),
                  decision_id: text(item.decision_id),
                  round_index: number(item.round_index),
                  method: text(item.method),
                },
              ]
            : [];
        })
      : [],
  };
}

export interface CalSpectra {
  configuration: Facts;
  decision_id: string;
  before: ProcessChart;
  after: ProcessChart;
}
export function parseCalSpectra(
  raw: unknown,
  sessionId: string,
  experimentId: string,
): CalSpectra | null {
  const value = record(raw);
  if (
    value.schema_version !== 1 ||
    value.evidence_status !== "available" ||
    value.scope !== "calibration" ||
    value.session_id !== sessionId ||
    value.experiment_id !== experimentId ||
    value.source !== "actual_cal_fitted_preprocessing_state"
  )
    return null;
  const decode = (rawChart: unknown, scope: string): ProcessChart | null => {
    const chart = record(rawChart);
    const x = chart.x;
    if (
      chart.type !== "spectra" ||
      chart.scope !== scope ||
      !Array.isArray(x) ||
      x.length < 2 ||
      x.length > 512 ||
      !x.every((v) => typeof v === "number" && Number.isFinite(v)) ||
      !Array.isArray(chart.series) ||
      !chart.series.length ||
      chart.series.length > 12
    )
      return null;
    const series = chart.series.map((item) => record(item));
    if (
      !Number.isInteger(chart.n_total) ||
      Number(chart.n_total) < series.length ||
      chart.n_returned !== series.length ||
      !Number.isInteger(chart.axis_total) ||
      Number(chart.axis_total) < x.length ||
      chart.axis_returned !== x.length ||
      !series.every(
        (item) =>
          Array.isArray(item.values) &&
          item.values.length === x.length &&
          item.values.every((v) => typeof v === "number" && Number.isFinite(v)),
      )
    )
      return null;
    const direction = x[1] > x[0] ? 1 : -1;
    if (!x.slice(1).every((v, i) => (v - x[i]) * direction > 0)) return null;
    return {
      chart_schema_version: 1,
      chart_id: text(chart.chart_id),
      run_id: "",
      attempt_id: experimentId,
      step_execution_id: text(value.decision_id),
      evidence_status: "available",
      type: "spectra",
      scope,
      x_label: chart.x_label === "wavelength" ? "wavelength" : "variable",
      y_label: "spectral_value",
      x_unit: ["nm", "cm^-1"].includes(String(chart.x_unit))
        ? String(chart.x_unit)
        : null,
      x,
      series: series.map((item) => ({
        id: text(item.id, 24),
        values: item.values as number[],
      })),
      n_total: number(chart.n_total) ?? 0,
      n_returned: series.length,
      axis_total: number(chart.axis_total) ?? x.length,
      axis_returned: x.length,
      sampling: text(chart.sampling),
    };
  };
  const before = decode(value.before, "calibration_before"),
    after = decode(value.after, "calibration_after");
  if (
    !before ||
    !after ||
    before.x_unit !== after.x_unit ||
    before.n_total !== after.n_total ||
    before.axis_total !== after.axis_total ||
    JSON.stringify(before.x) !== JSON.stringify(after.x) ||
    JSON.stringify(before.series?.map((item) => item.id)) !==
      JSON.stringify(after.series?.map((item) => item.id))
  )
    return null;
  return {
    configuration: safeFacts(value.configuration),
    decision_id: text(value.decision_id),
    before,
    after,
  };
}
export function validationSeries(
  summary: AutonomousSummary,
  metric: "RMSE" | "R2",
) {
  return summary.trends.flatMap((item) => {
    const val = record(item.metrics.val);
    return typeof val[metric] === "number" &&
      Number.isFinite(val[metric]) &&
      typeof item.round_index === "number" &&
      Number.isFinite(item.round_index) &&
      val.scope === "validation"
      ? [{ ...item, round_index: item.round_index, value: val[metric] }]
      : [];
  });
}
export function metricRows(raw: unknown) {
  const value = record(raw),
    metrics = scoped(value.metrics, value.stage === "finalized");
  return Object.entries(metrics).map(([scope, values]) => ({
    label: { cal: "Cal", cal_cv: "Cal CV", val: "Val", test: "Test" }[scope],
    values: record(values),
  }));
}
export function experimentMarkers(
  summary: AutonomousSummary,
  experimentId: string,
  zh: boolean,
) {
  const labels: string[] = [];
  if (experimentId === summary.best_val_experiment_id)
    labels.push(zh ? "当前最佳 Val RMSE" : "Best Val RMSE");
  if (experimentId === summary.retained_experiment_id)
    labels.push(
      summary.retained_selection_source === "llm"
        ? zh
          ? "LLM 已保留"
          : "LLM retained"
        : zh
          ? "已记录保留"
          : "Recorded retention",
    );
  return labels;
}
export function autonomousLifecycleFacts(
  summary: AutonomousSummary,
  workflow: unknown,
) {
  const raw = record(workflow),
    value = raw.search_session_id === summary.session_id ? raw : {};
  return {
    quality: summary.quality,
    user_approval_status:
      text(value.approval_status) || "not_recorded_in_session",
    registered_model_id: text(value.registered_model_id) || null,
    lifecycle_at_delivery: summary.lifecycle_at_delivery,
  };
}
export function autonomousTestStatusLabel(
  summary: AutonomousSummary,
  zh: boolean,
): string {
  if (summary.test_status === "sealed") return zh ? "封存" : "Sealed";
  if (summary.test_status === "consumed") {
    if (summary.stage === "finalized")
      return zh ? "已评价一次" : "Evaluated once";
    if (summary.stage === "final_test_failed")
      return zh
        ? "单次评价已消耗，Test 失败"
        : "Single use consumed, Test failed";
    return zh ? "已获得单次评价租约" : "Single-use lease acquired";
  }
  return zh ? "未记录" : "Not recorded";
}
export function actionLabel(value: string, zh: boolean) {
  return (
    {
      EXPLORE: ["探索新方向", "Explore"],
      TUNE: ["局部调参", "Tune"],
      CHANGE_PREPROCESS: ["调整预处理", "Change preprocessing"],
      SWITCH_MODEL: ["切换模型", "Switch model"],
      STOP: ["LLM 主动停止", "LLM STOP"],
      RETAIN: ["预算终止后保留", "Budget RETAIN"],
    }[value]?.[zh ? 0 : 1] ?? value
  );
}
export function executionLabel(value: string, zh: boolean) {
  return (
    {
      accepted: ["已接受，待执行", "Accepted, pending execution"],
      executing: ["正在训练", "Training"],
      succeeded: ["实际执行成功", "Execution succeeded"],
      failed: ["执行失败", "Execution failed"],
      stopped: ["已记录保留选择，无新增拟合", "Retention recorded, zero fits"],
      rejected: ["提案被拒绝，未执行", "Rejected, not executed"],
    }[value]?.[zh ? 0 : 1] ?? value
  );
}
export function isAutonomousTerminal(stage: unknown): boolean {
  return (
    typeof stage === "string" &&
    [
      "finalized",
      "provider_failed",
      "cancelled",
      "no_deliverable_model",
      "final_test_failed",
      "decision_invalid",
      "safety_closed",
      "finalization_unavailable",
      "stopped_pending_confirmation",
    ].includes(stage)
  );
}
export const autonomousPublicationPollingLimitMs = 300_000;
export function autonomousPollingInterval(
  raw: unknown,
  publicationElapsedMs = 0,
): number | false | null {
  const value = record(raw);
  if (
    value.stage === "finalized" &&
    value.autonomous_delivery === true &&
    !["succeeded", "failed"].includes(text(value.export_status))
  )
    return publicationElapsedMs < autonomousPublicationPollingLimitMs
      ? 5000
      : false;
  return isAutonomousTerminal(value.stage) ? false : null;
}
export function stopExplanation(raw: unknown, zh: boolean) {
  const value = record(raw);
  if (value.stop_source === "budget")
    return zh
      ? `资源预算终止搜索；${value.retained_selection_source === "llm" && text(value.retained_experiment_id) ? "LLM 已明确保留实验" : isAutonomousTerminal(value.stage) ? "尚未记录 LLM 保留实验" : "等待 LLM 选择保留实验"}。`
      : "Search ended by budget; retention is a separate LLM choice.";
  if (value.stop_source === "llm")
    return zh
      ? `LLM 根据实验反馈主动停止；${text(value.retained_experiment_id) ? "已指定保留实验" : "尚未记录保留实验"}。`
      : "LLM stopped from feedback and selected a retained experiment.";
  return zh
    ? "尚无正常停止记录；安全终止、失败与人工取消会单独显示。"
    : "No normal stop recorded; safety failures and cancellation are separate.";
}

export function autonomousIntakeStatus(raw: unknown, zh: boolean) {
  const workflow = record(raw);
  const evidence = record(workflow.ingestion_evidence);
  const failed = workflow.stage === "blocked";
  const ingestionFailed =
    failed &&
    evidence.source_tool === "nir_prepare_search" &&
    evidence.status === "failed";
  const code = text(evidence.code);
  return {
    failed,
    code: ingestionFailed && /^nir_ingestion_[a-z_]+$/.test(code) ? code : null,
    message: ingestionFailed
      ? zh
        ? "数据摄取失败，尚未建立封存搜索会话。请检查明确的目标列、文件格式和数值完整性；不要重复提交未更正的数据。"
        : "Ingestion failed before a sealed search session was created. Check the explicit target, format and numeric integrity before retrying."
      : failed
        ? zh
          ? "任务已被阻塞，尚无可引用的搜索会话或实验指标。请查看聊天中记录的失败原因。"
          : "The task is blocked; no search session or experiment metrics are available. Inspect the recorded failure in chat."
        : zh
          ? "正在确认数据和任务，等待封存搜索会话。目标列或波长含义不明确时，请在聊天中补充。"
          : "Waiting for a sealed session. Clarify target or wavelength metadata in chat when needed.",
  };
}
