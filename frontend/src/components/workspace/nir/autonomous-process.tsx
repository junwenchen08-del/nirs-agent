"use client";

import { useEffect, useState } from "react";

import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import {
  actionLabel,
  decisionPipelineLabel,
  autonomousIntakeStatus,
  autonomousDecisionPageSize,
  autonomousLifecycleFacts,
  autonomousTestStatusLabel,
  executionLabel,
  experimentMarkers,
  isAutonomousTerminal,
  metricRows,
  parameterTrials,
  parseDecision,
  parseDecisionPage,
  parseFinalChart,
  parseRejectionPage,
  parseSummary,
  record,
  stopExplanation,
  text,
  validationSeries,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { type ProcessChart } from "@/core/nir/process-contract";
import {
  factValue,
  formatProcessNumber as fmt,
} from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";

import { CalSpectraComparison } from "./cal-spectra";
import { ProcessChartView, plotLabel } from "./process-chart";
import { ProcessFacts } from "./process-facts";

const stageLabels: Record<string, [string, string]> = {
  searching: ["Cal/Val 搜索中", "Cal/Val search"],
  budget_exhausted: [
    "搜索预算耗尽，等待保留选择",
    "Budget exhausted, awaiting retention",
  ],
  stopped: ["停止选择已记录，等待冻结", "Choice recorded, awaiting freeze"],
  frozen: ["模型已冻结，等待独立 Test", "Frozen, awaiting independent Test"],
  testing: ["独立 Test 评价中", "Independent Test in progress"],
  finalized: ["独立 Test 已完成", "Independent Test completed"],
  final_test_failed: [
    "Test 评价失败，禁止自动重试",
    "Test failed, no automatic replay",
  ],
  provider_failed: ["模型服务失败", "Provider failure"],
  cancelled: ["人工取消", "Cancelled"],
  no_deliverable_model: ["没有可交付模型", "No deliverable model"],
  decision_invalid: ["提案修正次数耗尽", "Proposal repairs exhausted"],
  safety_closed: [
    "安全终止，查看已记录原因",
    "Safety stop; inspect recorded cause",
  ],
  finalization_unavailable: [
    "终结资源不可用",
    "Finalization resources unavailable",
  ],
  stopped_pending_confirmation: [
    "模型保留已记录，等待确认",
    "Retention recorded, awaiting confirmation",
  ],
};

const deliveryLabels: Record<string, [string, string]> = {
  queued: ["正在确认最终模型", "Confirming retained model"],
  freezing: ["正在冻结模型", "Freezing model"],
  frozen: ["模型已冻结", "Model frozen"],
  final_test_pending: ["正在执行最终评价", "Running final evaluation"],
  final_test_completed: ["最终评价已完成", "Final evaluation completed"],
  publishing: ["正在生成报告与下载文件", "Building report and downloads"],
  delivered: ["已完成交付，等待用户审批", "Delivered, awaiting approval"],
  failed: [
    "自动交付失败，保留已完成证据",
    "Delivery failed; completed evidence preserved",
  ],
  audit_required: [
    "安全关闭，待人工审计",
    "Closed safely; operator audit required",
  ],
  cancelled: [
    "交付已取消，保留已有证据",
    "Delivery cancelled; evidence preserved",
  ],
};

function Metrics({ value }: { value: unknown }) {
  const { locale } = useI18n();
  const zh = locale === "zh-CN";
  const rows = metricRows(value);
  return rows.length ? (
    <div className="grid gap-3 md:grid-cols-2">
      {rows.map((row) => (
        <div key={row.label} className="rounded border p-3 text-sm">
          <strong>
            {zh
              ? ({
                  Cal: "校准集拟合（Cal）",
                  "Cal CV": "校准交叉验证（Cal CV，折外预测）",
                  Val: "验证集（Val）",
                  Test: "最终测试集（Test）",
                }[row.label ?? ""] ?? row.label)
              : row.label}
            {!zh && row.label === "Cal CV" ? " (OOF)" : ""}
          </strong>
          <span className="text-muted-foreground ml-2 text-xs">
            {String(factValue("scope", row.values.scope, zh))} ·{" "}
            {zh ? "样本数" : "n"}={fmt(row.values.n)}
          </span>
          <div className="mt-2 flex gap-4">
            {["RMSE", "R2", "RPD"].map((key) => (
              <span key={key}>
                {key === "R2" ? "R²" : key}:{" "}
                {key in row.values ? fmt(row.values[key]) : "—"}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  ) : null;
}

function RecordedNarrative({ value, zh }: { value: string; zh: boolean }) {
  if (!value) return <span>{zh ? "未记录" : "Not recorded"}</span>;
  if (!zh || /[\u3400-\u9fff]/u.test(value))
    return <span className="whitespace-pre-wrap">{value}</span>;
  return (
    <details className="text-sm">
      <summary className="text-muted-foreground cursor-pointer">
        历史理由为英文，展开查看原始记录
      </summary>
      <p className="mt-2 whitespace-pre-wrap" lang="en">
        {value}
      </p>
    </details>
  );
}

function EvidenceChart({
  x,
  y,
  type,
  scope,
  zh,
  ids,
  sampling = "persisted_evidence",
  nTotal,
}: {
  x: number[];
  y: number[];
  type: string;
  scope: string;
  zh: boolean;
  ids?: string[];
  sampling?: string;
  nTotal?: number;
}) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  if (
    !x.length ||
    x.length !== y.length ||
    ![...x, ...y].every(Number.isFinite)
  )
    return <p>{zh ? "图表证据不可用。" : "Chart evidence unavailable."}</p>;
  const chart: ProcessChart = {
    chart_schema_version: 1,
    chart_id: `${type}_${scope}`,
    run_id: "",
    attempt_id: "",
    step_execution_id: "",
    evidence_status: "available",
    type,
    scope,
    x,
    y,
    x_label:
      type === "trend"
        ? "round"
        : type === "cv"
          ? "candidate_index"
          : "reference",
    y_label:
      type === "residual"
        ? "residual"
        : type === "cv"
          ? "RMSECV"
          : type === "trend"
            ? scope
            : "prediction",
    sample_ids: ids,
    n_total: nTotal ?? x.length,
    n_returned: x.length,
    sampling,
  };
  return (
    <ProcessChartView
      chart={chart}
      zh={zh}
      zoom={zoom}
      onZoom={setZoom}
      selected={selected}
      onSelect={setSelected}
    />
  );
}

function ParameterTrials({ value, zh }: { value: unknown; zh: boolean }) {
  const trials = parameterTrials(value);
  const [offset, setOffset] = useState(0);
  const page = trials.slice(offset, offset + autonomousDecisionPageSize);
  const points = page.filter(
    (trial): trial is typeof trial & { RMSECV: number } =>
      trial.RMSECV !== null,
  );
  return (
    <div className="mt-3 space-y-3">
      <p className="text-muted-foreground text-xs">
        {zh
          ? "只显示已记录的 Cal CV 参数评价；候选序号保留实际搜索顺序，折叠或翻页不会重新训练。"
          : "Recorded Cal CV evaluations only, in actual search order. Expanding or paging never retrains."}
      </p>
      {points.length > 0 && (
        <EvidenceChart
          x={points.map((trial) => trial.index)}
          y={points.map((trial) => trial.RMSECV)}
          type="cv"
          scope="calibration_cv"
          zh={zh}
          nTotal={trials.length}
        />
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              <th>{zh ? "候选序号" : "Candidate"}</th>
              <th>{zh ? "实际参数" : "Actual parameters"}</th>
              <th>{zh ? "Cal CV RMSE（折均值）" : "Cal CV mean-fold RMSE"}</th>
              <th>{zh ? "各折 RMSE" : "Fold RMSE"}</th>
            </tr>
          </thead>
          <tbody>
            {page.map((trial) => (
              <tr key={trial.index} className="border-t">
                <td className="py-2">{trial.index}</td>
                <td className="max-w-80 break-words">
                  <ProcessFacts value={trial.parameters} zh={zh} />
                </td>
                <td>{fmt(trial.RMSECV)}</td>
                <td>{trial.fold_rmse.map(fmt).join(" / ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!trials.length && (
        <p>
          {zh
            ? "没有可展示的 Cal CV 参数证据。"
            : "No Cal CV parameter evidence available."}
        </p>
      )}
      {trials.length > autonomousDecisionPageSize && (
        <div className="flex justify-between text-sm">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() =>
              setOffset(Math.max(0, offset - autonomousDecisionPageSize))
            }
          >
            {zh ? "上一页" : "Previous"}
          </button>
          <span>
            {offset + 1}–{offset + page.length} / {trials.length}
          </span>
          <button
            type="button"
            disabled={offset + page.length >= trials.length}
            onClick={() => setOffset(offset + autonomousDecisionPageSize)}
          >
            {zh ? "下一页" : "Next"}
          </button>
        </div>
      )}
    </div>
  );
}

function Trends({ summary, zh }: { summary: AutonomousSummary; zh: boolean }) {
  const [metric, setMetric] = useState<"RMSE" | "R2">("RMSE");
  const series = validationSeries(summary, metric);
  return (
    <section className="space-y-3 rounded-xl border p-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">
          {zh ? "实际实验的 Val 性能" : "Executed experiments: Val performance"}
        </h3>
        <select
          aria-label={zh ? "验证趋势指标" : "Validation trend metric"}
          value={metric}
          onChange={(event) => setMetric(event.target.value as "RMSE" | "R2")}
          className="bg-background rounded border p-1"
        >
          <option>RMSE</option>
          <option>R2</option>
        </select>
      </div>
      <p className="text-muted-foreground text-xs">
        {zh
          ? "仅包含同一封存会话中执行成功的实验；最低 Val RMSE 是信息标记，保留实验由 LLM 决定。"
          : "Only successful same-session experiments. The best Val RMSE marker is informational; LLM chooses retention."}
      </p>
      {series.length ? (
        <EvidenceChart
          key={metric}
          x={series.map((item) => item.round_index)}
          y={series.map((item) => item.value)}
          type="trend"
          scope={
            zh ? `验证集（Val）· ${plotLabel(metric, zh)}` : `Val ${metric}`
          }
          zh={zh}
        />
      ) : (
        <p className="text-muted-foreground text-sm">
          {zh
            ? "暂无可比较的实际验证结果。"
            : "No comparable executed validation evidence yet."}
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr>
              <th>{zh ? "轮次" : "Round"}</th>
              <th>{zh ? "实际模型" : "Actual model"}</th>
              <th>
                {zh ? `验证集 · ${plotLabel(metric, zh)}` : `Val ${metric}`}
              </th>
              <th>{zh ? "标记" : "Markers"}</th>
            </tr>
          </thead>
          <tbody>
            {series.map((item) => (
              <tr key={item.experiment_id} className="border-t">
                <td className="py-2">{item.round_index}</td>
                <td>{item.method.toUpperCase()}</td>
                <td>{fmt(item.value)}</td>
                <td>
                  {experimentMarkers(summary, item.experiment_id, zh).join(
                    " · ",
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DecisionDetails({
  threadId,
  sid,
  decision,
  zh,
  poll,
}: {
  threadId: string;
  sid: string;
  decision: DecisionView;
  zh: boolean;
  poll: boolean;
}) {
  const data = useProcessResource<unknown>(
    threadId,
    `autonomous/${sid}/decisions/${decision.decision_id}`,
    poll && ["accepted", "executing"].includes(decision.status),
  );
  const [showTrials, setShowTrials] = useState(false);
  const details = parseDecision(data.data);
  if (data.isError)
    return (
      <p role="alert">
        {zh
          ? "详细证据读取失败，可关闭后重试；不会重新训练。"
          : "Evidence read failed; reopening retries the read only."}
      </p>
    );
  if (!details)
    return (
      <p>
        {data.isPending
          ? zh
            ? "正在读取持久决策证据…"
            : "Reading persisted evidence…"
          : zh
            ? "详细决策证据缺失或格式不可用。"
            : "Detailed evidence is missing or invalid."}
      </p>
    );
  const proposal = details.proposal ?? {},
    feedback = details.feedback ?? {};
  const trials = Array.isArray(feedback.trials) ? feedback.trials : [];
  return (
    <div className="mt-3 space-y-4">
      <div>
        <h4 className="font-medium">
          {zh
            ? "主要观察、不确定性与假设"
            : "Observations, uncertainty and hypotheses"}
        </h4>
        <ProcessFacts
          zh={zh}
          value={{
            observations: proposal.observations,
            uncertainties: proposal.uncertainties,
            hypotheses: proposal.hypotheses,
            evidence_refs: proposal.evidence_refs,
          }}
        />
      </div>
      <div>
        <h4 className="font-medium">
          {zh
            ? "LLM 比较的候选动作（提出不代表执行）"
            : "LLM alternatives (proposals are not executions)"}
        </h4>
        <ProcessFacts
          zh={zh}
          value={{
            alternatives: proposal.alternatives,
            selected_alternative_id: proposal.selected_alternative_id,
            not_selected_reasons: proposal.not_selected_reasons,
          }}
        />
      </div>
      <div>
        <h4 className="font-medium">
          {zh
            ? "已接受配置与参数授权"
            : "Accepted configuration and parameter authorization"}
        </h4>
        <ProcessFacts
          zh={zh}
          value={{
            accepted_spec: details.accepted_spec,
            expected_fits: proposal.expected_fits,
            expected_improvement: proposal.expected_improvement,
            stop_evidence: proposal.stop_evidence,
            retained_experiment_id: proposal.retained_experiment_id,
          }}
        />
      </div>
      {details.feedback && (
        <div>
          <h4 className="font-medium">
            {zh
              ? "实际执行与预期效果对照"
              : "Execution and expectation feedback"}
          </h4>
          <Metrics value={{ metrics: feedback.metrics, stage: "searching" }} />
          <ProcessFacts
            zh={zh}
            value={{
              actual_configuration: feedback.actual_configuration,
              actual_parameters: feedback.actual_parameters,
              comparisons: feedback.comparisons,
              expected_improvement: feedback.expected_improvement,
              fits: feedback.fits,
              elapsed_seconds: feedback.elapsed_seconds,
              failure_reason: feedback.failure_reason,
              error_type: feedback.error_type,
              cv_settings: feedback.cv_settings,
            }}
          />
        </div>
      )}
      {trials.length > 0 && (
        <details onToggle={(event) => setShowTrials(event.currentTarget.open)}>
          <summary className="cursor-pointer text-sm">
            {zh
              ? `实际参数候选的 Cal CV 结果（${trials.length}）`
              : `Actual Cal CV parameter trials (${trials.length})`}
          </summary>
          {showTrials && <ParameterTrials zh={zh} value={trials} />}
        </details>
      )}
      <details>
        <summary className="cursor-pointer text-xs">
          {zh
            ? "决策与实验身份 / 调用来源"
            : "Decision/experiment identity and model origin"}
        </summary>
        <ProcessFacts
          zh={zh}
          value={{
            decision_id: details.decision_id,
            context_id: details.context_id,
            experiment_id: details.experiment_id,
            origin: details.origin,
          }}
        />
      </details>
    </div>
  );
}

function FinalCharts({
  threadId,
  summary,
  zh,
}: {
  threadId: string;
  summary: AutonomousSummary;
  zh: boolean;
}) {
  const query = useProcessResource<unknown>(
    threadId,
    summary.stage === "finalized" && summary.chart_scopes.includes("final_test")
      ? `autonomous/${summary.session_id}/charts/final_test`
      : null,
  );
  if (query.isError)
    return (
      <p role="alert">
        {zh
          ? "最终 Test 图表证据读取失败；重新打开图表可重试读取。"
          : "Final Test plot read failed; reopen charts to retry the read."}
      </p>
    );
  if (query.isPending && summary.chart_scopes.includes("final_test"))
    return (
      <p>
        {zh
          ? "正在读取最终 Test 图表证据…"
          : "Reading final Test plot evidence…"}
      </p>
    );
  const value = parseFinalChart(query.data, summary.stage);
  if (!value)
    return (
      <p className="text-muted-foreground text-sm">
        {zh
          ? "最终 Test 图表尚未记录；封存期间不可读取。旧运行不补算图表。"
          : "Final Test plots not recorded. Sealed Test cannot be read; old runs are not recomputed."}
      </p>
    );
  return (
    <div className="grid gap-3 xl:grid-cols-2">
      <EvidenceChart
        x={value.observed}
        y={value.predicted}
        type="prediction"
        scope="final_test"
        zh={zh}
        ids={value.sample_ids}
        sampling={value.sampling}
        nTotal={value.n_total}
      />
      <EvidenceChart
        x={value.observed}
        y={value.residual}
        type="residual"
        scope="final_test"
        zh={zh}
        ids={value.sample_ids}
        sampling={value.sampling}
        nTotal={value.n_total}
      />
    </div>
  );
}

function Rejections({
  threadId,
  sid,
  zh,
  poll,
}: {
  threadId: string;
  sid: string;
  zh: boolean;
  poll: boolean;
}) {
  const [offset, setOffset] = useState(0);
  const query = useProcessResource<{
    data: unknown[];
    next_offset: number | null;
  }>(threadId, `autonomous/${sid}/rejections?offset=${offset}`, poll);
  const page = parseRejectionPage(query.data, offset);
  return (
    <section className="rounded border p-3 text-sm">
      <h4 className="font-medium">
        {zh ? "被拒绝的提案（未执行）" : "Rejected proposals (not executed)"}
      </h4>
      {query.isError ? (
        <p role="alert">
          {zh
            ? "提案校验记录读取失败，请关闭后重新读取。"
            : "Proposal validation records failed to load; reopen to retry."}
        </p>
      ) : query.isPending ? (
        <p>
          {zh
            ? "正在读取提案校验记录…"
            : "Reading proposal validation records…"}
        </p>
      ) : page.data.length ? (
        page.data.map((raw, index) => (
          <div key={index} className="mt-2 border-t pt-2">
            <ProcessFacts zh={zh} value={record(raw)} />
          </div>
        ))
      ) : (
        <p className="text-muted-foreground mt-2">
          {zh
            ? "未记录提案拒绝事件。旧运行不补造记录。"
            : "No recorded rejection events. Old runs are not reconstructed."}
        </p>
      )}
      <div className="mt-2 flex justify-between">
        <button
          type="button"
          disabled={!offset}
          onClick={() => setOffset(Math.max(0, offset - 10))}
        >
          {zh ? "上一页" : "Previous"}
        </button>
        <button
          type="button"
          disabled={page.next_offset == null}
          onClick={() => {
            if (page.next_offset !== null) setOffset(page.next_offset);
          }}
        >
          {zh ? "下一页" : "Next"}
        </button>
      </div>
    </section>
  );
}

export function AutonomousProcess({
  threadId,
  sessionId,
  workflow,
}: {
  threadId: string;
  sessionId?: string;
  workflow?: unknown;
}) {
  const { locale } = useI18n(),
    zh = locale === "zh-CN";
  const sessions = useProcessResource<{ data: { session_id: string }[] }>(
    threadId,
    sessionId ? null : "autonomous",
  );
  const sid = sessionId ?? sessions.data?.data[0]?.session_id;
  const summaryQuery = useProcessResource<unknown>(
    threadId,
    sid ? `autonomous/${sid}` : null,
    true,
  );
  const summary = parseSummary(summaryQuery.data);
  const [offset, setOffset] = useState(0),
    [expanded, setExpanded] = useState<string | null>(null),
    [showRejections, setShowRejections] = useState(false);
  useEffect(() => {
    setOffset(0);
    setExpanded(null);
    setShowRejections(false);
  }, [threadId, sid]);
  const timeline = useProcessResource<{
    data: unknown[];
    total: number;
    next_offset: number | null;
  }>(
    threadId,
    sid
      ? `autonomous/${sid}/decisions?offset=${offset}&limit=${autonomousDecisionPageSize}`
      : null,
    !summary || !isAutonomousTerminal(summary.stage),
  );
  const page = parseDecisionPage(timeline.data, offset),
    rows = page.data;
  const [tab, setTab] = useState<"timeline" | "charts" | "delivery">(
    "timeline",
  );
  if (summaryQuery.isError || sessions.isError)
    return (
      <div className="p-4" role="alert">
        <p>
          {zh
            ? "自主建模证据读取失败或没有访问权限。"
            : "Autonomous evidence unavailable or access denied."}
        </p>
        <button
          type="button"
          onClick={() => {
            void summaryQuery.refetch();
            void sessions.refetch();
          }}
        >
          {zh ? "重新读取" : "Retry read"}
        </button>
      </div>
    );
  if (!summary || !sid) {
    const intake = autonomousIntakeStatus(workflow, zh);
    return (
      <div className="p-4" role={intake.failed ? "alert" : "status"}>
        <p>{intake.message}</p>
        {intake.code && (
          <p className="text-muted-foreground mt-2 text-xs">{intake.code}</p>
        )}
      </div>
    );
  }
  const limits = record(summary.budget.limits);
  const deliveryStage = text(record(summary.automatic_delivery).status);
  return (
    <div
      className="min-h-0 space-y-4 overflow-y-auto p-4"
      data-testid="autonomous-process"
    >
      <section className="rounded-xl border p-4">
        <h2 className="text-lg font-semibold">
          {zh ? "LLM 自主建模" : "LLM autonomous modeling"}
        </h2>
        <p className="mt-1 text-sm" role="status">
          {deliveryLabels[deliveryStage]?.[zh ? 0 : 1] ??
            stageLabels[summary.stage]?.[zh ? 0 : 1] ??
            summary.stage}
        </p>
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {[
            [
              zh ? "Cal/Val 搜索" : "Cal/Val search",
              `${fmt(summary.budget.experiments_used)}/${fmt(limits.max_experiments)}`,
            ],
            [
              zh ? "模型拟合" : "Model fits",
              `${fmt(summary.budget.fits_used)}/${fmt(limits.max_fits)}`,
            ],
            [
              zh ? "高层决策" : "Decisions",
              `${fmt(summary.budget.actions_used)}/${fmt(limits.max_decisions)}`,
            ],
            ["Test", autonomousTestStatusLabel(summary, zh)],
            [
              zh ? "模型调用" : "Model calls",
              fmt(
                Number(record(record(summary.budget.llm).search).calls ?? 0) +
                  Number(
                    record(record(summary.budget.llm).finalization).calls ?? 0,
                  ),
              ),
            ],
            [
              zh ? "冻结" : "Freeze",
              summary.freeze
                ? zh
                  ? "已冻结"
                  : "Frozen"
                : zh
                  ? "尚未冻结"
                  : "Pending",
            ],
            [
              zh ? "交付" : "Delivery",
              summary.export_status === "succeeded"
                ? zh
                  ? "已发布"
                  : "Published"
                : summary.export_status || "—",
            ],
          ].map(([label, value]) => (
            <span key={label} className="rounded border px-2 py-1">
              {label}: {value}
            </span>
          ))}
        </div>
        <p className="mt-3 text-sm">{stopExplanation(summary, zh)}</p>
        {(summary.stop_source || summary.terminal_failure_source) && (
          <details className="mt-2 text-xs">
            <summary>
              {zh ? "停止与资源依据" : "Stop and resource evidence"}
            </summary>
            <ProcessFacts
              zh={zh}
              value={{
                stop_source: summary.stop_source,
                stop_reason: summary.stop_reason,
                retained_experiment_id: summary.retained_experiment_id,
                terminal_failure_source: summary.terminal_failure_source,
                terminal_failure_reason: summary.terminal_failure_reason,
                budget: summary.budget,
              }}
            />
          </details>
        )}
        <p className="text-muted-foreground mt-2 text-xs">
          {summary.spectra_experiments.length
            ? zh
              ? "Cal 原始与预处理光谱已记录，可在“验证与图表”中查看。"
              : "Raw and preprocessed Cal spectra are available in Validation and charts."
            : zh
              ? "本运行未记录 Cal 光谱曲线；页面不读取原始文件或 Test 来补画。"
              : "No Cal spectra recorded for this run. The page does not reopen raw data or Test."}
        </p>
      </section>
      <nav
        className="flex gap-2"
        aria-label={zh ? "自主建模视图" : "Autonomous views"}
      >
        {(["timeline", "charts", "delivery"] as const).map((key) => (
          <button
            type="button"
            key={key}
            className={`rounded-lg border px-3 py-2 text-sm ${tab === key ? "bg-accent font-semibold" : ""}`}
            aria-pressed={tab === key}
            onClick={() => setTab(key)}
          >
            {
              {
                timeline: ["决策时间线", "Decision timeline"],
                charts: ["验证与图表", "Validation and charts"],
                delivery: ["最终模型与下载", "Model and downloads"],
              }[key][zh ? 0 : 1]
            }
          </button>
        ))}
      </nav>
      {tab === "timeline" && (
        <section className="space-y-3">
          <details
            open={showRejections}
            onToggle={(event) => setShowRejections(event.currentTarget.open)}
          >
            <summary className="cursor-pointer text-sm">
              {zh
                ? "查看提案校验错误与修正"
                : "Proposal validation errors and corrections"}
            </summary>
            {showRejections && (
              <Rejections
                threadId={threadId}
                sid={sid}
                zh={zh}
                poll={!isAutonomousTerminal(summary.stage)}
              />
            )}
          </details>
          {timeline.isError && (
            <p role="alert">
              {zh ? "决策列表读取失败，请刷新进展。" : "Timeline read failed."}
            </p>
          )}
          {!rows.length && (
            <p>
              {zh
                ? "尚无已接受决策；无训练结果时不显示模型排名。"
                : "No accepted decisions yet; no models are ranked without execution."}
            </p>
          )}
          {rows.map((item) => (
            <article key={item.decision_id} className="rounded-xl border p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-medium">
                  {zh ? "第" : "Round "}
                  {fmt(item.round_index)}
                  {zh ? "轮 · " : " · "}
                  {actionLabel(item.action, zh)}
                </h3>
                <span className="rounded border px-2 py-1 text-xs">
                  {executionLabel(item.status, zh)}
                </span>
              </div>
              <p className="mt-2 font-medium">
                {["STOP", "RETAIN"].includes(item.action)
                  ? zh
                    ? "保留已选实验，不新增拟合"
                    : "Retain selected experiment; no new fits"
                  : decisionPipelineLabel(item.selected, zh)}
              </p>
              <div className="mt-2 text-sm">
                <RecordedNarrative value={item.rationale} zh={zh} />
              </div>
              <div className="text-muted-foreground mt-2 text-xs">
                {zh ? "预期效果：" : "Expected: "}
                <RecordedNarrative value={item.expected_outcome} zh={zh} />
              </div>
              {item.feedback && (
                <>
                  <Metrics
                    value={{
                      stage: "searching",
                      metrics: item.feedback.metrics,
                    }}
                  />
                  <p className="mt-2 text-xs">
                    {zh ? "实际拟合 / 耗时：" : "Actual fits / seconds: "}
                    {fmt(item.feedback.fits)} /{" "}
                    {fmt(item.feedback.elapsed_seconds)}
                  </p>
                  {item.feedback.failure_reason !== undefined && (
                    <p role="alert" className="text-destructive text-sm">
                      {text(item.feedback.failure_reason)}
                    </p>
                  )}
                </>
              )}
              <button
                type="button"
                className="mt-3 text-sm underline"
                aria-expanded={expanded === item.decision_id}
                onClick={() =>
                  setExpanded(
                    expanded === item.decision_id ? null : item.decision_id,
                  )
                }
              >
                {zh
                  ? "查看观察、候选比较和实际配置"
                  : "Inspect observations, alternatives and configuration"}
              </button>
              {expanded === item.decision_id && (
                <DecisionDetails
                  threadId={threadId}
                  sid={sid}
                  decision={item}
                  zh={zh}
                  poll={!isAutonomousTerminal(summary.stage)}
                />
              )}
            </article>
          ))}
          <div className="flex items-center justify-between text-sm">
            <button
              type="button"
              disabled={!offset}
              onClick={() => {
                setOffset(Math.max(0, offset - autonomousDecisionPageSize));
                setExpanded(null);
              }}
            >
              {zh ? "上一页" : "Previous"}
            </button>
            <span>
              {offset + (rows.length ? 1 : 0)}–{offset + rows.length} /{" "}
              {fmt(page.total)}
            </span>
            <button
              type="button"
              disabled={page.next_offset == null}
              onClick={() => {
                if (page.next_offset !== null) setOffset(page.next_offset);
                setExpanded(null);
              }}
            >
              {zh ? "下一页" : "Next"}
            </button>
          </div>
        </section>
      )}
      {tab === "charts" && (
        <>
          <Trends summary={summary} zh={zh} />
          <Metrics value={summary} />
          <CalSpectraComparison threadId={threadId} summary={summary} zh={zh} />
          <FinalCharts threadId={threadId} summary={summary} zh={zh} />
        </>
      )}
      {tab === "delivery" && (
        <section className="space-y-4 rounded-xl border p-4">
          <h3 className="font-medium">
            {zh ? "最终保留模型" : "Final retained model"}
          </h3>
          {!summary.freeze ? (
            <p>
              {zh
                ? "尚无冻结模型；成功实验并不自动成为最终模型。"
                : "No frozen model; success does not automatically select the final model."}
            </p>
          ) : (
            <ProcessFacts zh={zh} value={summary.freeze} />
          )}
          <Metrics value={summary} />
          <p className="text-sm">
            {zh
              ? "科学协议、性能门禁、用户审批与注册分别记录。"
              : "Scientific protocol, quality, user approval and registration are separate."}
          </p>
          <ProcessFacts
            zh={zh}
            value={autonomousLifecycleFacts(summary, workflow)}
          />
          {summary.downloads.length ? (
            <div className="flex flex-wrap gap-2">
              {summary.downloads.map((file) => (
                <a
                  key={file.path}
                  href={urlOfArtifact({
                    threadId,
                    filepath: `/${file.path}`,
                    download: true,
                  })}
                  className="rounded-lg border px-3 py-2 text-sm"
                  download
                >
                  {zh
                    ? ({
                        "model.joblib": "预测模型（model.joblib）",
                        "report.md": "中文图文报告（Markdown）",
                        "report.html": "离线图文报告（HTML）",
                        "delivery.zip": "完整下载包（ZIP）",
                        "metrics.json": "完整指标证据（JSON）",
                        "decisions.json": "原始决策证据（JSON）",
                      }[file.name] ?? file.name)
                    : file.name}
                </a>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">
              {zh
                ? "尚无已发布的模型与报告文件。导出失败不自动重新运行 Test。"
                : "No published model/report files. Export failure never automatically reruns Test."}
            </p>
          )}
          <details>
            <summary className="cursor-pointer text-xs">
              {zh ? "任务与交付身份" : "Task and delivery identity"}
            </summary>
            <ProcessFacts
              zh={zh}
              value={{
                session_id: sid,
                task: summary.task,
                retained_experiment_id: summary.retained_experiment_id,
                best_val_experiment_id: summary.best_val_experiment_id,
              }}
            />
          </details>
        </section>
      )}
      <button
        type="button"
        className="rounded border px-3 py-2 text-sm"
        onClick={() => {
          void summaryQuery.refetch();
          void timeline.refetch();
        }}
      >
        {zh ? "刷新进展（只读）" : "Refresh progress (read-only)"}
      </button>
    </div>
  );
}
