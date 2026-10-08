"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeftIcon, RefreshCwIcon } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useAuth } from "@/core/auth/AuthProvider";
import { useI18n } from "@/core/i18n/hooks";
import { loadNIRWorkflowState } from "@/core/nir/api";
import { processCandidateLabel } from "@/core/nir/presentation";
import { ProcessAPIError } from "@/core/nir/process-api";
import {
  comparableScores,
  pipelineDifference,
  PROCESS_STEPS,
  type CandidatePage,
  type ChartRef,
  type ProcessChart,
  type ProcessStep,
  type ProcessSummary,
  type RunList,
  type StepKey,
} from "@/core/nir/process-contract";
import {
  candidateConfiguration,
  factLabel,
  factValue,
  formatProcessNumber,
  resolveProcessCandidate,
  selectionDecision,
} from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import { selectNIRProcessView } from "@/core/nir/process-selectors";
import { cn } from "@/lib/utils";

import { MethodKnowledge } from "./method-knowledge";
import { NIRProcessContent } from "./nir-process-content";
import { plotLabel, ProcessChartView } from "./process-chart";
import { ProcessFacts as Facts } from "./process-facts";

const stepCopy: Record<StepKey, [string, string, string, string]> = {
  audit: [
    "数据解释与审查",
    "Data inspection",
    "确认输入的数据结构、目标与光谱质量，为后续建模建立可信输入。",
    "Confirm data structure, target and spectral quality.",
  ],
  split: [
    "数据划分",
    "Data partitions",
    "明确哪些样本用于拟合、调参和独立评估，检查划分边界。",
    "Inspect the fitting, tuning and independent evaluation boundaries.",
  ],
  preprocessing: [
    "预处理",
    "Preprocessing",
    "查看实际处理顺序及同一校准样本处理前后的光谱变化。",
    "Inspect the recorded pipeline and the same calibration samples before and after processing.",
  ],
  wavelength: [
    "波长选择",
    "Wavelength selection",
    "查看保留的变量区间，以及选择方案相对于全谱基线的依据。",
    "Explore retained intervals and the recorded decision relative to the full-spectrum baseline.",
  ],
  model: [
    "模型与调参比较",
    "Model comparison",
    "查看实际运行的模型候选、参数和训练内评价分数。",
    "Explore executed candidates, parameters and training-only selection scores.",
  ],
  validation: [
    "验证与诊断",
    "Validation diagnostics",
    "通过预测及残差探索误差分布，并核对全量指标和质量门禁。",
    "Explore predictions and residuals alongside full-population metrics and quality gates.",
  ],
  reflection: [
    "反思与重试",
    "Reflection and retry",
    "查看为什么继续或停止，以及下一轮计划与实际执行的区别。",
    "Inspect recorded reasons to continue or stop and distinguish plans from execution.",
  ],
  registration: [
    "审批与注册",
    "Approval and registration",
    "核对审批、注册和入库各自的记录与证据。",
    "Inspect approval, registration and library records separately.",
  ],
};
const fmt = formatProcessNumber;
function statusText(value: string | undefined, zh: boolean) {
  const states: Record<string, [string, string]> = {
    pending: ["待执行", "Pending"],
    running: ["执行中", "Running"],
    waiting_user: ["等待确认", "Waiting for input"],
    succeeded: ["已完成", "Completed"],
    completed: ["已完成", "Completed"],
    failed: ["失败", "Failed"],
    cancelled: ["已取消", "Cancelled"],
    interrupted: ["已中断", "Interrupted"],
    skipped: ["已跳过", "Skipped"],
    unknown: ["状态未记录", "Unknown"],
    available: ["证据可用", "Evidence available"],
    partial: ["证据不完整", "Partial evidence"],
    not_recorded: ["当时未记录", "Not recorded"],
    not_applicable: ["不适用", "Not applicable"],
    unavailable: ["证据不可用", "Evidence unavailable"],
  };
  return (
    states[value ?? ""]?.[zh ? 0 : 1] ??
    value ??
    (zh ? "当时未记录" : "Not recorded")
  );
}

function ChartResource({
  threadId,
  runId,
  chart,
  zh,
  zoom,
  onZoom,
  selected,
  onSelect,
  scope,
}: {
  threadId: string;
  runId: string;
  chart: ChartRef;
  zh: boolean;
  zoom: [number, number];
  onZoom: (range: [number, number]) => void;
  selected: string[];
  onSelect: (ids: string[]) => void;
  scope?: string | null;
}) {
  const query = useProcessResource<ProcessChart>(
    threadId,
    `runs/${runId}/charts/${chart.chart_id}?version=${chart.version}`,
  );
  if (query.isPending)
    return (
      <div className="bg-muted/30 flex h-64 animate-pulse items-center justify-center rounded-xl border text-sm">
        {zh ? "正在读取图表证据…" : "Loading chart evidence…"}
      </div>
    );
  if (query.isError)
    return (
      <div className="rounded-xl border p-5 text-sm">
        {zh ? "图表读取失败" : "Unable to read chart"}
        <button className="ml-3 underline" onClick={() => void query.refetch()}>
          {zh ? "重试" : "Retry"}
        </button>
      </div>
    );
  if (query.data?.chart_id !== chart.chart_id) return null;
  if (
    scope &&
    query.data.evidence_status === "available" &&
    query.data.scope !== scope
  )
    return null;
  return (
    <ProcessChartView
      chart={query.data}
      zh={zh}
      zoom={zoom}
      onZoom={onZoom}
      selected={selected}
      onSelect={onSelect}
    />
  );
}

function StepDetail({
  threadId,
  runId,
  stepId,
  stepKey,
  zh,
  previousStepId,
}: {
  threadId: string;
  runId: string;
  stepId: string;
  stepKey: StepKey;
  zh: boolean;
  previousStepId?: string;
}) {
  const [zoom, setZoom] = useState<[number, number]>([0, 1]);
  const [selected, setSelected] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);
  const [candidateIndex, setCandidateIndex] = useState<number | null>(null);
  const [scope, setScope] = useState("holdout");
  const [beforeAfter, setBeforeAfter] = useState("both");
  const detail = useProcessResource<ProcessStep>(
    threadId,
    `runs/${runId}/steps/${stepId}`,
  );
  const previous = useProcessResource<ProcessStep>(
    threadId,
    previousStepId ? `runs/${runId}/steps/${previousStepId}` : null,
  );
  const candidates = useProcessResource<CandidatePage>(
    threadId,
    detail.data?.candidate_count
      ? `runs/${runId}/steps/${stepId}/candidates?offset=${offset}`
      : null,
  );
  if (detail.isPending)
    return (
      <div className="p-8 text-sm">
        {zh ? "正在读取步骤…" : "Loading step…"}
      </div>
    );
  if (detail.isError || !detail.data)
    return (
      <div className="p-8 text-sm">
        {zh ? "步骤证据读取失败。" : "Unable to read step evidence."}
        <button
          className="ml-2 underline"
          onClick={() => void detail.refetch()}
        >
          {zh ? "重试" : "Retry"}
        </button>
      </div>
    );
  const step = detail.data;
  const visibleCharts = step.charts.filter((chart) => {
    if (stepKey === "validation") return true;
    if (stepKey === "preprocessing" && beforeAfter !== "both")
      return chart === step.charts[beforeAfter === "before" ? 0 : 1];
    return true;
  });
  const resolvedCandidates =
    candidates.data?.data.map((item) =>
      resolveProcessCandidate(item, step.facts.selection),
    ) ?? [];
  const candidate =
    candidateIndex === null ? null : resolvedCandidates[candidateIndex];
  const {
    decision,
    pipeline,
    selection,
    parameters,
    method_knowledge,
    ...inputFacts
  } = step.facts;
  const decisionRecord = selectionDecision(decision, selection);
  return (
    <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="min-w-0 space-y-4">
        {stepKey === "preprocessing" && (
          <MethodKnowledge evidence={method_knowledge} zh={zh} />
        )}
        <section className="bg-primary/5 rounded-xl border p-5">
          <div className="mb-2 flex flex-wrap gap-2 text-xs">
            <span className="rounded border px-2 py-1">
              {statusText(step.execution_status, zh)}
            </span>
            <span className="rounded border px-2 py-1">
              {statusText(step.evidence_status, zh)}
            </span>
          </div>
          <h2 className="text-xl font-semibold">
            {stepCopy[stepKey][zh ? 0 : 1]}
          </h2>
          <p className="text-muted-foreground mt-2 text-sm leading-6">
            {stepCopy[stepKey][zh ? 2 : 3]}
          </p>
          {step.attempt_id === null && (
            <p className="mt-3 text-sm">
              {zh
                ? "这是该运行的共享工作流记录，不表示当前选中尝试完成了此步骤。"
                : "This is a shared run-level workflow record, not completion evidence for the selected attempt."}
            </p>
          )}
          {stepKey === "registration" && (
            <p className="text-muted-foreground mt-3 text-sm">
              {zh
                ? "审批、注册与入库分别按记录展示；缺少入库记录不表示已经入库。"
                : "Approval, registration and library storage follow their own evidence; missing storage evidence does not mean stored."}
            </p>
          )}
        </section>
        {stepKey === "preprocessing" && (
          <div className="flex flex-wrap gap-2">
            {["before", "after", "both"].map((mode) => (
              <button
                key={mode}
                type="button"
                aria-pressed={beforeAfter === mode}
                className={cn(
                  "rounded-lg border px-3 py-2 text-sm",
                  beforeAfter === mode && "bg-primary text-primary-foreground",
                )}
                onClick={() => setBeforeAfter(mode)}
              >
                {zh
                  ? { before: "处理前", after: "处理后", both: "前后对照" }[
                      mode
                    ]
                  : mode}
              </button>
            ))}
            <span className="text-muted-foreground self-center text-xs">
              {zh
                ? "前后图共享波长缩放区间"
                : "Before/after charts share wavelength zoom"}
            </span>
          </div>
        )}
        {stepKey === "validation" && (
          <div className="flex items-center gap-2 text-sm">
            <label htmlFor="evaluation-scope">
              {zh ? "查看评估分组" : "Evaluation scope"}
            </label>
            <select
              id="evaluation-scope"
              className="bg-background rounded border px-3 py-2"
              value={scope}
              onChange={(event) => {
                setScope(event.target.value);
                setSelected([]);
                setZoom([0, 1]);
              }}
            >
              <option value="holdout">
                {zh ? "留出 / 外部验证" : "Holdout / external"}
              </option>
              <option value="tuning">{plotLabel("tuning", zh)}</option>
            </select>
          </div>
        )}
        {step.metrics && (
          <section className="rounded-xl border p-4">
            <h3 className="mb-3 font-medium">
              {zh
                ? "全量指标（不随框选改变）"
                : "Full-population metrics (unaffected by selection)"}
            </h3>
            <div className="grid gap-3 sm:grid-cols-3">
              {Object.entries(step.metrics).map(([name, metrics]) => (
                <div key={name} className="bg-muted/30 rounded-lg p-3">
                  <strong className="text-sm">
                    {plotLabel(
                      name === "test"
                        ? "holdout"
                        : name === "val"
                          ? "tuning"
                          : "calibration",
                      zh,
                    )}
                  </strong>
                  {Object.entries(metrics).map(([key, value]) => (
                    <div
                      key={key}
                      className="mt-1 flex justify-between text-xs"
                    >
                      <span>{factLabel(key, zh) ?? key}</span>
                      <span>{fmt(value)}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>
        )}
        <div className="space-y-4">
          {visibleCharts.map((chart) => (
            <ChartResource
              key={chart.chart_id}
              threadId={threadId}
              runId={runId}
              chart={chart}
              zh={zh}
              zoom={zoom}
              onZoom={setZoom}
              selected={selected}
              onSelect={setSelected}
              scope={stepKey === "validation" ? scope : null}
            />
          ))}
        </div>
        {step.charts.length === 0 && (
          <p className="text-muted-foreground rounded-xl border border-dashed p-6 text-sm">
            {zh
              ? "本步没有可交互图表；以下仅展示已记录的事实。"
              : "No interactive chart was recorded for this step. Recorded facts remain available."}
          </p>
        )}
        {step.candidate_count > 0 && (
          <section className="rounded-xl border p-4">
            <h3 className="mb-3 font-medium">
              {zh
                ? "候选比较 · 点击查看实际参数与依据"
                : "Candidates · select to inspect parameters and evidence"}
            </h3>
            <p className="text-muted-foreground mb-3 text-xs">
              {zh
                ? "分数来自所标训练内范围；最终留出成绩不参与候选排序。"
                : "Scores use the labeled training-only scope; final holdout scores do not rank candidates."}
            </p>
            {candidates.isError ? (
              <button
                className="underline"
                onClick={() => void candidates.refetch()}
              >
                {zh ? "读取失败，重试" : "Load failed, retry"}
              </button>
            ) : (
              <div className="space-y-2">
                {resolvedCandidates.map((item, index) => (
                  <button
                    key={`${item.candidate_id}:${offset + index}`}
                    type="button"
                    aria-pressed={candidateIndex === index}
                    onClick={() => setCandidateIndex(index)}
                    className={cn(
                      "flex w-full items-center justify-between gap-3 rounded-lg border p-3 text-left text-sm",
                      candidateIndex === index && "border-primary bg-primary/5",
                    )}
                  >
                    <span className="min-w-0">
                      {processCandidateLabel(item, offset + index, zh)}
                      {item.selected && (
                        <span className="ml-2 text-emerald-600">
                          {zh ? "建模采用" : "Adopted"}
                        </span>
                      )}
                      {candidateIndex === index && (
                        <span className="text-primary ml-2">
                          {zh ? "查看中" : "Viewing"}
                        </span>
                      )}
                      {candidateConfiguration(item, zh) && (
                        <span className="text-muted-foreground mt-1 block text-xs">
                          {candidateConfiguration(item, zh)}
                        </span>
                      )}
                    </span>
                    <span className="shrink-0 text-right tabular-nums">
                      {item.error
                        ? zh
                          ? "执行失败"
                          : "Failed"
                        : `${fmt(item.score)} · ${item.evaluation_kind === "tuning" ? (zh ? "调参集误差（RMSE）" : "Tuning RMSE") : zh ? "交叉验证误差（RMSE）" : "CV RMSE"}`}
                    </span>
                  </button>
                ))}
              </div>
            )}
            {candidate && (
              <div
                className="bg-muted/30 mt-3 rounded-lg p-3"
                data-testid="candidate-detail"
              >
                <h4 className="mb-3 font-medium">
                  {zh ? "当前查看方案：" : "Viewing: "}
                  {processCandidateLabel(
                    candidate,
                    offset + (candidateIndex ?? 0),
                    zh,
                  )}
                </h4>
                <Facts value={candidate} zh={zh} />
              </div>
            )}
            <div className="mt-3 flex items-center justify-between text-xs">
              <span>
                {step.candidate_count} {zh ? "个候选" : "candidates"}
              </span>
              <div className="flex gap-3">
                <button
                  disabled={offset === 0}
                  className="disabled:opacity-40"
                  onClick={() => {
                    setOffset(Math.max(0, offset - 50));
                    setCandidateIndex(null);
                  }}
                >
                  {zh ? "上一页" : "Previous"}
                </button>
                <button
                  disabled={candidates.data?.next_offset == null}
                  className="disabled:opacity-40"
                  onClick={() => {
                    setOffset(candidates.data?.next_offset ?? 0);
                    setCandidateIndex(null);
                  }}
                >
                  {zh ? "下一页" : "Next"}
                </button>
              </div>
            </div>
          </section>
        )}
        {previous.data && (
          <section className="rounded-xl border p-4">
            <h3 className="mb-3 font-medium">
              {zh
                ? "与上一轮实际执行对照"
                : "Compare recorded execution with previous attempt"}
            </h3>
            <p className="mb-3 text-sm">
              {pipelineDifference(previous.data.facts, step.facts)
                ? zh
                  ? "记录的配置或决策发生变化；展开查看参数值和顺序。"
                  : "Configuration or decisions changed; inspect values and operation order below."
                : zh
                  ? "记录的配置与决策相同。"
                  : "Recorded configuration and decisions are unchanged."}
            </p>
            <details>
              <summary className="cursor-pointer text-sm">
                {zh ? "展开前后事实" : "Expand before/after facts"}
              </summary>
              <div className="mt-3 grid gap-3 sm:grid-cols-2">
                <Facts value={previous.data.facts} zh={zh} />
                <Facts value={step.facts} zh={zh} />
              </div>
            </details>
          </section>
        )}
      </div>
      <aside className="space-y-4 xl:sticky xl:top-4 xl:self-start">
        <section className="rounded-xl border p-4">
          <h3 className="mb-3 font-medium">
            {zh ? "输入与实际操作" : "Inputs and recorded operations"}
          </h3>
          <Facts
            value={{ ...inputFacts, ...(pipeline ? { pipeline } : {}) }}
            zh={zh}
          />
          {parameters !== undefined && parameters !== null && (
            <details className="mt-3 text-sm">
              <summary className="cursor-pointer">
                {zh ? "展开实际模型参数" : "Recorded model parameters"}
              </summary>
              <div className="mt-3">
                <Facts value={parameters} zh={zh} />
              </div>
            </details>
          )}
        </section>
        <section className="rounded-xl border p-4">
          <h3 className="mb-3 font-medium">
            {zh ? "为什么这样决定" : "Decision evidence"}
          </h3>
          <Facts value={decisionRecord} zh={zh} />
          {selection != null && decisionRecord !== selection && (
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer">
                {zh ? "展开完整比较记录" : "Full comparison record"}
              </summary>
              <div className="mt-3">
                <Facts value={selection} zh={zh} />
              </div>
            </details>
          )}
          {step.comparison_reason && (
            <p className="text-muted-foreground mt-3 text-sm">
              {zh
                ? "历史记录缺少评估口径，不能判定跨尝试更优。"
                : "Missing evaluation context prevents ranking attempts."}
            </p>
          )}
        </section>
        <section className="rounded-xl border p-4">
          <h3 className="mb-2 text-sm font-medium">
            {zh ? "证据来源与下一步" : "Provenance and next step"}
          </h3>
          <details className="text-muted-foreground text-xs">
            <summary className="cursor-pointer">
              {zh ? "步骤追溯编号" : "Step trace ID"}
            </summary>
            <p className="mt-2 break-all">{step.step_execution_id}</p>
          </details>
          <p className="mt-2 text-sm">
            {
              stepCopy[
                PROCESS_STEPS[
                  Math.min(
                    PROCESS_STEPS.indexOf(stepKey) + 1,
                    PROCESS_STEPS.length - 1,
                  )
                ]!
              ][zh ? 0 : 1]
            }
          </p>
        </section>
      </aside>
    </div>
  );
}

export function ProcessWorkspace({ threadId }: { threadId: string }) {
  const { locale } = useI18n();
  const zh = locale === "zh-CN";
  const { user } = useAuth();
  const client = useQueryClient();
  const router = useRouter(),
    pathname = usePathname(),
    search = useSearchParams();
  const [attemptOffset, setAttemptOffset] = useState(0);
  const runs = useProcessResource<RunList>(threadId, "runs", true);
  const runId = search.get("run") ?? runs.data?.data[0]?.run_id;
  const requestedAttempt = search.get("attempt");
  const summaryParams = new URLSearchParams();
  if (requestedAttempt) summaryParams.set("attempt", requestedAttempt);
  else if (attemptOffset) summaryParams.set("offset", String(attemptOffset));
  const summary = useProcessResource<ProcessSummary>(
    threadId,
    runId
      ? `runs/${runId}${summaryParams.size ? `?${summaryParams.toString()}` : ""}`
      : null,
    true,
  );
  const legacy = useQuery({
    queryKey: ["nir-process-legacy", user?.id, threadId],
    queryFn: ({ signal }) => loadNIRWorkflowState(threadId, signal),
    enabled: Boolean(user && runs.data?.data.length === 0),
    retry: false,
  });
  const legacyView = selectNIRProcessView(legacy.data);
  useEffect(
    () => () => {
      client.removeQueries({ queryKey: ["nir-process", user?.id] });
      client.removeQueries({ queryKey: ["nir-process-legacy", user?.id] });
    },
    [client, user?.id],
  );
  const attempt =
    summary.data?.attempts.find(
      (item) => item.attempt_id === search.get("attempt"),
    ) ?? summary.data?.attempts.at(-1);
  const requestedStep = search.get("step");
  const stepKey = PROCESS_STEPS.includes(requestedStep as StepKey)
    ? (requestedStep as StepKey)
    : attempt?.steps.some((item) => item.step_key === "validation")
      ? "validation"
      : attempt?.steps.at(-1)?.step_key === "model"
        ? "model"
        : "audit";
  const shared = summary.data?.shared_steps.filter(
    (item) => item.step_key === stepKey,
  );
  const step =
    attempt?.steps.find((item) => item.step_key === stepKey) ?? shared?.[0];
  const position =
    summary.data?.attempts.findIndex(
      (item) => item.attempt_id === attempt?.attempt_id,
    ) ?? -1;
  const previousStep =
    position > 0
      ? summary.data?.attempts[position - 1]?.steps.find(
          (item) => item.step_key === stepKey,
        )
      : null;
  const scores =
    attempt && summary.data
      ? comparableScores(summary.data.attempts, attempt)
      : [];
  useEffect(() => {
    if (runId && attempt && !requestedAttempt) {
      const params = new URLSearchParams(search.toString());
      params.set("run", runId);
      params.set("attempt", attempt.attempt_id);
      params.set("step", stepKey);
      window.history.replaceState(null, "", `${pathname}?${params.toString()}`);
    }
  }, [runId, attempt, requestedAttempt, search, stepKey, router, pathname]);
  function navigate(values: Record<string, string>) {
    const params = new URLSearchParams(search.toString());
    Object.entries(values).forEach(([key, value]) => params.set(key, value));
    // Same-page evidence navigation should not fetch a new server component.
    window.history.pushState(null, "", `${pathname}?${params.toString()}`);
  }
  return (
    <div className="h-full min-h-0 w-full overflow-y-auto">
      <header className="bg-background sticky top-0 z-10 border-b px-4 py-4 md:px-7">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <Link
              href={`/workspace/chats/${encodeURIComponent(threadId)}`}
              className="text-muted-foreground mb-2 inline-flex items-center gap-1 text-xs"
            >
              <ArrowLeftIcon className="size-3" />
              {zh ? "返回聊天" : "Back to chat"}
            </Link>
            <h1 className="text-xl font-semibold">
              {zh ? "建模过程探索" : "Modeling process explorer"}
            </h1>
          </div>
          <button
            type="button"
            className="flex items-center gap-2 rounded-lg border px-3 py-2 text-sm"
            onClick={() => {
              void runs.refetch();
              void summary.refetch();
              void legacy.refetch();
            }}
          >
            <RefreshCwIcon className="size-4" />
            {zh ? "刷新进展" : "Refresh"}
          </button>
        </div>
        {summary.data && !summary.isError && !runs.isError && (
          <div className="text-muted-foreground mt-3 flex flex-wrap items-center gap-3 text-xs">
            <label className="flex items-center gap-2">
              {zh ? "运行" : "Run"}
              <select
                className="bg-background rounded border p-1"
                aria-label={zh ? "选择运行" : "Select run"}
                value={runId}
                onChange={(event) => {
                  setAttemptOffset(0);
                  navigate({ run: event.target.value, attempt: "", step: "" });
                }}
              >
                {runs.data?.data.map((run, index) => (
                  <option key={run.run_id} value={run.run_id}>
                    {index + 1} · {run.run_id.slice(-8)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-2">
              {zh ? "建模尝试" : "Attempt"}
              <select
                className="bg-background rounded border p-1"
                aria-label={zh ? "选择建模尝试" : "Select attempt"}
                value={attempt?.attempt_id ?? ""}
                onChange={(event) =>
                  navigate({
                    run: runId!,
                    attempt: event.target.value,
                    step: stepKey,
                  })
                }
              >
                {summary.data.attempts.map((item) => (
                  <option key={item.attempt_id} value={item.attempt_id}>
                    {zh ? `第 ${item.number} 次` : `Attempt ${item.number}`} ·{" "}
                    {statusText(item.execution_status, zh)}
                  </option>
                ))}
              </select>
            </label>
            <span>
              {String(factValue("stage", summary.data.stage, zh))} ·{" "}
              {zh ? "证据更新" : "Updated"}:{" "}
              {new Date(summary.data.updated_at).toLocaleTimeString(locale)}
            </span>
            <details>
              <summary className="cursor-pointer">
                {String(
                  factValue("source_tool", attempt?.source_tool, zh) ?? "—",
                )}
              </summary>
              <span>{attempt?.source_tool}</span>
            </details>
            {summary.data.next_offset !== null && (
              <button
                onClick={() => {
                  setAttemptOffset(summary.data.next_offset!);
                  navigate({ attempt: "" });
                }}
                className="underline"
              >
                {zh ? "更早尝试" : "Earlier attempts"}
              </button>
            )}
            {attemptOffset > 0 && (
              <button
                onClick={() => {
                  setAttemptOffset(0);
                  navigate({ attempt: "" });
                }}
                className="underline"
              >
                {zh ? "最新尝试" : "Latest attempts"}
              </button>
            )}
          </div>
        )}
      </header>
      {runs.isPending && (
        <p className="p-8">
          {zh ? "正在读取建模记录…" : "Loading process records…"}
        </p>
      )}
      {(runs.isError || (runId && summary.isError)) && (
        <div className="p-8">
          <p>
            {summary.error instanceof ProcessAPIError &&
            summary.error.status === 404
              ? zh
                ? "该运行不存在或没有访问权限。"
                : "Run not found or inaccessible."
              : zh
                ? "过程记录读取失败，请重试。"
                : "Unable to read process records. Retry the request."}
          </p>
        </div>
      )}
      {runs.data?.data.length === 0 && (
        <div className="p-5">
          <p className="mb-4 rounded-lg border p-4 text-sm">
            {zh
              ? "此会话没有新版过程证据。历史记录只展示已保存的摘要，后续建模将生成交互图表。"
              : "No new process evidence exists for this chat. Saved legacy summaries remain available; future modeling creates interactive charts."}
          </p>
          {legacyView ? (
            <NIRProcessContent
              view={legacyView}
              onArtifacts={() => router.push(`/workspace/chats/${threadId}`)}
              hasArtifacts={false}
            />
          ) : (
            <p className="text-muted-foreground text-sm">
              {zh ? "暂无建模摘要。" : "No modeling summary yet."}
            </p>
          )}
        </div>
      )}
      {summary.data && !summary.isError && !runs.isError && (
        <div className="grid gap-5 p-4 md:p-7 lg:grid-cols-[200px_minmax(0,1fr)]">
          <nav
            className="flex gap-2 overflow-x-auto lg:sticky lg:top-4 lg:block lg:space-y-2 lg:self-start"
            aria-label={zh ? "建模步骤" : "Modeling steps"}
          >
            {PROCESS_STEPS.map((key, index) => {
              const node =
                attempt?.steps.find((item) => item.step_key === key) ??
                summary.data?.shared_steps.find(
                  (item) => item.step_key === key,
                );
              return (
                <button
                  key={key}
                  type="button"
                  aria-current={key === stepKey ? "step" : undefined}
                  className={cn(
                    "min-w-36 rounded-xl border p-3 text-left lg:w-full",
                    key === stepKey && "border-primary bg-primary/5",
                  )}
                  onClick={() =>
                    navigate({
                      run: runId!,
                      attempt: attempt?.attempt_id ?? "",
                      step: key,
                    })
                  }
                >
                  <span className="text-muted-foreground mb-1 block text-xs">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="block text-sm font-medium">
                    {stepCopy[key][zh ? 0 : 1]}
                  </span>
                  <span className="text-muted-foreground mt-1 block text-xs">
                    {statusText(
                      node?.execution_status ??
                        (attempt?.execution_status === "running"
                          ? "pending"
                          : "unknown"),
                      zh,
                    )}
                  </span>
                </button>
              );
            })}
            <div className="mt-4 hidden rounded-lg border p-3 text-xs lg:block">
              <strong>
                {zh
                  ? "可比尝试 · 调参集 RMSE"
                  : "Comparable attempts · tuning RMSE"}
              </strong>
              {scores.length > 1 && (
                <svg
                  viewBox="0 0 240 105"
                  role="img"
                  aria-label={
                    zh ? "同口径调参误差趋势" : "Comparable tuning error trend"
                  }
                  className="mt-3 w-full"
                >
                  <polyline
                    fill="none"
                    stroke="#2563eb"
                    strokeWidth="2"
                    points={scores
                      .map(
                        ({ value }, index) =>
                          `${20 + (index / (scores.length - 1)) * 200},${75 - ((value - Math.min(...scores.map((item) => item.value))) / (Math.max(...scores.map((item) => item.value)) - Math.min(...scores.map((item) => item.value)) || 1)) * 55}`,
                      )
                      .join(" ")}
                  />
                  {scores.map(({ attempt: item, value }, index) => (
                    <g key={item.attempt_id}>
                      <circle
                        cx={20 + (index / (scores.length - 1)) * 200}
                        cy={
                          75 -
                          ((value -
                            Math.min(...scores.map((point) => point.value))) /
                            (Math.max(...scores.map((point) => point.value)) -
                              Math.min(...scores.map((point) => point.value)) ||
                              1)) *
                            55
                        }
                        r={item.attempt_id === attempt?.attempt_id ? 5 : 3}
                        fill={
                          value ===
                          Math.min(...scores.map((point) => point.value))
                            ? "#059669"
                            : "#2563eb"
                        }
                      >
                        <title>
                          #{item.number}: {fmt(value)}
                        </title>
                      </circle>
                      <text
                        x={20 + (index / (scores.length - 1)) * 200}
                        y="98"
                        fontSize="10"
                        textAnchor="middle"
                        fill="currentColor"
                      >
                        #{item.number}
                      </text>
                    </g>
                  ))}
                </svg>
              )}
              {scores.length ? (
                scores.map(({ attempt: item, value }) => (
                  <button
                    key={item.attempt_id}
                    className="mt-2 flex w-full justify-between"
                    onClick={() =>
                      navigate({
                        run: runId!,
                        attempt: item.attempt_id,
                        step: "model",
                      })
                    }
                  >
                    <span>
                      #{item.number}
                      {value ===
                        Math.min(...scores.map((point) => point.value)) &&
                      scores.length > 1
                        ? zh
                          ? " · 本页最低"
                          : " · page minimum"
                        : ""}
                    </span>
                    <span>{fmt(value)}</span>
                  </button>
                ))
              ) : (
                <p className="text-muted-foreground mt-2">
                  {zh
                    ? "缺少完整口径，无法跨尝试排名。"
                    : "Incomplete context prevents ranking attempts."}
                </p>
              )}
              {scores.length > 0 && (
                <p className="text-muted-foreground mt-2">
                  {zh
                    ? "仅比较当前页同口径记录；留出集不参与排名。"
                    : "Only matching context on this page is compared; holdout results do not rank attempts."}
                </p>
              )}
            </div>
          </nav>
          <main className="min-w-0">
            {step ? (
              <StepDetail
                key={step.step_execution_id}
                threadId={threadId}
                runId={runId!}
                stepId={step.step_execution_id}
                stepKey={stepKey}
                zh={zh}
                previousStepId={previousStep?.step_execution_id}
              />
            ) : (
              <section className="rounded-xl border border-dashed p-8">
                <h2 className="text-xl font-semibold">
                  {stepCopy[stepKey][zh ? 0 : 1]}
                </h2>
                <p className="text-muted-foreground mt-3 text-sm">
                  {attempt?.execution_status === "running"
                    ? zh
                      ? "训练工具正在执行，完成后将展示记录的内部步骤。"
                      : "Training is running. Recorded internal steps appear when the tool completes."
                    : zh
                      ? "本次运行没有对应执行证据，不能推断已完成或跳过。"
                      : "No execution evidence exists for this step; completion or skipping cannot be inferred."}
                </p>
              </section>
            )}
          </main>
        </div>
      )}
    </div>
  );
}
