"use client";

import { useState } from "react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/core/i18n/hooks";
import {
  candidateLabel,
  methodLabel,
  pipelineLabel,
  protocolLabel,
  selectedLabel,
  selectionExplanation,
} from "@/core/nir/presentation";
import {
  factLabel,
  factValue,
  formatProcessNumber,
} from "@/core/nir/process-display";
import type {
  AttemptView,
  CandidateView,
  ProcessView,
  SelectionView,
} from "@/core/nir/process-selectors";
import { selectNIRMilestones } from "@/core/nir/progress";
import { cn } from "@/lib/utils";

const fmt = formatProcessNumber;

function Card({
  title,
  children,
  className,
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("bg-card rounded-xl border p-4", className)}>
      <h3 className="mb-3 text-sm font-semibold">{title}</h3>
      {children}
    </section>
  );
}

function Fact({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-4 border-b py-2 text-sm last:border-0">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="min-w-0 text-right font-medium break-words">
        {value ?? "—"}
      </span>
    </div>
  );
}

function MetricTrend({ view, zh }: { view: ProcessView; zh: boolean }) {
  const points = view.trend;
  if (!points.length)
    return (
      <p className="text-muted-foreground text-sm">
        {zh ? "暂无可比较的数值指标" : "No comparable numeric metrics yet"}
      </p>
    );
  const values = points.flatMap((point) => [point.value, point.best]);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const y = (v: number) =>
    84 - (high === low ? 0.5 : (v - low) / (high - low)) * 65;
  const x = (i: number) =>
    points.length === 1 ? 150 : 26 + (i / (points.length - 1)) * 248;
  return (
    <div>
      <svg
        role="img"
        aria-label={
          zh
            ? "尝试指标和截至当前最佳值趋势"
            : "Attempt metric and best so far trend"
        }
        viewBox="0 0 300 110"
        className="h-36 w-full"
        preserveAspectRatio="none"
      >
        <line
          x1="25"
          x2="275"
          y1="84"
          y2="84"
          stroke="currentColor"
          opacity=".25"
        />
        {points.length > 1 && (
          <>
            <polyline
              points={points.map((p, i) => `${x(i)},${y(p.value)}`).join(" ")}
              fill="none"
              stroke="#2563eb"
              strokeWidth="2"
            />
            <polyline
              points={points.map((p, i) => `${x(i)},${y(p.best)}`).join(" ")}
              fill="none"
              stroke="#059669"
              strokeWidth="2"
              strokeDasharray="4 3"
            />
          </>
        )}
        {points.map((p, i) => (
          <g key={p.attempt}>
            <circle cx={x(i)} cy={y(p.value)} r="4" fill="#2563eb" />
            <text
              x={x(i)}
              y="104"
              textAnchor="middle"
              fontSize="10"
              fill="currentColor"
            >
              #{p.attempt}
            </text>
          </g>
        ))}
      </svg>
      <div className="flex gap-4 text-xs">
        <span className="text-blue-600">● {zh ? "当轮" : "Attempt"}</span>
        <span className="text-emerald-600">
          ┄ {zh ? "同组截至当前最佳" : "Best in comparable group"}
        </span>
      </div>
      <p className="text-muted-foreground mt-2 text-xs">
        {factLabel(
          view.attempts.find((a) => a.number === points[0]?.attempt)
            ?.metricKey ?? "",
          zh,
        ) ?? (zh ? "已记录指标" : "Recorded metric")}{" "}
        ·{" "}
        {zh
          ? "仅比较相同指标、协议和验证范围；不代表外部验证"
          : "Same metric, protocol and validation scope only; not necessarily external validation"}
      </p>
    </div>
  );
}

function CandidateChart({
  candidates,
  kind,
  zh,
}: {
  candidates: CandidateView[];
  kind: "preprocessing" | "wavelength" | "model";
  zh: boolean;
}) {
  const scored = candidates.filter((candidate) => candidate.score !== null);
  const key = scored[0]?.scoreKey;
  const comparable = scored.filter((candidate) => candidate.scoreKey === key);
  if (!key || !comparable.length)
    return (
      <p className="text-muted-foreground text-xs">
        {zh ? "没有可比较的候选指标" : "No comparable candidate scores"}
      </p>
    );
  const max = Math.max(
    ...comparable.map((candidate) => candidate.score!),
    0.000001,
  );
  return (
    <div
      role="img"
      aria-label={`${key} ${zh ? "训练内候选误差比较" : "candidate comparison"}`}
      className="space-y-2"
    >
      {comparable.map((candidate) => (
        <div
          key={candidate.name}
          className="grid grid-cols-[minmax(8rem,1.2fr)_minmax(5rem,1fr)_4rem] items-center gap-2 text-xs"
        >
          <span className="min-w-0 break-words" title={candidate.name}>
            {candidateLabel(candidate, kind, candidates.indexOf(candidate), zh)}
            {candidate.selected && (
              <span className="ml-1 font-semibold text-emerald-700">
                {zh ? "· 已选" : "· Selected"}
              </span>
            )}
          </span>
          <div className="bg-muted h-3 rounded">
            <div
              className={cn(
                "h-3 rounded",
                candidate.selected ? "bg-emerald-500" : "bg-blue-500",
              )}
              style={{
                width: `${Math.max(2, (candidate.score! / max) * 100)}%`,
              }}
            />
          </div>
          <span className="text-right tabular-nums">
            {fmt(candidate.score)}
          </span>
        </div>
      ))}
      <p className="text-muted-foreground text-xs">
        {zh
          ? "训练数据内部的误差，越低越好；绿色是本次选择。这里不是最终测试成绩。"
          : "Error within training data; lower is better. Green marks the choice. These are not final test scores."}
      </p>
    </div>
  );
}

function SelectionCard({
  title,
  selection,
  kind,
  zh,
}: {
  title: string;
  selection: SelectionView;
  kind: "preprocessing" | "wavelength" | "model";
  zh: boolean;
}) {
  const blank = zh ? "证据未记录" : "Evidence not recorded";
  return (
    <Card title={title}>
      <p className="text-muted-foreground text-xs">
        {zh ? "本次采用" : "Chosen option"}
      </p>
      <p className="mt-1 text-lg font-semibold">
        {selection.selected ? selectedLabel(selection, kind, zh) : blank}
      </p>
      <p className="mt-2 text-sm leading-6">
        {selection.selected ? selectionExplanation(kind, selection, zh) : blank}
      </p>
      {selection.improvement !== null && (
        <Fact
          label={
            zh ? "调参误差改善 / 最低要求" : "Tuning improvement / minimum"
          }
          value={`${(selection.improvement * 100).toFixed(2)}% / ${selection.minImprovement === null ? "—" : `${(selection.minImprovement * 100).toFixed(2)}%`}`}
        />
      )}
      {selection.candidates.length ? (
        <div className="mt-5 border-t pt-4">
          <h4 className="mb-3 text-sm font-medium">
            {zh
              ? `比较过的方案（${selection.candidates.length}）`
              : `Compared options (${selection.candidates.length})`}
          </h4>
          <CandidateChart
            candidates={selection.candidates}
            kind={kind}
            zh={zh}
          />
        </div>
      ) : (
        <p className="text-muted-foreground mt-3 text-xs">
          {zh ? "候选详情未记录" : "Candidate details not recorded"}
        </p>
      )}
      <details className="mt-4 border-t pt-3 text-xs">
        <summary className="text-muted-foreground cursor-pointer font-medium">
          {zh ? "查看原始技术记录" : "Show original technical record"}
        </summary>
        <div className="text-muted-foreground mt-3 space-y-1 break-all">
          <p>
            {zh ? "原始选择" : "Raw selection"}: {selection.selected ?? blank}
          </p>
          {selection.reasonCode && (
            <p>
              {zh ? "原因代码" : "Reason code"}: {selection.reasonCode}
            </p>
          )}
          {selection.comparisonReasonCode && (
            <p>
              {zh ? "比较触发" : "Comparison trigger"}:{" "}
              {selection.comparisonReasonCode}
            </p>
          )}
          {selection.rule && (
            <p>
              {zh ? "规则" : "Rule"}: {selection.rule}
            </p>
          )}
          {selection.reason && (
            <p>
              {zh ? "原始说明" : "Original explanation"}: {selection.reason}
            </p>
          )}
          {selection.candidates.map((candidate) => (
            <p key={candidate.name}>
              {candidate.name}: {candidate.scoreKey ?? "score"}{" "}
              {fmt(candidate.score)}
              {candidate.reason ? ` · ${candidate.reason}` : ""}
            </p>
          ))}
        </div>
      </details>
    </Card>
  );
}

function AttemptStory({
  attempt,
  zh,
  best,
}: {
  attempt: AttemptView;
  zh: boolean;
  best: boolean;
}) {
  return (
    <li className="border-primary/30 relative border-l-2 pb-5 pl-5 last:pb-0">
      <span className="bg-primary absolute top-1 -left-[5px] size-2 rounded-full" />
      <div className="flex flex-wrap items-center gap-2">
        <h4 className="font-semibold">
          {zh ? `第 ${attempt.number} 次建模` : `Attempt ${attempt.number}`}
        </h4>
        {best && (
          <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-700">
            {zh ? "同组最佳" : "Best comparable"}
          </span>
        )}
        <span
          className={cn(
            "rounded px-2 py-0.5 text-xs",
            attempt.passed
              ? "bg-emerald-500/10 text-emerald-700"
              : "bg-amber-500/10 text-amber-700",
          )}
        >
          {attempt.passed === null
            ? "—"
            : attempt.passed
              ? zh
                ? "质量达标"
                : "Passed"
              : zh
                ? "未达标"
                : "Did not pass"}
        </span>
      </div>
      <p className="mt-2 text-sm">
        {pipelineLabel(attempt.pipeline, zh)} →{" "}
        {methodLabel(attempt.method, zh)}
      </p>
      {attempt.pipelineChanges.length > 0 && (
        <p className="text-muted-foreground mt-1 text-xs">
          {zh ? "与上轮差异" : "Changed from prior attempt"}:{" "}
          {attempt.pipelineChanges.join(", ")}
        </p>
      )}
      <p className="text-muted-foreground mt-1 text-xs">
        {attempt.metricKey
          ? `${factLabel(attempt.metricKey, zh) ?? attempt.metricKey}: ${fmt(attempt.metricValue)}`
          : zh
            ? "指标未记录"
            : "Metric not recorded"}{" "}
        ·{" "}
        {String(
          factValue("validation_scope", attempt.validationScope, zh) ??
            (zh ? "验证范围未记录" : "Validation scope unknown"),
        )}
      </p>
      {attempt.reflection && (
        <div className="bg-muted/40 mt-3 rounded-lg p-3 text-sm">
          <strong>{zh ? "反思" : "Reflection"}</strong>
          <p className="mt-1">
            {attempt.reflection.reason ??
              (zh ? "原因未记录" : "Reason not recorded")}
          </p>
          {attempt.reflection.diagnostics.length > 0 && (
            <p className="text-muted-foreground mt-1 text-xs">
              {zh ? "诊断字段" : "Diagnostics"}:{" "}
              {attempt.reflection.diagnostics
                .map((item) => `${item.key}: ${item.value}`)
                .join(", ")}
            </p>
          )}
          <p className="text-muted-foreground mt-1 text-xs">
            {attempt.reflection.shouldRetry
              ? zh
                ? "决定继续尝试"
                : "Retry planned"
              : zh
                ? "决定停止重试"
                : "Retry stopped"}
            {attempt.reflection.stopReason
              ? ` · ${attempt.reflection.stopReason}`
              : ""}
          </p>
        </div>
      )}
      {attempt.retry && (
        <div className="mt-2 rounded-lg border border-dashed p-3 text-sm">
          <strong>{zh ? "下一轮计划" : "Next retry plan"}</strong>
          <p className="mt-1">
            {pipelineLabel(attempt.retry.pipeline, zh)} →{" "}
            {methodLabel(attempt.retry.method, zh)}
          </p>
          <p className="text-muted-foreground mt-1">
            {attempt.retry.rationale ??
              (zh ? "理由未记录" : "Rationale not recorded")}
          </p>
          {attempt.retry.expectedImprovement && (
            <p className="text-muted-foreground mt-1 text-xs">
              {zh ? "预期改善" : "Expected improvement"}:{" "}
              {attempt.retry.expectedImprovement}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

export function NIRProcessContent({
  view,
  onArtifacts,
  hasArtifacts,
}: {
  view: ProcessView;
  onArtifacts: () => void;
  hasArtifacts: boolean;
}) {
  const { locale, t } = useI18n();
  const zh = locale === "zh-CN";
  const [selectedAttempt, setSelectedAttempt] = useState<number | null>(null);
  const active =
    view.attempts.find((attempt) => attempt.number === selectedAttempt) ??
    view.attempts.at(-1);
  const v = view.base;
  const milestones = selectNIRMilestones(v);
  const stage = t.nirWorkflow.stages[v.stage] ?? v.stage;
  const validation = active?.validationScope
    ? (t.nirWorkflow.validationScopes[active.validationScope] ??
      active.validationScope)
    : (t.nirWorkflow.validationGoals[v.validationGoal ?? ""] ?? "—");
  return (
    <Tabs defaultValue="process" className="min-h-0 flex-1">
      <TabsList
        className="mx-5 mt-4 grid w-[calc(100%-2.5rem)] grid-cols-3"
        aria-label={zh ? "建模过程视图" : "Modeling process views"}
      >
        <TabsTrigger value="process">
          {zh ? "建模过程" : "Modeling process"}
        </TabsTrigger>
        <TabsTrigger value="selection">
          {zh ? "选择依据" : "Selection evidence"}
        </TabsTrigger>
        <TabsTrigger value="final">
          {zh ? "最终结果" : "Final result"}
        </TabsTrigger>
      </TabsList>
      <TabsContent
        value="process"
        className="min-h-0 flex-1 overflow-y-auto px-5 pb-6"
      >
        {milestones && (
          <ol
            aria-label={t.nirWorkflow.progress}
            className="mb-4 grid grid-cols-4 gap-2"
          >
            {milestones.map((step) => (
              <li
                key={step.key}
                aria-current={step.status === "current" ? "step" : undefined}
                className={cn(
                  "rounded-lg border px-2 py-2 text-center text-xs",
                  step.status === "complete" &&
                    "border-emerald-500/30 bg-emerald-500/10",
                  step.status === "current" &&
                    "border-primary bg-primary/10 font-semibold",
                  step.status === "upcoming" && "text-muted-foreground",
                )}
              >
                {t.nirWorkflow.milestones[step.key]}
              </li>
            ))}
          </ol>
        )}
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]">
          <div className="space-y-4">
            <Card title={zh ? "任务概览" : "Task overview"}>
              <Fact
                label={zh ? "任务" : "Task"}
                value={t.nirWorkflow.taskTypes[v.taskType] ?? v.taskType}
              />
              <Fact
                label={zh ? "数据集" : "Dataset"}
                value={v.datasetId ?? "—"}
              />
              <Fact label={zh ? "当前阶段" : "Stage"} value={stage} />
              <Fact
                label={zh ? "尝试进度" : "Attempts"}
                value={t.nirWorkflow.attempt(v.attempt, v.maxAttempts)}
              />
              <Fact
                label={zh ? "下一步" : "Next action"}
                value={
                  t.nirWorkflow.nextActions[v.nextAction ?? ""] ??
                  v.nextAction ??
                  "—"
                }
              />
            </Card>
            <Card title={zh ? "数据检查" : "Data audit"}>
              <Fact
                label={zh ? "样本数" : "Samples"}
                value={view.audit.samples}
              />
              <Fact
                label={zh ? "光谱变量" : "Spectral variables"}
                value={view.audit.wavelengths}
              />
              <Fact
                label={zh ? "波长范围" : "Wavelength range"}
                value={view.audit.range}
              />
              <Fact
                label={zh ? "可用波长" : "Usable wavelengths"}
                value={view.audit.usableWavelengths}
              />
              <Fact
                label={zh ? "恒定波长" : "Constant wavelengths"}
                value={view.audit.constantWavelengths}
              />
              <Fact
                label={zh ? "缺失值" : "Missing values"}
                value={
                  view.audit.hasNaN === null
                    ? "—"
                    : view.audit.hasNaN
                      ? zh
                        ? "存在"
                        : "Present"
                      : zh
                        ? "未发现"
                        : "None found"
                }
              />
            </Card>
            <Card title={zh ? "尝试指标趋势" : "Attempt metric trend"}>
              <MetricTrend view={view} zh={zh} />
            </Card>
          </div>
          <Card
            title={
              zh ? "尝试、反思与重试" : "Attempts, reflections and retries"
            }
          >
            <ol className="ml-2">
              {view.attempts.length ? (
                view.attempts.map((attempt) => (
                  <AttemptStory
                    key={attempt.number}
                    attempt={attempt}
                    zh={zh}
                    best={attempt.number === view.bestAttempt}
                  />
                ))
              ) : (
                <p className="text-muted-foreground text-sm">
                  {zh ? "尚未开始建模尝试" : "No modeling attempts yet"}
                </p>
              )}
            </ol>
          </Card>
        </div>
      </TabsContent>
      <TabsContent
        value="selection"
        className="min-h-0 flex-1 overflow-y-auto px-5 pb-6"
      >
        <div className="mb-4 flex items-center gap-3 text-sm">
          <label htmlFor="nir-attempt-select">
            {zh ? "查看尝试" : "View attempt"}
          </label>
          <select
            id="nir-attempt-select"
            className="bg-background rounded-md border px-2 py-1"
            value={active?.number ?? ""}
            onChange={(event) => setSelectedAttempt(Number(event.target.value))}
          >
            {view.attempts.map((attempt) => (
              <option key={attempt.number} value={attempt.number}>
                #{attempt.number}
              </option>
            ))}
          </select>
          {active?.protocol && (
            <span className="text-muted-foreground">
              {protocolLabel(active.protocol, zh)}
            </span>
          )}
        </div>
        {active ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="bg-primary/5 rounded-xl border p-4 lg:col-span-2">
              <p className="text-sm font-semibold">
                {zh ? "本轮选定的方案" : "Chosen for this attempt"}
              </p>
              <p className="mt-1 text-sm leading-6">
                {zh ? "预处理：" : "Preprocessing: "}
                {selectedLabel(active.preprocessing, "preprocessing", zh)}
                {" · "}
                {zh ? "波长：" : "Wavelengths: "}
                {selectedLabel(active.wavelength, "wavelength", zh)}
                {" · "}
                {zh ? "模型：" : "Model: "}
                {selectedLabel(active.model, "model", zh)}
              </p>
              <p className="text-muted-foreground mt-2 text-xs">
                {zh
                  ? "下方比较只用于说明如何选方案。模型实际表现请看“最终结果”。"
                  : "The comparisons below explain the choices. See Final result for model performance."}
                {" · "}
                {zh ? "验证范围：" : "Validation scope: "}
                {validation}
              </p>
            </div>
            <div className="lg:col-span-2">
              <SelectionCard
                title={zh ? "预处理选择" : "Preprocessing selection"}
                selection={active.preprocessing}
                kind="preprocessing"
                zh={zh}
              />
            </div>
            <SelectionCard
              title={zh ? "波长选择" : "Wavelength selection"}
              selection={active.wavelength}
              kind="wavelength"
              zh={zh}
            />
            <SelectionCard
              title={zh ? "模型选择" : "Model selection"}
              selection={active.model}
              kind="model"
              zh={zh}
            />
          </div>
        ) : (
          <p className="text-muted-foreground">
            {zh ? "尚无选择记录" : "No selection evidence yet"}
          </p>
        )}
      </TabsContent>
      <TabsContent
        value="final"
        className="min-h-0 flex-1 overflow-y-auto px-5 pb-6"
      >
        <div className="grid gap-4 lg:grid-cols-2">
          <Card title={zh ? "当前结果" : "Current result"}>
            <Fact label={zh ? "状态" : "Stage"} value={stage} />
            <Fact
              label={zh ? "质量门禁" : "Quality gate"}
              value={
                active?.passed === null || active?.passed === undefined
                  ? "—"
                  : active.passed
                    ? t.nirWorkflow.qualityPassed
                    : t.nirWorkflow.qualityFailed
              }
            />
            <Fact
              label={zh ? "审批" : "Approval"}
              value={
                t.nirWorkflow.approvals[v.approvalStatus ?? ""] ??
                v.approvalStatus ??
                "—"
              }
            />
            <Fact
              label={zh ? "注册" : "Registration"}
              value={
                view.registered
                  ? zh
                    ? "工作流已注册"
                    : "Workflow registered"
                  : zh
                    ? "未记录为已注册"
                    : "Not recorded as registered"
              }
            />
            <Fact
              label="Model Library"
              value={zh ? "未提供入库证据" : "No library evidence in workflow"}
            />
            <Fact
              label={zh ? "停止原因" : "Stop reason"}
              value={
                view.stopReason === "retry_budget_exhausted"
                  ? zh
                    ? "重试预算耗尽"
                    : "Retry budget exhausted"
                  : view.stopReason === "reflection_stop"
                    ? zh
                      ? "反思决定停止"
                      : "Reflection stopped retries"
                    : active?.passed
                      ? zh
                        ? "质量达标，待复核或交付"
                        : "Quality passed; review or delivery pending"
                      : (view.stopReason ?? "—")
              }
            />
          </Card>
          <Card
            title={
              v.stage === "blocked"
                ? zh
                  ? "当前最佳可得评价"
                  : "Best available evaluation"
                : zh
                  ? "最终评价"
                  : "Final evaluation"
            }
          >
            {view.finalReady && active ? (
              <>
                <Fact
                  label={zh ? "Pipeline" : "Pipeline"}
                  value={`${pipelineLabel(active.pipeline, zh)} → ${methodLabel(active.method, zh)}`}
                />
                <Fact
                  label={zh ? "验证范围" : "Validation scope"}
                  value={validation}
                />
                <Fact
                  label={zh ? "是否外部验证" : "External validation"}
                  value={
                    active.validationScope === "independent_external_validation"
                      ? zh
                        ? "是"
                        : "Yes"
                      : zh
                        ? "没有外部验证证据"
                        : "No external validation evidence"
                  }
                />
                <Fact
                  label={zh ? "当前选择依据" : "Recorded choice evidence"}
                  value={
                    active.preprocessing.selected ||
                    active.wavelength.selected ||
                    active.model.selected
                      ? zh
                        ? "已记录；请在“选择依据”中查看"
                        : "Recorded; see Selection evidence"
                      : zh
                        ? "未记录"
                        : "Not recorded"
                  }
                />
                {active.metrics.map((item) => (
                  <Fact
                    key={item.key}
                    label={`${factLabel(item.key, zh) ?? item.key}${item.key.startsWith("RMSE") && view.unit ? ` (${view.unit})` : ""}`}
                    value={fmt(item.value)}
                  />
                ))}
              </>
            ) : (
              <p className="text-muted-foreground text-sm">
                {zh
                  ? "尚未进入结果复核阶段或证据不足"
                  : "Result review has not begun or evidence is insufficient"}
              </p>
            )}
          </Card>
          <Card
            title={
              zh ? "其他尝试与选择依据" : "Other attempts and choice evidence"
            }
          >
            <Fact
              label={zh ? "同组最佳尝试" : "Best comparable attempt"}
              value={view.bestAttempt ? `#${view.bestAttempt}` : "—"}
            />
            <p className="text-muted-foreground mt-2 text-xs">
              {zh
                ? "仅比较相同指标、协议和验证范围；当前 Pipeline 的正式选择以工作流状态及选择记录为准。"
                : "Comparison uses the same metric, protocol and validation scope. The recorded workflow decision remains authoritative."}
            </p>
            <ul className="mt-3 space-y-1 text-sm">
              {view.attempts
                .filter((attempt) => attempt.number !== active?.number)
                .map((attempt) => (
                  <li key={attempt.number}>
                    #{attempt.number} · {attempt.method ?? "—"} ·{" "}
                    {attempt.metricKey ?? "—"} {fmt(attempt.metricValue)}
                  </li>
                ))}
            </ul>
          </Card>
          <Card title={zh ? "科学结果与文件" : "Scientific results and files"}>
            {hasArtifacts ? (
              <button
                type="button"
                onClick={onArtifacts}
                className="bg-primary text-primary-foreground rounded-lg px-4 py-2 text-sm"
              >
                {t.nirWorkflow.resultFiles}
              </button>
            ) : (
              <p className="text-muted-foreground text-sm">
                {zh
                  ? "本对话暂无可查看的图表或报告"
                  : "No charts or reports available in this chat"}
              </p>
            )}
          </Card>
        </div>
      </TabsContent>
    </Tabs>
  );
}
