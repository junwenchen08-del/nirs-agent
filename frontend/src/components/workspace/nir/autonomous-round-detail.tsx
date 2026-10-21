"use client";

import { useEffect, useRef } from "react";

import {
  executionLabel,
  isAutonomousTerminal,
  parseDecision,
  record,
  sameRecordedValue,
  text,
  type AutonomousSummary,
  type CalSpectra,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import {
  hasRecordedContent,
  recordedStatements,
} from "@/core/nir/round-content";
import { roundConfiguration } from "@/core/nir/round-process";

import { ParameterTrials, RecordedNarrative } from "./autonomous-evidence";
import { type NodeDetailTab } from "./node-detail-tabs";
import { RoundComparison } from "./round-comparison";
import { RoundDiagnostics } from "./round-diagnostics";
import { RoundNextRecommendation } from "./round-next-recommendation";
import { RecordedStatements, RoundParameters } from "./round-recorded-content";
import { RoundStepSpectra, type SpectraStatus } from "./round-step-spectra";

export function AutonomousRoundDetail({
  threadId,
  summary,
  decision,
  zh,
  onSelect,
  tab,
  onTabChange,
  focusStep = null,
  spectra = null,
  spectraStatus,
  onRetrySpectra,
}: {
  threadId: string;
  summary: AutonomousSummary;
  decision: DecisionView;
  zh: boolean;
  onSelect: (decision: DecisionView, focus?: string) => void;
  tab: NodeDetailTab;
  onTabChange: (tab: NodeDetailTab) => void;
  focusStep?: string | null;
  embedded?: boolean;
  spectra?: CalSpectra | null;
  spectraStatus?: SpectraStatus;
  onRetrySpectra?: () => void;
}) {
  const query = useProcessResource<unknown>(
    threadId,
    `autonomous/${summary.session_id}/decisions/${decision.decision_id}`,
    !isAutonomousTerminal(summary.stage),
  );
  const revision = useRef(summary.revision);
  const { refetch } = query;
  useEffect(() => {
    if (revision.current !== summary.revision) {
      revision.current = summary.revision;
      void refetch();
    }
  }, [summary.revision, refetch]);
  const parsed = parseDecision(query.data);
  const details =
    parsed?.decision_id === decision.decision_id ? parsed : decision;
  const stopping = ["STOP", "RETAIN"].includes(details.action);
  const configuration = roundConfiguration(details);
  const proposal = details.proposal ?? {};
  const feedback = details.feedback ?? {};
  const next =
    parsed?.decision_id === decision.decision_id ? parsed.next_decision : null;
  const step = stopping
    ? "decision"
    : (focusStep ??
      (tab === "reason"
        ? "decision"
        : tab === "config"
          ? "model"
          : "validation"));
  const index = step.startsWith("preprocess-")
    ? Number(step.split("-")[1])
    : -1;
  const preprocessing =
    index >= 0 && index < configuration.steps.length
      ? record(configuration.steps[index])
      : null;
  const spectralStep = step === "raw" || Boolean(preprocessing);
  const validSpectra =
    spectra &&
    configuration.actual &&
    spectra.decision_id === details.decision_id &&
    spectra.configuration.method === configuration.method &&
    sameRecordedValue(spectra.configuration.steps, configuration.steps)
      ? spectra
      : null;
  const title =
    step === "decision"
      ? zh
        ? "决策与下一步"
        : "Decision and next step"
      : step === "model"
        ? methodLabel(configuration.method, zh)
        : step === "validation"
          ? zh
            ? "验证结果"
            : "Validation results"
          : step === "raw"
            ? zh
              ? "原始输入"
              : "Raw input"
            : preprocessing
              ? methodLabel(text(preprocessing.method), zh)
              : zh
                ? "预处理记录"
                : "Preprocessing record";
  const source = record(summary.data_context.source_file);
  const noteGroups = [
    [zh ? "主要观察" : "Observations", proposal.observations],
    [zh ? "待验证假设" : "Hypotheses", proposal.hypotheses],
    [zh ? "问题与不确定性" : "Uncertainties", proposal.uncertainties],
  ] as const;
  const hasNotes =
    details.rationale ||
    details.expected_outcome ||
    noteGroups.some(([, value]) => recordedStatements(value).length);
  return (
    <section
      className="process-detail-pane bg-card rounded-xl border"
      aria-label={zh ? "步骤详情" : "Step detail"}
      data-testid="round-detail"
      data-decision-id={decision.decision_id}
      data-focus-step={step}
    >
      <header className="flex flex-wrap items-center justify-between gap-2 px-4 pt-4">
        <h3 className="font-semibold">
          {title || (zh ? "方法未记录" : "Method not recorded")}
        </h3>
        <span className="text-muted-foreground text-xs">
          {executionLabel(details.status, zh)}
        </span>
      </header>
      <div className="process-detail-body space-y-4 p-4">
        {query.isError && (
          <p role="alert" className="text-sm">
            {zh
              ? "详细记录读取失败，当前显示已加载摘要。"
              : "Detail unavailable; showing the loaded summary."}
            <button
              type="button"
              className="ml-2 underline"
              onClick={() => void query.refetch()}
            >
              {zh ? "重新读取" : "Retry"}
            </button>
          </p>
        )}
        {details.status === "failed" && (
          <div
            role="alert"
            className="rounded border border-amber-500/40 p-3 text-sm"
          >
            <p>
              {zh
                ? "本轮执行失败，没有成功模型结果。"
                : "Execution failed; no successful model result."}
            </p>
            <RecordedStatements value={feedback.failure_reason} zh={zh} />
          </div>
        )}
        {spectralStep && (
          <>
            <p className="text-sm">
              {step === "raw"
                ? zh
                  ? "输入：本轮建模使用的原始 Cal 光谱。"
                  : "Input: original Cal spectra used by this round."
                : configuration.actual
                  ? zh
                    ? `行动：执行${methodLabel(text(preprocessing?.method), zh)}；下图为累计处理到本步的输出。`
                    : `Action: apply ${methodLabel(text(preprocessing?.method), zh)}; the plot shows cumulative output.`
                  : zh
                    ? "以下为计划步骤，尚无成功执行的处理结果。"
                    : "Planned step; no successful processing output."}
            </p>
            {step === "raw" && (
              <dl className="grid gap-3 text-sm sm:grid-cols-2">
                <div>
                  <dt className="text-muted-foreground text-xs">
                    {zh ? "文件" : "File"}
                  </dt>
                  <dd>
                    {source.evidence_status === "recorded"
                      ? text(source.name, 255) ||
                        (zh ? "未记录名称" : "Name not recorded")
                      : zh
                        ? "来源记录未保存文件名"
                        : "Source filename not recorded"}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground text-xs">
                    {zh ? "预测目标" : "Target"}
                  </dt>
                  <dd>
                    {text(summary.task.target_name) ||
                      text(summary.data_context.target_name) ||
                      text(summary.task.analyte) ||
                      (zh ? "未记录目标名称" : "Target name not recorded")}
                  </dd>
                </div>
              </dl>
            )}
            <RoundStepSpectra
              decision={details}
              summary={summary}
              zh={zh}
              selectedStep={step}
              spectra={validSpectra}
              spectraStatus={
                spectra && !validSpectra ? "mismatch" : spectraStatus
              }
              onRetrySpectra={onRetrySpectra}
            />
          </>
        )}
        {step.startsWith("preprocess-") && !preprocessing && (
          <p className="text-muted-foreground text-sm">
            {zh
              ? "该处理步骤未记录，请选择管线中已有的步骤。"
              : "This processing step was not recorded; select an available pipeline step."}
          </p>
        )}
        {step === "preprocessing" && (
          <p className="text-sm">
            {Array.isArray(record(feedback.actual_configuration).steps) &&
            configuration.actual
              ? zh
                ? "本轮使用原始光谱，没有预处理步骤。"
                : "Raw spectra used without preprocessing."
              : zh
                ? "未保存可核对的预处理执行记录。"
                : "No verifiable preprocessing execution recorded."}
          </p>
        )}
        {step === "model" && (
          <>
            <p className="text-sm">
              {configuration.actual
                ? zh
                  ? `行动：使用${methodLabel(configuration.method, zh)}完成本轮拟合。`
                  : `Action: fitted ${methodLabel(configuration.method, zh)}.`
                : zh
                  ? `计划：使用${methodLabel(configuration.method, zh)}建模；尚无成功执行配置。`
                  : `Plan: ${methodLabel(configuration.method, zh)}; no successful configuration.`}
            </p>
            <details>
              <summary className="cursor-pointer text-sm">
                {configuration.actual
                  ? zh
                    ? "查看实际模型参数"
                    : "Fitted model parameters"
                  : zh
                    ? "查看计划模型参数"
                    : "Planned model parameters"}
              </summary>
              <div className="mt-3">
                <RoundParameters value={configuration.parameters} zh={zh} />
              </div>
            </details>
            {Array.isArray(feedback.trials) && feedback.trials.length > 0 && (
              <details data-testid="model-parameter-search">
                <summary className="cursor-pointer text-sm">
                  {zh
                    ? "参数评价记录（Cal CV）"
                    : "Recorded parameter evaluations (Cal CV)"}
                </summary>
                <ParameterTrials value={feedback.trials} zh={zh} />
              </details>
            )}
            {hasRecordedContent(configuration.optimization) && (
              <details>
                <summary className="cursor-pointer text-sm">
                  {zh
                    ? "执行前的参数搜索计划"
                    : "Parameter search plan before execution"}
                </summary>
                <div className="mt-3">
                  <RoundParameters value={configuration.optimization} zh={zh} />
                </div>
              </details>
            )}
            <button
              type="button"
              className="text-primary text-sm underline"
              onClick={() => onTabChange("evaluation")}
            >
              {zh ? "查看本轮验证结果" : "View validation results"}
            </button>
          </>
        )}
        {step === "validation" && (
          <>
            <RoundDiagnostics
              key={details.decision_id}
              threadId={threadId}
              summary={summary}
              decision={details}
              zh={zh}
              comparison={
                <RoundComparison
                  key={details.decision_id}
                  threadId={threadId}
                  summary={summary}
                  decision={details}
                  zh={zh}
                  onSelect={onSelect}
                />
              }
            />
          </>
        )}
        {step === "decision" && (
          <>
            <RoundNextRecommendation
              current={details}
              next={next}
              summary={summary}
              terminal={isAutonomousTerminal(summary.stage)}
              loading={query.isPending}
              failed={query.isError}
              zh={zh}
              onSelect={onSelect}
            />
            {!stopping && hasNotes && (
              <details data-testid="recorded-pre-execution-reasons">
                <summary className="cursor-pointer text-sm">
                  {zh
                    ? "本轮执行前的判断与预期"
                    : "This round's recorded reasoning and expectations"}
                </summary>
                <div className="mt-3 space-y-4 text-sm">
                  <p className="text-muted-foreground text-xs">
                    {zh
                      ? "以下是智能体当时的判断与预期，不代表执行后已经验证的结论。"
                      : "Recorded pre-execution reasoning and expectations, not verified outcomes."}
                  </p>
                  {details.rationale && (
                    <div>
                      <h4 className="mb-1 font-medium">
                        {zh ? "选择理由" : "Rationale"}
                      </h4>
                      <RecordedNarrative value={details.rationale} zh={zh} />
                    </div>
                  )}
                  {noteGroups
                    .filter(([, value]) => recordedStatements(value).length > 0)
                    .map(([label, value]) => (
                      <div key={label}>
                        <h4 className="mb-1 font-medium">{label}</h4>
                        <RecordedStatements value={value} zh={zh} />
                      </div>
                    ))}
                  {details.expected_outcome && !stopping && (
                    <div>
                      <h4 className="mb-1 font-medium">
                        {zh
                          ? "执行前预期"
                          : "Expected outcome before execution"}
                      </h4>
                      <RecordedNarrative
                        value={details.expected_outcome}
                        zh={zh}
                      />
                    </div>
                  )}
                </div>
              </details>
            )}
          </>
        )}
        <details
          className="border-t pt-3 text-xs"
          data-testid="round-run-record"
        >
          <summary className="cursor-pointer">
            {zh ? "运行记录" : "Run record"}
          </summary>
          {typeof feedback.elapsed_seconds === "number" && (
            <p className="mt-3">
              {zh ? "实际耗时：约 " : "Elapsed: about "}
              {fmt(Math.round(feedback.elapsed_seconds * 10) / 10)}{" "}
              {zh ? "秒" : "s"}
            </p>
          )}
          <pre className="mt-3 max-h-80 overflow-auto break-all whitespace-pre-wrap">
            {JSON.stringify(
              {
                decision_id: details.decision_id,
                experiment_id: details.experiment_id,
                base_experiment_id: details.base_experiment_id,
                retained_experiment_id: details.retained_experiment_id,
                origin: details.origin,
                proposal: details.proposal,
                accepted_spec: details.accepted_spec,
                feedback: details.feedback,
              },
              null,
              2,
            )}
          </pre>
        </details>
      </div>
    </section>
  );
}
