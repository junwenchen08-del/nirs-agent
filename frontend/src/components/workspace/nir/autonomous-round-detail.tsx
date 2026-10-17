"use client";

import { useEffect, useRef, useState } from "react";

import {
  actionLabel,
  decisionPipelineLabel,
  executionLabel,
  isAutonomousTerminal,
  parseDecision,
  record,
  text,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";
import { roundConfiguration } from "@/core/nir/round-process";

import { ParameterTrials, RecordedNarrative } from "./autonomous-evidence";
import { CalSpectraComparison } from "./cal-spectra";
import { ProcessFacts } from "./process-facts";
import { RoundDiagnostics } from "./round-diagnostics";
import { RoundNextRecommendation } from "./round-next-recommendation";

export function AutonomousRoundDetail({
  threadId,
  summary,
  decision,
  zh,
  onSelect,
}: {
  threadId: string;
  summary: AutonomousSummary;
  decision: DecisionView;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
}) {
  const [selectedTab, setTab] = useState<"config" | "evaluation" | "reason">(
    "evaluation",
  );
  const [spectraOpen, setSpectraOpen] = useState(false);
  const [trialsOpen, setTrialsOpen] = useState(false);
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
  const tab = stopping ? "reason" : selectedTab;
  const configuration = roundConfiguration(details);
  const proposal = details.proposal ?? {};
  const feedback = details.feedback ?? {};
  const next =
    parsed?.decision_id === decision.decision_id ? parsed.next_decision : null;
  const pipeline = configuration.actual
    ? [
        ...configuration.steps.map((step) =>
          methodLabel(text(record(step).method), zh),
        ),
        methodLabel(configuration.method, zh),
      ].join(" → ")
    : decisionPipelineLabel(details.selected, zh);
  const id = `round-detail-${decision.decision_id}`;
  return (
    <section
      className="bg-card min-w-0 space-y-4 rounded-xl border p-4"
      aria-label={zh ? "轮次详情" : "Round detail"}
      data-testid="round-detail"
      data-decision-id={decision.decision_id}
    >
      <header>
        <h3 className="text-lg font-medium">
          {stopping
            ? actionLabel(details.action, zh)
            : zh
              ? `第 ${fmt(details.round_index)} 轮 · ${actionLabel(details.action, zh)}`
              : `Round ${fmt(details.round_index)} · ${actionLabel(details.action, zh)}`}
        </h3>
        <p className="text-muted-foreground mt-1 text-xs">
          {stopping
            ? zh
              ? "停止选择已记录"
              : "Recorded stop selection"
            : executionLabel(details.status, zh)}
          {typeof feedback.elapsed_seconds === "number"
            ? ` · ${fmt(feedback.elapsed_seconds)} ${zh ? "秒" : "s"}`
            : ""}
        </p>
        {!stopping && (
          <p className="bg-muted mt-3 rounded-lg p-3 text-sm">
            {configuration.actual
              ? zh
                ? "实际方案："
                : "Executed: "
              : zh
                ? "计划方案："
                : "Planned: "}
            {pipeline}
          </p>
        )}
      </header>
      {query.isError && (
        <div role="alert" className="text-sm">
          {zh
            ? "详细记录读取失败，当前显示已读取的轮次摘要。"
            : "Detail read failed; showing the loaded round summary."}
          <button
            className="ml-2 underline"
            type="button"
            onClick={() => void query.refetch()}
          >
            {zh ? "重新读取" : "Retry"}
          </button>
        </div>
      )}
      <div
        className="flex flex-wrap gap-1 border-b"
        role="tablist"
        aria-label={zh ? "轮次信息" : "Round information"}
      >
        {(["config", "evaluation", "reason"] as const)
          .filter((key) => !stopping || key === "reason")
          .map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              id={`${id}-${key}`}
              aria-selected={tab === key}
              aria-controls={`${id}-panel`}
              onClick={() => setTab(key)}
              className={`border-b-2 px-3 py-2 text-sm ${tab === key ? "border-primary font-medium" : "text-muted-foreground border-transparent"}`}
            >
              {
                {
                  config: ["方案与参数", "Configuration"],
                  evaluation: ["指标与图表", "Metrics and plots"],
                  reason: ["决策依据", "Decision evidence"],
                }[key][zh ? 0 : 1]
              }
            </button>
          ))}
      </div>
      <div
        id={`${id}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-${tab}`}
        className="min-w-0 space-y-4"
      >
        {tab === "config" && (
          <>
            <p className="text-sm font-medium">
              {configuration.actual
                ? zh
                  ? "实际执行配置"
                  : "Executed configuration"
                : zh
                  ? "已接受的计划配置（尚未成功执行）"
                  : "Accepted plan (not successfully executed)"}
            </p>
            <dl className="grid grid-cols-[6rem_minmax(0,1fr)] gap-x-3 text-sm">
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "数据范围" : "Data scope"}
              </dt>
              <dd className="border-b py-2">
                <ProcessFacts value={summary.data_context} zh={zh} />
              </dd>
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "预处理顺序" : "Preprocessing"}
              </dt>
              <dd className="border-b py-2">
                {configuration.steps.length ? (
                  <ol className="space-y-2">
                    {configuration.steps.map((step, index) => (
                      <li key={index}>
                        <span className="font-medium">
                          {index + 1}.{" "}
                          {methodLabel(text(record(step).method), zh)}
                        </span>
                        <ProcessFacts value={record(step).params} zh={zh} />
                      </li>
                    ))}
                  </ol>
                ) : Array.isArray(
                    record(details.selected.pipeline).preprocessing,
                  ) ? (
                  zh ? (
                    "原始光谱，无预处理"
                  ) : (
                    "Raw spectra, no preprocessing"
                  )
                ) : zh ? (
                  "未记录"
                ) : (
                  "Not recorded"
                )}
              </dd>
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "建模算法" : "Model"}
              </dt>
              <dd className="border-b py-2">
                {methodLabel(configuration.method, zh) || "—"}
              </dd>
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "波长选择" : "Wavelength selection"}
              </dt>
              <dd className="border-b py-2">
                {zh
                  ? "本轮未记录波长筛选操作"
                  : "No wavelength selection operation recorded for this round"}
              </dd>
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "参数" : "Parameters"}
              </dt>
              <dd className="min-w-0 border-b py-2">
                <ProcessFacts value={configuration.parameters} zh={zh} />
              </dd>
              <dt className="text-muted-foreground border-b py-2">
                {zh ? "搜索设置" : "Search plan"}
              </dt>
              <dd className="min-w-0 border-b py-2">
                <ProcessFacts value={configuration.optimization} zh={zh} />
              </dd>
            </dl>
            <details
              onToggle={(event) => setTrialsOpen(event.currentTarget.open)}
            >
              <summary className="cursor-pointer text-sm">
                {zh ? "参数候选与交叉验证" : "Parameter trials and CV"}
              </summary>
              {trialsOpen && (
                <ParameterTrials value={feedback.trials} zh={zh} />
              )}
            </details>
            <details
              onToggle={(event) => setSpectraOpen(event.currentTarget.open)}
            >
              <summary className="cursor-pointer text-sm">
                {zh
                  ? "预处理前后光谱"
                  : "Spectra before and after preprocessing"}
              </summary>
              {spectraOpen && details.experiment_id && (
                <CalSpectraComparison
                  threadId={threadId}
                  summary={summary}
                  experimentId={details.experiment_id}
                  zh={zh}
                />
              )}
            </details>
          </>
        )}
        {tab === "evaluation" && (
          <>
            <RoundDiagnostics
              key={details.decision_id}
              threadId={threadId}
              summary={summary}
              decision={details}
              zh={zh}
            />
            {Object.keys(record(feedback.comparisons)).length > 0 && (
              <details>
                <summary className="cursor-pointer text-sm">
                  {zh
                    ? "与已有方案比较"
                    : "Comparison with recorded experiments"}
                </summary>
                <ProcessFacts value={feedback.comparisons} zh={zh} />
              </details>
            )}
          </>
        )}
        {tab === "reason" && (
          <>
            <section>
              <h4 className="mb-2 text-sm font-medium">
                {zh
                  ? "执行前记录的选择依据"
                  : "Recorded rationale before execution"}
              </h4>
              <div className="text-sm">
                <RecordedNarrative value={details.rationale} zh={zh} />
              </div>
            </section>
            {!stopping && (
              <section>
                <h4 className="mb-2 text-sm font-medium">
                  {zh ? "执行前的预期" : "Expected outcome before execution"}
                </h4>
                <div className="text-sm">
                  <RecordedNarrative value={details.expected_outcome} zh={zh} />
                </div>
              </section>
            )}
            <ProcessFacts
              value={
                stopping
                  ? {
                      stop_evidence: proposal.stop_evidence,
                      retained_experiment_id: details.retained_experiment_id,
                      stop_source: summary.stop_source,
                    }
                  : {
                      observations: proposal.observations,
                      uncertainties: proposal.uncertainties,
                      hypotheses: proposal.hypotheses,
                      expected_improvement: feedback.expected_improvement,
                      failure_reason: feedback.failure_reason,
                    }
              }
              zh={zh}
            />
            <details>
              <summary className="cursor-pointer text-sm">
                {zh
                  ? "其他备选方案与未采用原因"
                  : "Alternatives and recorded rejection reasons"}
              </summary>
              <p className="text-muted-foreground my-2 text-xs">
                {zh
                  ? "提出备选方案不表示已经训练或完成验证。"
                  : "A proposed alternative is not a trained or validated experiment."}
              </p>
              <ProcessFacts
                value={{
                  alternatives: proposal.alternatives,
                  not_selected_reasons: proposal.not_selected_reasons,
                }}
                zh={zh}
              />
            </details>
          </>
        )}
      </div>
      <footer className="space-y-2 border-t pt-3 text-sm">
        <RoundNextRecommendation
          current={details}
          next={next}
          terminal={isAutonomousTerminal(summary.stage)}
          loading={query.isPending}
          failed={query.isError}
          zh={zh}
          onSelect={onSelect}
        />
        <details className="text-xs">
          <summary className="cursor-pointer">
            {zh
              ? "追溯信息与完整配置"
              : "Provenance and accepted configuration"}
          </summary>
          <ProcessFacts
            value={{
              decision_id: details.decision_id,
              experiment_id: details.experiment_id,
              base_experiment_id: details.base_experiment_id,
              origin: details.origin,
              accepted_spec: details.accepted_spec,
              cv_settings: feedback.cv_settings,
            }}
            zh={zh}
          />
        </details>
      </footer>
    </section>
  );
}
