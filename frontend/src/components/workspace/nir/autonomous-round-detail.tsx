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
import { Trends } from "./autonomous-final-views";
import { CalSpectraComparison } from "./cal-spectra";
import { NodeDetailTabs, type NodeDetailTab } from "./node-detail-tabs";
import { ProcessFacts } from "./process-facts";
import { RoundDiagnostics } from "./round-diagnostics";
import { RoundNextRecommendation } from "./round-next-recommendation";

export function AutonomousRoundDetail({
  threadId,
  summary,
  decision,
  zh,
  onSelect,
  tab,
  onTabChange,
}: {
  threadId: string;
  summary: AutonomousSummary;
  decision: DecisionView;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
  tab: NodeDetailTab;
  onTabChange: (tab: NodeDetailTab) => void;
}) {
  const [chartGroup, setChartGroup] = useState<
    "prediction" | "spectra" | "search" | "trend"
  >("prediction");
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
  const pipeline = configuration.actual
    ? [
        ...configuration.steps.map((step) =>
          methodLabel(text(record(step).method), zh),
        ),
        methodLabel(configuration.method, zh),
      ].join(" → ")
    : decisionPipelineLabel(details.selected, zh);
  const id = `round-detail-${decision.decision_id}`;
  const metrics = record(record(feedback.metrics).val);
  return (
    <section
      className="process-detail-pane bg-card rounded-xl border"
      aria-label={zh ? "轮次详情" : "Round detail"}
      data-testid="round-detail"
      data-decision-id={decision.decision_id}
    >
      <header className="shrink-0 space-y-2 p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <h3 className="text-lg font-semibold">
            {stopping
              ? actionLabel(details.action, zh)
              : zh
                ? `第 ${fmt(details.round_index)} 轮 · ${actionLabel(details.action, zh)}`
                : `Round ${fmt(details.round_index)} · ${actionLabel(details.action, zh)}`}
          </h3>
          <span className="text-muted-foreground bg-muted rounded px-2 py-1 text-xs">
            {executionLabel(details.status, zh)}
          </span>
        </div>
        {!stopping && (
          <p className="text-sm break-words">
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
        {!stopping && Object.keys(metrics).length > 0 && (
          <div>
            <p className="text-muted-foreground text-xs">
              {zh ? "验证集（Val）" : "Validation (Val)"}
              {typeof metrics.n === "number"
                ? ` · ${metrics.n} ${zh ? "个样本" : "samples"}`
                : ""}
            </p>
            <dl className="mt-2 flex flex-wrap gap-6">
              {["RMSE", "R2", "RPD"].map((key) => (
                <div key={key}>
                  <dt className="text-muted-foreground text-xs">
                    {key === "R2" ? "R²" : key}
                  </dt>
                  <dd className="text-2xl font-semibold tabular-nums">
                    {fmt(metrics[key])}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        )}
      </header>
      <NodeDetailTabs value={tab} onChange={onTabChange} id={id} zh={zh} />
      <div
        className="process-detail-body space-y-4 p-4"
        id={`${id}-panel`}
        role="tabpanel"
        aria-labelledby={`${id}-${tab}`}
        tabIndex={0}
      >
        {query.isError && (
          <p role="alert" className="text-sm">
            {zh
              ? "详细记录读取失败，当前显示轮次摘要。"
              : "Detail read failed; showing the loaded summary."}
            <button
              type="button"
              className="ml-2 underline"
              onClick={() => void query.refetch()}
            >
              {zh ? "重新读取" : "Retry"}
            </button>
          </p>
        )}
        {tab === "overview" && (
          <>
            <h4 className="text-sm font-semibold">
              {zh ? "使用的方法与动作" : "Methods and action"}
            </h4>
            <p className="text-sm">
              {stopping ? actionLabel(details.action, zh) : pipeline}
            </p>
            <h4 className="text-sm font-semibold">
              {zh ? "本轮选择依据" : "Recorded rationale"}
            </h4>
            <RecordedNarrative value={details.rationale} zh={zh} />
            <h4 className="text-sm font-semibold">
              {zh ? "原始预期" : "Original expectation"}
            </h4>
            <RecordedNarrative value={details.expected_outcome} zh={zh} />
            <ProcessFacts
              value={{
                elapsed_seconds: feedback.elapsed_seconds,
                failure_reason: feedback.failure_reason,
                comparisons: feedback.comparisons,
                retained_experiment_id: details.retained_experiment_id,
              }}
              zh={zh}
            />
            <p className="text-muted-foreground text-xs">
              {zh
                ? "参数、各范围指标与图表、下一轮原始建议分别在上方主分区查看。"
                : "Use the tabs for parameters, scoped diagnostics and the original next proposal."}
            </p>
          </>
        )}
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

            <h4 className="text-sm font-semibold">
              {zh
                ? "固定模型参数（执行前）"
                : "Fixed model parameters (before execution)"}
            </h4>
            <ProcessFacts
              value={record(details.selected.pipeline).fixed_params}
              zh={zh}
            />
            <h4 className="text-sm font-semibold">
              {zh ? "最终拟合参数" : "Final fitted parameters"}
            </h4>
            {configuration.actual ? (
              <ProcessFacts value={configuration.parameters} zh={zh} />
            ) : (
              <p className="text-muted-foreground text-sm">
                {zh
                  ? "本轮尚无成功执行的最终参数。"
                  : "No successful fitted parameters recorded."}
              </p>
            )}
          </>
        )}
        {tab === "evaluation" && (
          <>
            {!stopping && (
              <div
                className="flex flex-wrap gap-2"
                role="group"
                aria-label={zh ? "图表分组" : "Chart groups"}
              >
                {(["prediction", "spectra", "search", "trend"] as const).map(
                  (key) => (
                    <button
                      key={key}
                      type="button"
                      aria-pressed={chartGroup === key}
                      className={`rounded-md border px-3 py-2 text-xs ${chartGroup === key ? "bg-accent text-accent-foreground border-primary" : ""}`}
                      onClick={() => setChartGroup(key)}
                    >
                      {
                        {
                          prediction: ["预测诊断", "Prediction"],
                          spectra: ["光谱对照", "Spectra"],
                          search: ["参数搜索", "Parameter search"],
                          trend: ["轮次比较", "Round comparison"],
                        }[key][zh ? 0 : 1]
                      }
                    </button>
                  ),
                )}
              </div>
            )}
            {stopping || chartGroup === "trend" ? (
              <Trends summary={summary} zh={zh} />
            ) : chartGroup === "prediction" ? (
              <RoundDiagnostics
                key={details.decision_id}
                threadId={threadId}
                summary={summary}
                decision={details}
                zh={zh}
              />
            ) : chartGroup === "spectra" ? (
              details.experiment_id ? (
                <CalSpectraComparison
                  threadId={threadId}
                  summary={summary}
                  experimentId={details.experiment_id}
                  zh={zh}
                />
              ) : (
                <p className="text-sm">
                  {zh
                    ? "本轮尚无实验光谱记录。"
                    : "No experiment spectra recorded for this round."}
                </p>
              )
            ) : (
              <ParameterTrials value={feedback.trials} zh={zh} />
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

            <RoundNextRecommendation
              current={details}
              next={next}
              terminal={isAutonomousTerminal(summary.stage)}
              loading={query.isPending}
              failed={query.isError}
              zh={zh}
              onSelect={onSelect}
            />
          </>
        )}
        <details className="border-t pt-3 text-xs">
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
      </div>
      <footer className="text-muted-foreground shrink-0 border-t px-4 py-2 text-xs">
        {zh
          ? "方法、参数、结果与建议均来自本节点记录"
          : "Methods, parameters, results and advice use this node's records"}
      </footer>
    </section>
  );
}
