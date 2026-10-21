"use client";

import {
  actionLabel,
  executionLabel,
  experimentMarkers,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import {
  pipelineChanges,
  roundPipeline,
  roundActionSummary,
  type PipelineChange,
} from "@/core/nir/round-pipeline";

function Change({ value, zh }: { value: PipelineChange; zh: boolean }) {
  return (
    <span
      className={`mt-2 block text-xs ${["added", "replaced", "tuned", "moved"].includes(value) ? "text-primary font-semibold" : "text-muted-foreground"}`}
    >
      {pipelineChanges[value][zh ? 0 : 1]}
    </span>
  );
}

export function RoundPipelineRow({
  decision,
  base,
  summary,
  zh,
  expanded,
  selectedStep,
  onSelect,
}: {
  decision: DecisionView;
  base?: DecisionView;
  summary: AutonomousSummary;
  zh: boolean;
  expanded: boolean;
  selectedStep: string | null;
  onSelect: (step?: string) => void;
}) {
  const pipeline = roundPipeline(decision, base);
  return (
    <div
      className={`overflow-hidden rounded-xl border ${expanded ? "border-primary/60 ring-primary/10 ring-2" : ""}`}
      data-testid="pipeline-row"
      data-decision-id={decision.decision_id}
    >
      <button
        type="button"
        aria-expanded={expanded}
        aria-controls={`expanded-${decision.decision_id}`}
        data-testid="round-node"
        data-decision-id={decision.decision_id}
        onClick={() => onSelect()}
        className={`flex w-full flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 text-left ${expanded ? "bg-accent/50" : "bg-muted/20 hover:bg-accent/30"}`}
      >
        <strong className="text-sm">
          {zh
            ? `第 ${fmt(decision.round_index)} 轮`
            : `Round ${fmt(decision.round_index)}`}
        </strong>
        <span className="text-muted-foreground text-xs">
          {actionLabel(decision.action, zh)} ·{" "}
          {executionLabel(decision.status, zh)}
        </span>
        {decision.base_experiment_id && (
          <span className="text-muted-foreground text-xs">
            {base
              ? zh
                ? `参考第 ${fmt(base.round_index)} 轮`
                : `Based on round ${fmt(base.round_index)}`
              : zh
                ? "参考轮次尚未加载"
                : "Reference not loaded"}
          </span>
        )}
        {decision.experiment_id &&
          experimentMarkers(summary, decision.experiment_id, zh).map(
            (marker) => (
              <span
                key={marker}
                className="rounded bg-emerald-500/10 px-2 py-1 text-xs text-emerald-700 dark:text-emerald-300"
              >
                {marker}
              </span>
            ),
          )}
        <span className="text-muted-foreground ml-auto text-xs">
          {expanded
            ? zh
              ? "收起详情 ▴"
              : "Collapse ▴"
            : zh
              ? "展开详情 ▾"
              : "Expand ▾"}
        </span>
      </button>
      <div className="space-y-4 p-4">
        <p className="text-sm leading-6" data-testid="round-action-summary">
          {zh ? "本轮行动：" : "Round action: "}
          {roundActionSummary(decision, base, zh)}
        </p>
        <ol
          className="round-pipeline-strip"
          data-testid="round-pipeline-strip"
          aria-label={zh ? "本轮处理顺序" : "Round processing order"}
        >
          <li>
            <button
              type="button"
              data-step-id="raw"
              aria-pressed={selectedStep === "raw"}
              onClick={() => onSelect("raw")}
            >
              {zh ? "原始输入" : "Raw input"}
            </button>
          </li>
          {pipeline.steps.length ? (
            pipeline.steps.map((item) => (
              <li key={item.id}>
                <button
                  type="button"
                  data-step-id={item.id}
                  aria-pressed={selectedStep === item.id}
                  onClick={() => onSelect(item.id)}
                >
                  {methodLabel(item.method, zh)}
                  <Change value={item.change} zh={zh} />
                </button>
              </li>
            ))
          ) : (
            <li>
              <button
                type="button"
                data-step-id="preprocessing"
                aria-pressed={selectedStep === "preprocessing"}
                onClick={() => onSelect("preprocessing")}
              >
                {pipeline.stepsRecorded
                  ? zh
                    ? "原始光谱 · 无预处理"
                    : "Raw spectra · no preprocessing"
                  : zh
                    ? "预处理未记录"
                    : "Preprocessing not recorded"}
              </button>
            </li>
          )}
          <li>
            <button
              type="button"
              data-step-id="model"
              aria-pressed={selectedStep === "model"}
              onClick={() => onSelect("model")}
            >
              {methodLabel(pipeline.method, zh) ||
                (zh ? "方法未记录" : "Method not recorded")}
              <Change value={pipeline.modelChange} zh={zh} />
            </button>
          </li>
          <li>
            <button
              type="button"
              data-step-id="validation"
              aria-pressed={selectedStep === "validation"}
              onClick={() => onSelect("validation")}
            >
              {zh ? "验证结果" : "Validation"}
            </button>
          </li>
          <li>
            <button
              type="button"
              data-step-id="decision"
              aria-pressed={selectedStep === "decision"}
              onClick={() => onSelect("decision")}
            >
              {zh ? "决策与下一步" : "Decision and next step"}
            </button>
          </li>
        </ol>
      </div>
    </div>
  );
}
