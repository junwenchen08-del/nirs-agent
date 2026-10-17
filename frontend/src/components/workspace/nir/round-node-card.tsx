"use client";

import {
  actionLabel,
  decisionPipelineLabel,
  executionLabel,
  experimentMarkers,
  record,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";

export function RoundNodeCard({
  decision,
  summary,
  zh,
  selected,
  onSelect,
  rows,
}: {
  decision: DecisionView;
  summary: AutonomousSummary;
  zh: boolean;
  selected: boolean;
  onSelect: (decision: DecisionView) => void;
  rows: DecisionView[];
}) {
  const stopping = ["STOP", "RETAIN"].includes(decision.action);
  const base = rows.find(
    (row) =>
      row.experiment_id && row.experiment_id === decision.base_experiment_id,
  );
  const retained = rows.find(
    (row) =>
      row.experiment_id &&
      row.experiment_id === decision.retained_experiment_id,
  );
  const metrics = record(record(decision.feedback?.metrics).val);
  return (
    <button
      type="button"
      onClick={() => onSelect(decision)}
      aria-pressed={selected}
      data-testid="round-node"
      data-decision-id={decision.decision_id}
      className={`nodrag nopan w-full rounded-xl border p-3 text-left text-sm transition-colors ${selected ? "border-primary bg-accent ring-primary/15 ring-2" : "bg-card hover:bg-accent/50"}`}
    >
      <span className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium">
          {stopping
            ? actionLabel(decision.action, zh)
            : zh
              ? `第 ${fmt(decision.round_index)} 轮 · ${actionLabel(decision.action, zh)}`
              : `Round ${fmt(decision.round_index)} · ${actionLabel(decision.action, zh)}`}
        </span>
        <span className="text-muted-foreground text-xs">
          {stopping
            ? zh
              ? "已记录"
              : "Recorded"
            : executionLabel(decision.status, zh)}
        </span>
      </span>
      <span className="mt-2 line-clamp-2 break-words">
        {stopping
          ? retained
            ? zh
              ? `明确保留第 ${fmt(retained.round_index)} 轮`
              : `Explicitly retain round ${fmt(retained.round_index)}`
            : decision.retained_experiment_id
              ? zh
                ? "已指定保留实验"
                : "Retained experiment recorded"
              : zh
                ? "尚无保留实验记录"
                : "No recorded retention"
          : decisionPipelineLabel(decision.selected, zh)}
      </span>
      {!stopping && typeof metrics.RMSE === "number" && (
        <span className="mt-2 block tabular-nums">
          Val RMSE {fmt(metrics.RMSE)}
        </span>
      )}
      {base && (
        <span className="text-muted-foreground mt-1 block text-xs">
          {zh
            ? `反馈来源：第 ${fmt(base.round_index)} 轮`
            : `Feedback from round ${fmt(base.round_index)}`}
        </span>
      )}
      {decision.base_experiment_id && !base && (
        <span className="text-muted-foreground mt-1 block text-xs">
          {zh ? "来源轮次尚未加载" : "Source round not loaded"}
        </span>
      )}
      {!stopping && decision.experiment_id && (
        <span className="mt-2 flex flex-wrap gap-1">
          {experimentMarkers(summary, decision.experiment_id, zh).map(
            (marker) => (
              <span
                key={marker}
                className="rounded bg-emerald-500/10 px-2 py-0.5 text-xs text-emerald-700 dark:text-emerald-300"
              >
                {marker}
              </span>
            ),
          )}
        </span>
      )}
    </button>
  );
}
