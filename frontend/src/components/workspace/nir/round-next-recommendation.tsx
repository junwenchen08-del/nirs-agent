import {
  actionLabel,
  executionLabel,
  record,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { decisionPipelineLabel } from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";

import { RecordedNarrative } from "./autonomous-evidence";

export function RoundNextRecommendation({
  current,
  next,
  summary,
  terminal,
  loading,
  failed,
  zh,
  onSelect,
}: {
  current: DecisionView;
  next: DecisionView | null | undefined;
  summary: AutonomousSummary;
  terminal: boolean;
  loading: boolean;
  failed: boolean;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
}) {
  const ended = ["STOP", "RETAIN"].includes(current.action);
  const decision = ended ? current : next;
  const stop = decision && ["STOP", "RETAIN"].includes(decision.action);
  const retained = summary.trends.find(
    (row) => row.experiment_id === decision?.retained_experiment_id,
  );
  return (
    <section
      className="space-y-3 text-sm"
      data-testid="round-next-recommendation"
    >
      {decision ? (
        <>
          <p className="font-medium">
            {zh ? "后续行动：" : "Next action: "}
            {actionLabel(decision.action, zh)}
            {stop && retained
              ? zh
                ? `，保留第 ${fmt(retained.round_index)} 轮`
                : `; retain round ${fmt(retained.round_index)}`
              : ""}
          </p>
          {!stop && (
            <p>
              {zh ? "调整方案：" : "Proposed pipeline: "}
              {decisionPipelineLabel(decision.selected, zh)}
            </p>
          )}
          {decision.rationale && (
            <div>
              <h4 className="text-muted-foreground mb-1 text-xs">
                {zh ? "智能体记录的理由" : "Recorded agent rationale"}
              </h4>
              <div className="line-clamp-3 whitespace-pre-wrap">
                <RecordedNarrative value={decision.rationale} zh={zh} />
              </div>
              {decision.rationale.length > 180 && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer">
                    {zh ? "完整理由" : "Full rationale"}
                  </summary>
                  <RecordedNarrative value={decision.rationale} zh={zh} />
                </details>
              )}
            </div>
          )}
          <p className="text-muted-foreground text-xs">
            {zh ? "执行状态：" : "Execution: "}
            {decision.status === "accepted"
              ? zh
                ? "已提交，等待执行"
                : "Submitted, awaiting execution"
              : executionLabel(decision.status, zh) ||
                (zh ? "未记录" : "Not recorded")}
          </p>
          {!ended && (
            <button
              type="button"
              className="text-primary underline"
              onClick={() => onSelect(decision)}
            >
              {stop
                ? zh
                  ? "查看停止决策"
                  : "View stop decision"
                : zh
                  ? "查看下一轮执行记录"
                  : "View next round"}
            </button>
          )}
          {ended && (
            <p className="text-muted-foreground text-xs">
              {zh
                ? "这是搜索停止／保留事件，没有新的模型拟合。最终评价与审批状态请查看左侧对应阶段。"
                : "Search stop/retention event; no new fit. Final evaluation and approval are shown in their respective stages."}
            </p>
          )}
          {stop && !retained && decision.retained_experiment_id && (
            <p className="text-muted-foreground text-xs">
              {zh
                ? "已记录保留实验，关联轮次暂未加载。"
                : "Retention recorded; the linked round is not loaded."}
            </p>
          )}
        </>
      ) : (
        <p className="text-muted-foreground">
          {loading
            ? zh
              ? "正在读取后续决策…"
              : "Reading next decision…"
            : failed
              ? zh
                ? "后续决策暂时无法读取。"
                : "Next decision unavailable."
              : terminal
                ? zh
                  ? "搜索已结束，没有记录后续决策。"
                  : "Search ended; no further decision recorded."
                : zh
                  ? "智能体尚未提交后续决策。"
                  : "No next decision submitted yet."}
        </p>
      )}
      {!stop && decision && record(decision.selected).pipeline == null && (
        <p className="text-muted-foreground text-xs">
          {zh
            ? "未记录后续管线，未补写方案。"
            : "No subsequent pipeline recorded."}
        </p>
      )}
    </section>
  );
}
