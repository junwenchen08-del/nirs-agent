import {
  actionLabel,
  decisionPipelineLabel,
  executionLabel,
  record,
  text,
  type DecisionView,
} from "@/core/nir/autonomous";
import { methodLabel } from "@/core/nir/presentation";

import { RecordedNarrative } from "./autonomous-evidence";
import { ProcessFacts } from "./process-facts";

export function RoundNextRecommendation({
  current,
  next,
  terminal,
  loading,
  failed,
  zh,
  onSelect,
}: {
  current: DecisionView;
  next: DecisionView | null | undefined;
  terminal: boolean;
  loading: boolean;
  failed: boolean;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
}) {
  const ended = ["STOP", "RETAIN"].includes(current.action);
  const stop = next && ["STOP", "RETAIN"].includes(next.action);
  // Suggestions always come from the model's pre-execution proposal. Fitted
  // parameters and later scores belong to the next round's execution detail.
  const pipeline = record(next?.selected.pipeline);
  const optimization = record(next?.selected.optimization);
  const parameters = record(pipeline.fixed_params);
  const steps = Array.isArray(pipeline.preprocessing)
    ? pipeline.preprocessing
    : [];
  return (
    <section className="space-y-3" data-testid="round-next-recommendation">
      <h4 className="font-medium">
        {ended
          ? zh
            ? "搜索结束"
            : "Search ended"
          : zh
            ? "大模型的下一轮建议"
            : "Model recommendation for the next round"}
      </h4>
      {!ended && next ? (
        <>
          <p className="text-muted-foreground text-xs">
            {zh
              ? "以下方案、理由和预期由大模型在下一轮执行前提交。"
              : "The model submitted this plan, rationale and expectation before the next execution."}
          </p>
          <p className="font-medium">
            {zh ? "建议动作：" : "Recommended action: "}
            {actionLabel(next.action, zh)}
          </p>
          {!stop && (
            <dl className="space-y-3 rounded-lg border p-3">
              <div>
                <dt className="text-muted-foreground mb-1 text-xs">
                  {zh ? "建议方案" : "Recommended pipeline"}
                </dt>
                <dd>{decisionPipelineLabel(next.selected, zh)}</dd>
              </div>
              {steps.some(
                (step) => Object.keys(record(record(step).params)).length,
              ) && (
                <div>
                  <dt className="text-muted-foreground mb-1 text-xs">
                    {zh
                      ? "建议预处理参数"
                      : "Proposed preprocessing parameters"}
                  </dt>
                  <dd className="space-y-2">
                    {steps.map((step, index) =>
                      Object.keys(record(record(step).params)).length ? (
                        <div key={index}>
                          <p>{methodLabel(text(record(step).method), zh)}</p>
                          <ProcessFacts value={record(step).params} zh={zh} />
                        </div>
                      ) : null,
                    )}
                  </dd>
                </div>
              )}
              <div>
                <dt className="text-muted-foreground mb-1 text-xs">
                  {zh ? "建议模型参数" : "Proposed model parameters"}
                </dt>
                <dd>
                  {Object.keys(parameters).length ? (
                    <ProcessFacts value={parameters} zh={zh} />
                  ) : (
                    <span className="text-muted-foreground">
                      {Object.hasOwn(pipeline, "fixed_params")
                        ? zh
                          ? "建议未指定固定参数，参数候选见搜索计划。"
                          : "No fixed parameters specified; see the proposed search plan."
                        : zh
                          ? "未记录建议模型参数。"
                          : "Proposed model parameters were not recorded."}
                    </span>
                  )}
                </dd>
              </div>
              {Object.keys(optimization).length > 0 && (
                <div>
                  <dt className="text-muted-foreground mb-1 text-xs">
                    {zh ? "参数搜索计划" : "Proposed parameter search"}
                  </dt>
                  <dd>
                    <ProcessFacts value={optimization} zh={zh} />
                  </dd>
                </div>
              )}
            </dl>
          )}
          <div>
            <h5 className="mb-1 font-medium">
              {zh ? "建议依据" : "Rationale"}
            </h5>
            <RecordedNarrative value={next.rationale} zh={zh} />
          </div>
          {next.expected_outcome && (
            <div>
              <h5 className="mb-1 font-medium">
                {zh ? "预期改善" : "Expected outcome"}
              </h5>
              <RecordedNarrative value={next.expected_outcome} zh={zh} />
            </div>
          )}
          <div className="space-y-2 border-t pt-3">
            <p className="text-muted-foreground text-xs">
              {zh ? "后续执行状态：" : "Subsequent execution: "}
              {next.status === "accepted"
                ? zh
                  ? "建议已提交，等待执行"
                  : "Submitted, awaiting execution"
                : executionLabel(next.status, zh) ||
                  (zh ? "未记录" : "Not recorded")}
            </p>
            <button
              type="button"
              className="rounded-md border px-3 py-2"
              onClick={() => onSelect(next)}
            >
              {stop
                ? zh
                  ? "查看停止决策"
                  : "View stop decision"
                : zh
                  ? "查看下一轮执行记录"
                  : "View next round record"}
            </button>
          </div>
        </>
      ) : (
        <p className="text-muted-foreground">
          {ended
            ? zh
              ? "搜索已结束，最终评价与模型状态见“最终模型与下载”。"
              : "Search has ended; see the final model and delivery view."
            : loading
              ? zh
                ? "正在读取下一轮建议…"
                : "Reading the next recommendation…"
              : failed
                ? zh
                  ? "下一轮建议暂时无法读取，请重试。"
                  : "Could not read the next recommendation; please retry."
                : terminal
                  ? zh
                    ? "本次搜索已结束，没有后续建议。已有结果保留。"
                    : "This search has ended with no further recommendation; existing results are preserved."
                  : zh
                    ? "大模型尚未提交下一轮建议。"
                    : "The model has not submitted the next recommendation yet."}
        </p>
      )}
    </section>
  );
}
