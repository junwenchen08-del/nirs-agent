import { record, text, type DecisionView } from "./autonomous";
import { methodLabel } from "./presentation";
import { roundConfiguration } from "./round-process";

export type PipelineChange =
  | "initial"
  | "same"
  | "added"
  | "replaced"
  | "tuned"
  | "moved"
  | "unknown";
export type PipelineStep = {
  id: string;
  method: string;
  params: Record<string, unknown>;
  change: PipelineChange;
};

// Compare values independent of object key order. Array order remains significant.
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object")
    return `{${Object.entries(value)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
      .join(",")}}`;
  return JSON.stringify(value) ?? "null";
}

export function roundPipeline(decision: DecisionView, base?: DecisionView) {
  const config = roundConfiguration(decision);
  const previous = base ? roundConfiguration(base) : null;
  const stepsRecorded = Array.isArray(
    config.actual
      ? record(decision.feedback?.actual_configuration).steps
      : record(decision.selected.pipeline).preprocessing,
  );
  const priorStepsRecorded =
    previous?.actual &&
    Array.isArray(record(base?.feedback?.actual_configuration).steps);
  const comparable =
    config.actual &&
    previous?.actual &&
    Boolean(decision.base_experiment_id) &&
    decision.base_experiment_id === base?.experiment_id;
  const initial: PipelineChange = decision.base_experiment_id
    ? "unknown"
    : "initial";
  const used = new Set<number>();
  const steps: PipelineStep[] = config.steps.map((raw, index) => {
    const step = record(raw);
    const method = text(step.method);
    const params = record(step.params);
    let change: PipelineChange = initial;
    if (comparable && priorStepsRecorded) {
      const match = previous.steps.findIndex(
        (item, i) => !used.has(i) && text(record(item).method) === method,
      );
      if (match >= 0) {
        used.add(match);
        change =
          canonical(record(previous.steps[match]).params ?? {}) !==
          canonical(params)
            ? "tuned"
            : match !== index
              ? "moved"
              : "same";
      } else
        change =
          previous.steps[index] &&
          !config.steps.some(
            (item) =>
              text(record(item).method) ===
              text(record(previous.steps[index]).method),
          )
            ? "replaced"
            : "added";
    }
    return { id: `preprocess-${index}`, method, params, change };
  });
  const modelChange: PipelineChange = comparable
    ? config.method !== previous.method
      ? "replaced"
      : canonical(config.parameters) !== canonical(previous.parameters)
        ? "tuned"
        : "same"
    : initial;
  const removed =
    comparable && priorStepsRecorded
      ? previous.steps
          .filter((_, i) => !used.has(i))
          .map((step) => text(record(step).method))
      : [];
  return { ...config, stepsRecorded, steps, removed, modelChange, comparable };
}

export const pipelineChanges: Record<PipelineChange, [string, string]> = {
  initial: ["初始方案", "Initial"],
  same: ["沿用", "Unchanged"],
  added: ["新增", "Added"],
  replaced: ["替换", "Replaced"],
  tuned: ["调参", "Tuned"],
  moved: ["顺序调整", "Reordered"],
  unknown: ["待核对", "Unverified change"],
};

export function roundActionSummary(
  decision: DecisionView,
  base: DecisionView | undefined,
  zh: boolean,
): string {
  const pipeline = roundPipeline(decision, base);
  if (!pipeline.actual)
    return zh
      ? "下列为计划方案，尚无成功执行配置。"
      : "Planned pipeline; no successful configuration recorded.";
  if (!pipeline.comparable)
    return decision.base_experiment_id
      ? zh
        ? "已执行下列管线；参考记录不足，未推断步骤变化。"
        : "Pipeline executed; reference evidence is insufficient to infer changes."
      : zh
        ? "首轮实验，处理顺序如下。"
        : "First experiment; processing order is shown below.";
  const parts = pipeline.steps.map(
    (step) =>
      `${pipelineChanges[step.change][zh ? 0 : 1]}${zh ? " " : ": "}${methodLabel(step.method, zh)}`,
  );
  if (pipeline.modelChange !== "same")
    parts.push(
      `${zh ? "模型" : "Model: "}${pipelineChanges[pipeline.modelChange][zh ? 0 : 1]} ${methodLabel(pipeline.method, zh)}`,
    );
  if (pipeline.removed.length)
    parts.push(
      `${zh ? "移除" : "Removed: "} ${pipeline.removed.map((method) => methodLabel(method, zh)).join(zh ? "、" : ", ")}`,
    );
  return parts.length
    ? parts.join(zh ? "；" : "; ")
    : zh
      ? "沿用参考轮次管线。"
      : "Pipeline unchanged from the reference.";
}
