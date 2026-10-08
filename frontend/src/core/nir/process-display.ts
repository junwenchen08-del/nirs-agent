import { methodLabel, protocolLabel } from "./presentation";
import { type ProcessCandidate, stableJSON } from "./process-contract";

const LABELS: Record<string, [string, string]> = {
  target: ["目标变量", "Target"],
  unit: ["单位", "Unit"],
  protocol: ["验证协议", "Protocol"],
  validation_scope: ["验证范围", "Validation scope"],
  method: ["方法", "Method"],
  n_components: ["潜变量数", "Components"],
  best_n_comp: ["最佳潜变量数", "Best components"],
  n_samples: ["样本数", "Samples"],
  n_wavelengths: ["波长数", "Wavelengths"],
  finite_values: ["数值均有限", "All values finite"],
  pipeline: ["有序预处理步骤", "Ordered preprocessing"],
  steps: ["处理步骤与参数", "Steps and parameters"],
  description: ["方案说明", "Description"],
  decision: ["选择依据", "Decision"],
  reason: ["原因", "Reason"],
  selection_rule: ["选择规则", "Selection rule"],
  partitions: ["样本分组", "Partitions"],
  split: ["划分依据", "Split"],
  selected_method: ["采用方法", "Chosen method"],
  selected: ["建模是否采用", "Adopted for modeling"],
  mode: ["选择模式", "Mode"],
  minimum_required_improvement: ["最低改善要求", "Required improvement"],
  relative_RMSE_improvement: ["实际相对误差改善", "Relative error improvement"],
  adoption: ["采用判断", "Adoption"],
  quality: ["质量评估", "Quality"],
  passed: ["是否达标", "Passed"],
  grade: ["质量等级", "Grade"],
  thresholds_used: ["使用阈值", "Thresholds"],
  params: ["处理参数", "Parameters"],
  parameters: ["实际模型参数", "Model parameters"],
  evaluation_kind: ["评估范围", "Evaluation scope"],
  score: ["评估分数", "Score"],
  RMSECV: ["交叉验证误差（RMSE）", "Cross-validation RMSE"],
  cv_rmse: ["交叉验证误差（RMSE）", "Cross-validation RMSE"],
  cv_r2: ["交叉验证决定系数（R²）", "Cross-validation R²"],
  val_r2: ["调参集决定系数（R²）", "Tuning R²"],
  R2: ["决定系数（R²）", "R²"],
  R2_val: ["调参集决定系数（R²）", "Tuning R²"],
  R2_test: ["测试集决定系数（R²）", "Test R²"],
  RMSE: ["均方根误差（RMSE）", "RMSE"],
  RMSE_tuning: ["调参集误差（RMSE）", "Tuning RMSE"],
  RMSEP: ["预测误差（RMSEP）", "Prediction RMSEP"],
  best_cv_rmse: ["最低交叉验证误差（RMSE）", "Lowest CV RMSE"],
  materiality_threshold_cv_rmse: [
    "简单方案允许的误差上限",
    "Error limit for a simpler option",
  ],
  RPD: ["性能偏差比（RPD）", "RPD"],
  MAE: ["平均绝对误差（MAE）", "MAE"],
  bias: ["平均偏差", "Bias"],
  training_scope: ["最终拟合范围", "Final fitting scope"],
  adaptive_holdout_warning: ["留出集解释边界", "Holdout limitation"],
  stage: ["工作流阶段", "Workflow stage"],
  result: ["实际记录", "Result"],
  min_r2: ["最低 R²", "Minimum R²"],
  min_rpd: ["最低 RPD", "Minimum RPD"],
  expected_improvement: ["预期改善", "Expected improvement"],
  rationale: ["计划依据", "Rationale"],
  calibration: ["校准集", "Calibration"],
  tuning: ["调参集", "Tuning"],
  holdout: ["留出集", "Holdout"],
  test: ["测试集", "Test"],
  val: ["调参集", "Tuning"],
  external: ["外部验证集", "External validation"],
  inner_folds: ["内部交叉验证折数", "Inner CV folds"],
  max_components: ["潜变量数上限", "Component cap"],
  random_state: ["随机种子", "Random seed"],
  scope: ["数据范围", "Data scope"],
  window: ["窗口", "Window"],
  window_length: ["窗口长度", "Window length"],
  order: ["多项式阶数", "Polynomial order"],
  polyorder: ["多项式阶数", "Polynomial order"],
  polynomial_order: ["多项式阶数", "Polynomial order"],
  lambda_: ["平滑强度", "Smoothing strength"],
  lam: ["平滑惩罚强度", "Smoothing penalty"],
  p: ["权重／缩放指数", "Weight / scaling exponent"],
  max_iter: ["最大迭代次数", "Maximum iterations"],
  tol: ["收敛容差", "Convergence tolerance"],
  deriv: ["导数阶数", "Derivative order"],
  derivative: ["导数阶数", "Derivative order"],
  gap: ["差分间隔", "Difference gap"],
  scale: ["是否缩放", "Scale"],
  alpha: ["正则化强度", "Regularization"],
  n_estimators: ["树数量", "Trees"],
  max_depth: ["最大树深度", "Maximum tree depth"],
  C: ["误差惩罚强度", "Error penalty"],
  gamma: ["核宽度规则", "Kernel width"],
  kernel: ["核函数", "Kernel"],
  error: ["执行错误", "Execution error"],
};

const STAGES: Record<string, string> = {
  intake: "接收任务",
  data_audit: "数据审查",
  clarification: "等待数据说明",
  planning: "制定方案",
  execution: "执行建模",
  knowledge: "检索知识",
  evaluation: "评估模型",
  review: "等待审批",
  approved: "已批准",
  registered: "已注册",
  completed: "已完成",
  blocked: "需处理问题",
  pending: "待执行",
  running: "执行中",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  interrupted: "已中断",
  skipped: "已跳过",
  waiting_user: "等待确认",
};

export function formatProcessNumber(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return String(value);
  if (Math.abs(value) < 0.0001) return Number(value.toPrecision(4)).toString();
  return Number(value.toFixed(4)).toString();
}

export function factLabel(key: string, zh: boolean): string | null {
  return LABELS[key]?.[zh ? 0 : 1] ?? null;
}

export function isTechnicalFact(key: string): boolean {
  return (
    key === "error" ||
    /(?:_id|_ids|_hash|_sha256)$/.test(key) ||
    factLabel(key, true) === null
  );
}

export function factValue(key: string, value: string, zh: boolean): string;
export function factValue(
  key: string,
  value: string | null | undefined,
  zh: boolean,
): string | null | undefined;
export function factValue(key: string, value: unknown, zh: boolean): unknown;
export function factValue(key: string, value: unknown, zh: boolean): unknown {
  if (key === "adaptive_holdout_warning" && value && zh)
    return "反复查看留出集结果后，该集合不能作为未使用过的最终独立验证。";
  if (typeof value !== "string") return value;
  if (key === "method" || key === "selected_method")
    return methodLabel(value, zh);
  if (key === "protocol") return protocolLabel(value, zh) ?? value;
  if (!zh) return value;
  if (key === "stage" || key === "execution_status")
    return STAGES[value] ?? value;
  if (key === "source_tool")
    return (
      (
        {
          nir_train_auto_split_model: "自动划分数据并训练模型",
          nir_train_partitioned_model: "按指定分区训练模型",
          nir_train_model: "训练回归模型",
          nir_analyze: "自动光谱分析",
          nir_train_classifier: "训练分类模型",
        } as Record<string, string>
      )[value] ?? value
    );
  if (
    [
      "evaluation_kind",
      "validation_scope",
      "training_scope",
      "scope",
      "mode",
      "grade",
      "selection_rule",
      "kernel",
    ].includes(key)
  )
    return (
      (
        {
          cross_validation: "训练内交叉验证",
          tuning: "调参集",
          calibration: "仅校准集",
          calibration_only: "仅校准集",
          independent_holdout_not_external: "内部独立留出验证（非外部验证）",
          independent_external_validation: "独立外部验证",
          calibration_and_tuning_refit: "校准与调参集合并重拟合",
          auto: "自动选择",
          explicit: "按指定方案",
          manual: "人工指定",
          rmsecv_1pct: "误差相差不超过 1% 时优先简单方案",
          excellent: "优秀",
          good: "良好",
          fair: "一般",
          poor: "较差",
          rbf: "径向基核",
          linear: "线性核",
        } as Record<string, string>
      )[value] ?? value
    );
  return value;
}

const object = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

export function selectionDecision(
  decision: unknown,
  selection: unknown,
): unknown {
  return (
    decision ??
    object(object(selection).evaluation).selection_decision ??
    selection
  );
}

export function resolveProcessCandidate(
  candidate: ProcessCandidate,
  selection: unknown,
): ProcessCandidate {
  if (Array.isArray(candidate.steps)) return candidate;
  const record = object(selection);
  const lists = [record.candidates, object(record.recommendation).candidates];
  const matches = lists
    .flatMap((list) => (Array.isArray(list) ? list : []))
    .map(object)
    .filter(
      (item) =>
        item.candidate_id === candidate.candidate_id &&
        Array.isArray(item.steps),
    );
  if (
    !matches.length ||
    matches.some(
      (item) => stableJSON(item.steps) !== stableJSON(matches[0]!.steps),
    )
  )
    return candidate;
  return { ...candidate, steps: matches[0]!.steps };
}

export function candidateConfiguration(
  candidate: ProcessCandidate,
  zh: boolean,
): string {
  if (!Array.isArray(candidate.steps)) return "";
  return candidate.steps
    .map((step: unknown) => {
      const parameters = object(object(step).params);
      return Object.entries(parameters)
        .filter(([, value]) =>
          ["number", "string", "boolean"].includes(typeof value),
        )
        .map(([key, value]) => {
          const formatted =
            typeof value === "number"
              ? formatProcessNumber(value)
              : typeof value === "boolean"
                ? zh
                  ? value
                    ? "是"
                    : "否"
                  : String(value)
                : typeof value === "string"
                  ? value
                  : "";
          return `${factLabel(key, zh) ?? key}：${formatted}`;
        })
        .join("，");
    })
    .filter(Boolean)
    .join(" · ");
}
