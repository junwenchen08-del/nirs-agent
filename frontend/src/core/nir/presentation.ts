import type { ProcessCandidate } from "./process-contract";
import type { CandidateView, SelectionView } from "./process-selectors";

type Kind = "preprocessing" | "wavelength" | "model";

const METHOD_LABELS: Record<string, [string, string]> = {
  raw: ["不做预处理", "Raw spectra"],
  none: ["使用全部波长", "Full spectrum"],
  snv: ["标准正态变量变换（SNV）", "SNV"],
  msc: ["多元散射校正（MSC）", "MSC"],
  sg_smooth: ["光谱平滑", "Spectral smoothing"],
  sg: ["光谱平滑", "Spectral smoothing"],
  derivative1: ["一阶导数", "First derivative"],
  derivative2: ["二阶导数", "Second derivative"],
  norris_derivative1: [
    "Norris-Williams 一阶导数",
    "Norris-Williams first derivative",
  ],
  norris_derivative2: [
    "Norris-Williams 二阶导数",
    "Norris-Williams second derivative",
  ],
  whittaker_smooth: ["Whittaker 平滑", "Whittaker smoothing"],
  median_filter: ["中值滤波", "Median filtering"],
  robust_snv: ["稳健标准正态变量变换", "Robust SNV"],
  rnv: ["稳健正态变量变换（RNV）", "RNV"],
  emsc: ["扩展多元散射校正（EMSC）", "EMSC"],
  despike: ["孤立尖峰去除", "Spike removal"],
  detrend: ["去趋势", "Detrending"],
  asls: ["非对称最小二乘基线校正", "AsLS baseline correction"],
  airpls: ["自适应迭代基线校正（AirPLS）", "AirPLS baseline correction"],
  arpls: ["自适应重加权基线校正（ArPLS）", "ArPLS baseline correction"],
  rubberband: ["橡皮筋基线校正", "Rubberband baseline correction"],
  autoscale: ["单位方差缩放", "Autoscaling"],
  normalize: ["光谱归一化", "Spectral normalization"],
  mean_center: ["均值中心化", "Mean centering"],
  pls: ["偏最小二乘回归（PLS）", "PLS regression"],
  svr: ["支持向量回归（SVR）", "SVR"],
  ridge: ["岭回归", "Ridge regression"],
  et: ["极端随机树", "Extra Trees"],
  pcr: ["主成分回归（PCR）", "PCR"],
  pls_da: ["偏最小二乘判别（PLS-DA）", "PLS-DA"],
  cars: ["CARS 波长筛选", "CARS wavelength selection"],
  auto: ["自动选择", "Automatic selection"],
  tool_default: ["工具默认处理", "Tool default"],
};

export function methodLabel(code: string | null, zh: boolean): string {
  if (!code) return zh ? "尚未记录" : "Not recorded";
  const known = METHOD_LABELS[code.toLowerCase()];
  return known ? known[zh ? 0 : 1] : code;
}

export function pipelineLabel(steps: string[], zh: boolean): string {
  return steps.length
    ? steps.map((step) => methodLabel(step, zh)).join(" → ")
    : zh
      ? "不做预处理或未记录"
      : "No preprocessing or not recorded";
}

export function processCandidateLabel(
  candidate: ProcessCandidate,
  index: number,
  zh: boolean,
): string {
  if (Array.isArray(candidate.steps)) {
    const methods = candidate.steps.flatMap((step: unknown) => {
      if (typeof step === "string") return [step];
      if (step && typeof step === "object") {
        const fields = step as Record<string, unknown>;
        const name = fields.method ?? fields.name;
        if (typeof name === "string") return [name];
      }
      return [];
    });
    if (methods.length) return pipelineLabel(methods, zh);
  }
  if (candidate.method) return methodLabel(candidate.method, zh);
  if (/^[a-f0-9]{12,}$/i.test(candidate.candidate_id))
    return zh ? `候选方案 ${index + 1}` : `Option ${index + 1}`;
  return methodLabel(candidate.candidate_id, zh);
}

export function candidateLabel(
  candidate: CandidateView,
  kind: Kind,
  index: number,
  zh: boolean,
): string {
  if (kind === "preprocessing" && candidate.steps.length)
    return pipelineLabel(candidate.steps, zh);
  if (METHOD_LABELS[candidate.name.toLowerCase()])
    return methodLabel(candidate.name, zh);
  if (/^[a-f0-9]{12,}$/i.test(candidate.name))
    return zh ? `候选方案 ${index + 1}` : `Option ${index + 1}`;
  return candidate.name;
}

export function selectedLabel(
  selection: SelectionView,
  kind: Kind,
  zh: boolean,
): string {
  const chosen = selection.candidates.find(
    (candidate) => candidate.selected || candidate.name === selection.selected,
  );
  if (chosen)
    return candidateLabel(
      chosen,
      kind,
      selection.candidates.indexOf(chosen),
      zh,
    );
  if (selection.selected && !/^[a-f0-9]{12,}$/i.test(selection.selected))
    return methodLabel(selection.selected, zh);
  return zh ? "尚未记录" : "Not recorded";
}

export function selectionExplanation(
  kind: Kind,
  selection: Pick<
    SelectionView,
    "reasonCode" | "comparisonReasonCode" | "reason"
  >,
  zh: boolean,
): string {
  switch (selection.reasonCode) {
    case "simplest_within_one_percent_rmsecv":
      return zh
        ? "与误差最低方案的差距不超过 1%，因此采用步骤更少的方案。"
        : "Within 1% of the lowest training error, so the simpler option was chosen.";
    case "lowest_cv":
    case "lowest_cv_rmse":
      return zh
        ? "这个方案在训练数据内部验证中的误差最低。"
        : "This option had the lowest error in training data validation.";
    case "full_spectrum_for_non_pls_method":
      return zh
        ? "当前使用的不是 PLS 模型，自动 CARS 比较只适用于 PLS，因此保留全部波长。"
        : "Automatic CARS comparison applies only to PLS; this model uses the full spectrum.";
    case "cars_not_evaluated":
      return zh
        ? "这次没有运行 CARS 候选比较，继续使用全部波长。"
        : "CARS was not compared in this run, so the full spectrum was kept.";
    case "cars_improvement_below_threshold":
      return zh
        ? "CARS 的训练内误差改善未达到预设门槛，因此保留全部波长。"
        : "CARS did not meet the required training error improvement, so the full spectrum was kept.";
    case "cars_improved_tuning_rmse":
      return zh
        ? "CARS 的训练内误差改善达到预设门槛，因此采用筛选后的波长。"
        : "CARS met the required training error improvement, so its selected wavelengths were used.";
    case "explicit_selection_honored":
      return zh
        ? "本次按明确指定的波长方案执行。"
        : "The explicitly requested wavelength option was used.";
    case "pls_only_candidate":
      return zh
        ? "本次只有 PLS 候选参与模型比较。"
        : "PLS was the only model candidate in this run.";
    case "explicit_or_no_pls_baseline":
      return selection.comparisonReasonCode === "explicit_method"
        ? zh
          ? "本次按明确指定的模型方法执行，没有自动比较模型家族。"
          : "The requested model was used without automatic model family comparison."
        : zh
          ? "没有可用的 PLS 基线，采用已记录的候选模型。"
          : "No PLS baseline was available, so the recorded candidate was used.";
    case "alternative_improved_tuning_rmse":
      return zh
        ? "其他模型在训练内调参中的误差改善达到预设门槛，因此采用该模型。"
        : "Another model met the required improvement in training error and was selected.";
    case "alternative_improvement_below_threshold":
      return zh
        ? "其他模型的训练内误差改善未达到预设门槛，因此保留 PLS 基线。"
        : "Other models did not meet the required training error improvement, so PLS was kept.";
    case "explicit_method":
      return zh
        ? "本次按明确指定的模型方法执行。"
        : "The explicitly requested model was used.";
  }
  if (selection.reason && /[\u3400-\u9fff]/u.test(selection.reason))
    return selection.reason;
  return zh
    ? "详细选择原因尚无通俗说明，可展开技术信息查看原始记录。"
    : "A plain-language explanation is not available; open the technical details for the original record.";
}

export function protocolLabel(
  protocol: string | null,
  zh: boolean,
): string | null {
  if (!protocol) return null;
  if (protocol === "automated_analysis_three_way_holdout")
    return zh
      ? "训练、调参、测试分开进行"
      : "Separate training, tuning and test sets";
  return zh ? "验证方案已记录" : protocol;
}
