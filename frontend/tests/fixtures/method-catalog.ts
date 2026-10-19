import type { MethodCard, MethodCatalog } from "@/core/knowledge/method-api";

function card(values: Partial<MethodCard>): MethodCard {
  return {
    method_id: "sg_smooth",
    title: "Savitzky-Golay 平滑",
    method_kind: "preprocessing",
    category: "smooth",
    summary_zh: "保留峰形并降低高频噪声。",
    applicability_zh: "适合有高频噪声且需要保留吸收峰的光谱。",
    limitations_zh: "过大的窗口可能削弱窄峰。",
    parameter_guidance_zh: "窗口长度必须为奇数，多项式阶数小于窗口长度。",
    planning_notes_zh: "通过校准数据上的交叉验证比较候选参数。",
    keywords: "noise smoothing 平滑",
    problem_tags: ["high_frequency_noise"],
    avoid_tags: [],
    source_url: "https://chemotools.org/methods/",
    source_section: "Parameters",
    candidate_params: [{ window: 7, order: 2 }],
    priority: 10,
    review_status: "published",
    provider: "chemotools",
    provider_class: "chemotools.smooth.SavitzkyGolayFilter",
    runtime_provider_version: "0.4.4",
    runtime_parameters: {
      window: {
        type: "integer",
        default: 7,
        minimum: 5,
        maximum: 15,
        odd: true,
      },
    },
    auto_eligible: true,
    eligibility_reason: null,
    regular_axis_required: true,
    has_default: true,
    content_sha256: "abcdef",
    evidence_id: "method:sg_smooth:abcdef",
    retrieved_on: "2026-10-04",
    ...values,
  };
}

export const methodCards: MethodCard[] = [
  card({}),
  card({
    method_id: "snv",
    title: "SNV 散射校正",
    category: "scatter",
    summary_zh: "逐条光谱校正散射差异。",
    applicability_zh: "适合颗粒大小变化引起的散射差异。",
    keywords: "scatter 散射",
    provider_class: "chemotools.scatter.StandardNormalVariate",
    regular_axis_required: false,
    runtime_parameters: {},
    candidate_params: [],
  }),
  card({
    method_id: "pls",
    title: "PLS 回归",
    method_kind: "modeling",
    category: "regression",
    summary_zh: "适合共线光谱的回归建模。",
    applicability_zh: "变量较多且存在共线性时可比较 PLS。",
    keywords: "regression 回归",
    provider: "scikit-learn",
    provider_class: "sklearn.cross_decomposition.PLSRegression",
    source_url:
      "https://scikit-learn.org/stable/modules/cross_decomposition.html",
    auto_eligible: false,
    regular_axis_required: false,
  }),
  card({
    method_id: "chemotools.adaptation.XAxisInterpolator",
    title: "光谱轴对齐",
    method_kind: "mcp_reference",
    category: "adaptation",
    summary_zh: "将光谱对齐到指定坐标轴。",
    applicability_zh: "不同采样轴需要统一时使用。",
    keywords: "alignment 对齐",
    auto_eligible: false,
    mcp_capability_id: "chemotools.adaptation.XAxisInterpolator",
    mcp_available: true,
    mcp_parameters: {
      properties: {
        method: {
          type: "string",
          default: "linear",
          description: "Interpolation method",
        },
      },
    },
  }),
];

export const methodCatalog: MethodCatalog = {
  can_edit: true,
  revision: 0,
  index_version: "method-library-test",
  cards: methodCards,
  count: methodCards.length,
  problem_tags: { high_frequency_noise: "高频噪声" },
  category_labels: {
    smooth: "平滑降噪",
    scatter: "散射校正",
    regression: "回归建模",
    adaptation: "迁移与轴对齐",
  },
  available_methods: [],
  history: [],
};
