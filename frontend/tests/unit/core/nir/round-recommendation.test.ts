import { expect, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

import { RoundNextRecommendation } from "@/components/workspace/nir/round-next-recommendation";
import { parseDecision } from "@/core/nir/autonomous";

const current = parseDecision({
  decision_id: "d_current",
  action: "EXPLORE",
  status: "succeeded",
})!;
const proposed = {
  decision_id: "d_next",
  action: "SWITCH_MODEL",
  status: "accepted",
  rationale: "保留 SNV，尝试非线性模型",
  expected_outcome: "验证误差有望降低，仍需实际评价",
  selected: {
    pipeline: {
      preprocessing: [{ method: "snv", params: {} }],
      model_method: "svr",
      fixed_params: { kernel: "rbf", epsilon: 0.1 },
    },
    optimization: {
      strategy: "grid",
      search_space: { C: [1, 10] },
    },
  },
};
function render(next: unknown, overrides: Record<string, unknown> = {}) {
  return renderToStaticMarkup(
    createElement(RoundNextRecommendation, {
      current,
      next: parseDecision(next),
      terminal: false,
      loading: false,
      failed: false,
      zh: true,
      onSelect: () => undefined,
      ...overrides,
    }),
  );
}

test("a pending model proposal shows its plan, search range, rationale and expectation before execution", () => {
  const html = render(proposed);
  expect(html).toContain("大模型的下一轮建议");
  expect(html).toContain("建议方案");
  expect(html).toContain("支持向量回归");
  expect(html).toContain("建议模型参数");
  expect(html).toContain("径向基核");
  expect(html).toContain("参数搜索计划");
  expect(html).toContain("保留 SNV，尝试非线性模型");
  expect(html).toContain("验证误差有望降低，仍需实际评价");
  expect(html).toContain("建议已提交，等待执行");
  expect(html).not.toContain("实际执行配置");
});

test("completed execution does not replace proposed parameters with fitted parameters or scores", () => {
  const html = render({
    ...proposed,
    status: "succeeded",
    feedback: {
      actual_parameters: { C: 999 },
      metrics: { val: { RMSE: 777 } },
    },
  });
  expect(html).toContain("预期改善");
  expect(html).toContain("后续执行状态");
  expect(html).toContain("实际执行成功");
  expect(html).not.toContain("999");
  expect(html).not.toContain("777");
  expect(html).toContain("查看下一轮执行记录");
});

test("stop recommendations contain the recorded reason and no invented training plan", () => {
  const html = render({
    decision_id: "d_stop",
    action: "STOP",
    status: "stopped",
    rationale: "继续尝试收益有限，保留已有方案",
  });
  expect(html).toContain("LLM 主动停止");
  expect(html).toContain("继续尝试收益有限，保留已有方案");
  expect(html).not.toContain("建议模型参数");
  expect(html).not.toContain("参数搜索计划");
  expect(html).toContain("查看停止决策");
});

test("missing recommendations remain missing for both live and ended searches", () => {
  expect(render(null)).toContain("大模型尚未提交下一轮建议");
  expect(render(null, { terminal: true })).toContain(
    "本次搜索已结束，没有后续建议",
  );
  expect(render(null, { failed: true })).toContain("下一轮建议暂时无法读取");
  expect(render(null, { loading: true })).toContain("正在读取下一轮建议");
});

test("missing old plan fields are not presented as raw preprocessing or default parameters", () => {
  const html = render({ decision_id: "d_old", action: "SWITCH_MODEL" });
  expect(html).toContain("方案未记录");
  expect(html).toContain("未记录建议模型参数");
  expect(html).not.toContain("原始光谱");
  expect(html).not.toContain("使用算法默认参数");
});
