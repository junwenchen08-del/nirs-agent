import { expect, rs, test } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/i18n/hooks", () => ({ useI18n: () => ({ locale: "zh-CN" }) }));
rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));

import { AutonomousRoundDetail } from "@/components/workspace/nir/autonomous-round-detail";
import { parseDecision, parseSummary } from "@/core/nir/autonomous";
import { useProcessResource } from "@/core/nir/process-hooks";

const summary = parseSummary({
  schema_version: 1,
  session_id: "search_a",
  stage: "finalized",
})!;
function render(id: string, response: unknown) {
  rs.mocked(useProcessResource).mockReturnValue({
    data: response,
    isError: false,
    isPending: false,
    refetch: rs.fn(),
  } as unknown as ReturnType<typeof useProcessResource>);
  return renderToStaticMarkup(
    createElement(AutonomousRoundDetail, {
      threadId: "thread_a",
      summary,
      decision: parseDecision({ decision_id: id })!,
      zh: true,
      onSelect: rs.fn(),
    }),
  );
}

test("a restored stop outside loaded history opens decision evidence after its detail arrives", () => {
  const html = render("d_stop", {
    decision_id: "d_stop",
    action: "STOP",
    status: "stopped",
    rationale: "保留早期方案",
    retained_experiment_id: "exp_1",
  });
  expect(html).toContain("LLM 主动停止");
  expect(html).toContain("保留早期方案");
  expect(html).not.toContain("指标与图表");
  expect(html).not.toContain("尚未成功执行");
});

test("a response for another decision cannot replace the selected detail", () => {
  const html = render("d_current", {
    decision_id: "d_other",
    action: "STOP",
    rationale: "另一轮的专有说明",
    next_decision: {
      decision_id: "d_unrelated_next",
      action: "TUNE",
      rationale: "错误轮次的下一轮建议",
    },
  });
  expect(html).not.toContain("另一轮的专有说明");
  expect(html).not.toContain("错误轮次的下一轮建议");
  expect(html).not.toContain("LLM 主动停止");
  expect(html).toContain('data-decision-id="d_current"');
});
