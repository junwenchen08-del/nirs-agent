import { expect, rs, test } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

import { RoundPipelineRow } from "@/components/workspace/nir/round-pipeline-row";
import { parseDecision, parseSummary } from "@/core/nir/autonomous";

const summary = parseSummary({
  schema_version: 1,
  session_id: "search_a",
  comparison_group_id: "group",
  data_context: { calibration_samples: 36, n_features: 8 },
})!;
function decision(id: string, rmse: number, manifest = "same") {
  return parseDecision({
    decision_id: id,
    experiment_id: `exp_${id}`,
    action: "EXPLORE",
    round_index: 1,
    status: "succeeded",
    feedback: {
      actual_configuration: { method: "ridge", steps: [] },
      actual_parameters: { alpha: 1 },
      metrics: { val: { scope: "validation", RMSE: rmse } },
      cv_settings: { manifest_id: manifest },
    },
  })!;
}
const base = decision("a", 0.4);
function render(raw: unknown, reference = base) {
  return renderToStaticMarkup(
    <RoundPipelineRow
      decision={parseDecision(raw)!}
      base={reference}
      summary={summary}
      zh
      expanded={false}
      selectedStep={null}
      onSelect={rs.fn()}
    />,
  );
}
test("the pipeline strip has a single decision entry and no duplicated result cards", () => {
  const html = render({ ...decision("b", 0.2), base_experiment_id: "exp_a" });
  expect(html).toContain('data-step-id="decision"');
  expect(html).toContain("本轮行动");
  expect(html).not.toContain("Val RMSE");
  expect(html).not.toContain("Cal CV");
});

test("accepted and failed proposals never show successful metrics or claim execution", () => {
  const current = {
    ...decision("b", 0.2),
    selected: {
      pipeline: { model_method: "svr", preprocessing: [{ method: "snv" }] },
    },
  };
  for (const status of ["accepted", "failed"]) {
    const html = render({ ...current, status });
    expect(html).toContain("计划方案");
    expect(html).toContain("尚无成功执行配置");
    expect(html).not.toContain("0.2");
    expect(html).not.toContain("实际执行管线");
  }
});
test("missing historical actual steps never turn into a raw pipeline", () => {
  const current = decision("b", 0.2);
  const html = render({
    ...current,
    feedback: {
      ...current.feedback,
      actual_configuration: { method: "ridge" },
    },
  });
  expect(html).toContain("预处理未记录");
  expect(html).not.toContain("原始光谱 · 无预处理");
});
