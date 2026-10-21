import { expect, rs, test } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));

import { ProcessDataOverview } from "@/components/workspace/nir/process-data-overview";
import { RoundComparison } from "@/components/workspace/nir/round-comparison";
import { parseDecision, parseSummary } from "@/core/nir/autonomous";
import { useProcessResource } from "@/core/nir/process-hooks";
import { processLifecycle } from "@/core/nir/process-lifecycle";

const source = {
  name: "tablets.MAT",
  format: "MAT",
  size_bytes: 8192,
  sha256: "a".repeat(64),
  evidence_status: "recorded",
};
const summary = parseSummary({
  schema_version: 1,
  session_id: "search_1",
  stage: "searching",
  comparison_group_id: "group_1",
  task: { analyte: "水分" },
  data_context: {
    calibration_samples: 36,
    n_features: 8,
    unit: "%",
    source_file: source,
  },
  trends: [
    {
      experiment_id: "exp_1",
      decision_id: "d_1",
      method: "ridge",
      round_index: 1,
    },
  ],
})!;
const previous = {
  decision_id: "d_1",
  experiment_id: "exp_1",
  action: "EXPLORE",
  status: "succeeded",
  feedback: {
    cv_settings: { manifest_id: "cv_same" },
    actual_configuration: { method: "ridge", steps: [] },
    actual_parameters: { alpha: 1 },
    metrics: {
      val: { scope: "validation", RMSE: 0.4 },
      cal_cv: { scope: "calibration_cv", RMSE: 0.5 },
    },
  },
};
const current = parseDecision({
  ...previous,
  decision_id: "d_2",
  experiment_id: "exp_2",
  base_experiment_id: "exp_1",
  feedback: {
    ...previous.feedback,
    metrics: {
      val: { scope: "validation", RMSE: 0.2 },
      cal_cv: { scope: "calibration_cv", RMSE: 0.3 },
    },
  },
})!;
function compare(response: unknown) {
  rs.mocked(useProcessResource).mockReturnValue({
    data: response,
    isPending: false,
    isError: false,
    refetch: rs.fn(),
  } as unknown as ReturnType<typeof useProcessResource>);
  return renderToStaticMarkup(
    <RoundComparison
      threadId="thread_1"
      summary={summary}
      decision={current}
      zh
      onSelect={rs.fn()}
    />,
  );
}

test("source overview uses fixed preparation evidence and shortens hashes without inventing intake execution", () => {
  const nodes = processLifecycle(summary, {});
  const html = renderToStaticMarkup(
    <ProcessDataOverview summary={summary} node={nodes[0]!} zh />,
  );
  expect(nodes[0]?.status).toBe("source_recorded");
  expect(html).toContain("tablets.MAT");
  expect(html).toContain("水分");
  expect(html).toContain("8,192");
  expect(html).toContain("a".repeat(12));
  expect(html).not.toContain("a".repeat(64));
});

test("historical missing source metadata stays missing even when finalized", () => {
  const old = parseSummary({
    schema_version: 1,
    session_id: "old",
    stage: "finalized",
  })!;
  const nodes = processLifecycle(old, {
    search_session_id: "another",
    ingestion_evidence: { filename: "wrong.csv" },
  });
  const html = renderToStaticMarkup(
    <ProcessDataOverview summary={old} node={nodes[0]!} zh />,
  );
  expect(html).toContain("来源记录未提供");
  expect(html).not.toContain("wrong.csv");
  expect(nodes[0]?.status).toBe("not_recorded");
});

test("matching execution and CV identities permit scoped deltas", () => {
  const html = compare(previous);
  expect(html).toContain("降低 0.2");
  expect(html).toContain("智能体参考");
  expect(html).not.toContain("不计算改善幅度");
});

test("different CV manifests retain values without calculated improvements", () => {
  const html = compare({
    ...previous,
    feedback: {
      ...previous.feedback,
      cv_settings: { manifest_id: "cv_other" },
    },
  });
  expect(html).toContain("不计算改善幅度");
  expect(html).not.toContain("-0.2");
});

test("stale comparison response cannot replace the requested experiment", () => {
  const html = compare({
    ...previous,
    decision_id: "wrong",
    experiment_id: "wrong",
  });
  expect(html).toContain("身份不匹配");
  expect(html).not.toContain("<table");
});

test("historical missing preprocessing is not described as an executed raw pipeline", () => {
  const html = compare({
    ...previous,
    feedback: {
      ...previous.feedback,
      actual_configuration: { method: "ridge" },
    },
  });
  expect(html).not.toContain("原始光谱 · 无预处理");
});
