import { expect, test } from "@rstest/core";

import {
  comparableScores,
  displayedPointSummary,
  pipelineDifference,
  type ProcessAttempt,
  type ProcessChart,
} from "@/core/nir/process-contract";

import examples from "../../../../../contracts/nir-process-v1.examples.json";


test("parameter and operation order changes are visible", () => {
  expect(
    pipelineDifference(
      [{ method: "sg", params: { window: 11, order: 2 } }],
      [{ params: { order: 2, window: 11 }, method: "sg" }],
    ),
  ).toBe(false);
  expect(
    pipelineDifference(
      [{ method: "sg", params: { window: 11 } }],
      [{ method: "sg", params: { window: 15 } }],
    ),
  ).toBe(true);
  expect(pipelineDifference(["snv", "sg"], ["sg", "snv"])).toBe(true);
});

test("unknown and different split identities do not join a trend", () => {
  const attempts = [null, "split-a", "split-b", "split-a"].map(
    (group, index) => ({
      number: index + 1,
      steps: [
        { step_key: "model", comparison_group_id: group, score: 4 - index },
      ],
    }),
  ) as ProcessAttempt[];
  expect(comparableScores(attempts, attempts[0]!)).toEqual([]);
  expect(
    comparableScores(attempts, attempts[1]!).map((item) => item.attempt.number),
  ).toEqual([2, 4]);
});

test("selection statistics only summarize displayed matching IDs", () => {
  const chart = {
    n_total: 10000,
    sample_ids: ["one", "two"],
    y: [10, 20],
  } as ProcessChart;
  expect(displayedPointSummary(chart, ["one", "not-displayed"])).toEqual({
    count: 1,
    mean: 10,
  });
  expect(chart.n_total).toBe(10000);
});

test("shared missing and legacy contract examples stay unranked", () => {
  expect(examples.map((item) => item.case)).toHaveLength(6);
  const missing = examples.find(
    (item) => item.case === "missing_evidence",
  )!.payload;
  expect(missing.evidence_status).toBe("unavailable");
  const legacy = examples.find((item) => item.case === "legacy")!.payload;
  expect(legacy.attempts).toEqual([]);
  const model = examples.find(
    (item) => item.case === "not_comparable",
  )!.payload;
  const attempt = { number: 1, steps: [model] } as unknown as ProcessAttempt;
  expect(comparableScores([attempt], attempt)).toEqual([]);
});
