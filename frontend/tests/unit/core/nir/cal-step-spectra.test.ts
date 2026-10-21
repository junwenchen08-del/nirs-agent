import { expect, test } from "@rstest/core";

import { parseCalSpectra } from "@/core/nir/autonomous";

function chart(scope: string, values: number[]) {
  return {
    type: "spectra",
    scope,
    x: [1200, 1300, 1400],
    series: [{ id: "cal-row", values }],
    n_total: 30,
    n_returned: 1,
    axis_total: 3,
    axis_returned: 3,
    x_unit: "nm",
  };
}
const make = () => ({
  schema_version: 2,
  evidence_status: "available",
  scope: "calibration",
  session_id: "s",
  experiment_id: "e",
  decision_id: "d",
  source: "actual_cal_fitted_preprocessing_state",
  configuration: {
    method: "pls",
    steps: [
      { method: "snv", params: {} },
      { method: "mean_center", params: {} },
    ],
  },
  before: chart("calibration_before", [2, 3, 5]),
  after: chart("calibration_after", [-0.4, 0.1, 0.3]),
  steps: [
    {
      index: 0,
      method: "snv",
      params: {},
      chart: chart("calibration_step_0", [-1, 0, 1]),
    },
    {
      index: 1,
      method: "mean_center",
      params: {},
      chart: chart("calibration_step_1", [-0.4, 0.1, 0.3]),
    },
  ],
});
test("cumulative snapshots preserve method order, Cal identity and final output", () => {
  const value = parseCalSpectra(make(), "s", "e")!;
  expect(value.steps?.map((step) => step.method)).toEqual([
    "snv",
    "mean_center",
  ]);
  expect(value.steps?.[0]?.chart.series?.[0]?.values).toEqual([-1, 0, 1]);
  expect(value.steps?.[1]?.chart.series).toEqual(value.after.series);
  expect(parseCalSpectra(make(), "other", "e")).toBeNull();
});
test("legacy before/after caches do not synthesize intermediate spectra", () => {
  const { steps: _steps, ...legacy } = make();
  expect(_steps).toHaveLength(2);
  const value = parseCalSpectra({ ...legacy, schema_version: 1 }, "s", "e")!;
  expect(value.steps).toBeNull();
  expect(value.after.series).toHaveLength(1);
});
test("step identity, membership, axis and final-output mismatches fail closed", () => {
  const variants = [
    (v: ReturnType<typeof make>) => {
      v.steps.reverse();
    },
    (v: ReturnType<typeof make>) => {
      v.steps[0]!.method = "msc";
    },
    (v: ReturnType<typeof make>) => {
      v.steps[0]!.chart.scope = "final_test";
    },
    (v: ReturnType<typeof make>) => {
      v.steps[0]!.chart.series[0]!.id = "other";
    },
    (v: ReturnType<typeof make>) => {
      v.steps[0]!.chart.x[0] = 1201;
    },
    (v: ReturnType<typeof make>) => {
      v.steps[0]!.chart.series[0]!.values[0] = Infinity;
    },
    (v: ReturnType<typeof make>) => {
      v.after.series[0]!.values[0] = 9;
    },
    (v: ReturnType<typeof make>) => {
      v.steps.pop();
    },
  ];
  for (const mutate of variants) {
    const v = make();
    mutate(v);
    expect(parseCalSpectra(v, "s", "e")).toBeNull();
  }
});
