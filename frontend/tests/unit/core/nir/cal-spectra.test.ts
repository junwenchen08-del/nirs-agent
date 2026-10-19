import { expect, test, rs } from "@rstest/core";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

rs.mock("@/core/nir/process-hooks", () => ({ useProcessResource: rs.fn() }));

import { CalSpectraComparison } from "@/components/workspace/nir/cal-spectra";
import { parseCalSpectra, parseSummary } from "@/core/nir/autonomous";
import { useProcessResource } from "@/core/nir/process-hooks";

const chart = (scope: string) => ({
  chart_schema_version: 1,
  chart_id: scope,
  evidence_status: "available",
  type: "spectra",
  scope,
  x_label: "wavelength",
  x_unit: "nm",
  y_label: "spectral_value",
  x: [1100, 1102, 1104],
  series: [{ id: "sample_a", values: [1, 2, 3] }],
  n_total: 48,
  n_returned: 1,
  axis_total: 700,
  axis_returned: 3,
  sampling: "even_cal_rows_and_ordered_original_axis_indices_v1",
});
const cache = {
  schema_version: 1,
  evidence_status: "available",
  session_id: "search_a",
  experiment_id: "exp_a",
  decision_id: "decision_a",
  scope: "calibration",
  source: "actual_cal_fitted_preprocessing_state",
  configuration: { method: "svr", steps: [{ method: "snv", params: {} }] },
  before: chart("calibration_before"),
  after: chart("calibration_after"),
  private_path: "C:/private",
};

test("Cal curves keep real ordering, configuration and identities while Test stays sealed", () => {
  const parsed = parseCalSpectra(cache, "search_a", "exp_a");
  expect(parsed?.before.x).toEqual([1100, 1102, 1104]);
  expect(parsed?.after.x_unit).toBe("nm");
  expect(parsed?.configuration.steps).toEqual(cache.configuration.steps);
  expect(JSON.stringify(parsed)).not.toContain("private_path");
  const summary = parseSummary({
    schema_version: 1,
    session_id: "search_a",
    stage: "searching",
    test_status: "sealed",
    spectra_experiments: [
      {
        experiment_id: "exp_a",
        decision_id: "decision_a",
        round_index: 1,
        method: "svr",
      },
    ],
  });
  expect(summary?.spectra_experiments[0]?.experiment_id).toBe("exp_a");
});

test("wrong scope or identity and mismatched/unsafe curves are never rendered", () => {
  expect(
    parseCalSpectra({ ...cache, scope: "final_test" }, "search_a", "exp_a"),
  ).toBeNull();
  expect(parseCalSpectra(cache, "search_other", "exp_a")).toBeNull();
  expect(
    parseCalSpectra(
      { ...cache, after: { ...cache.after, x: [1100, 1103, 1104] } },
      "search_a",
      "exp_a",
    ),
  ).toBeNull();
  expect(
    parseCalSpectra(
      {
        ...cache,
        after: { ...cache.after, series: [{ id: "a", values: [1, NaN, 3] }] },
      },
      "search_a",
      "exp_a",
    ),
  ).toBeNull();
  expect(
    parseCalSpectra(
      {
        ...cache,
        after: {
          ...cache.after,
          series: Array(13).fill(cache.after.series[0]),
        },
      },
      "search_a",
      "exp_a",
    ),
  ).toBeNull();
  expect(
    parseCalSpectra(
      { ...cache, evidence_status: "not_recorded" },
      "search_a",
      "exp_a",
    ),
  ).toBeNull();
});

test("population metadata and axis units must match the displayed evidence", () => {
  for (const invalid of [
    { ...cache, before: { ...cache.before, n_total: 0 } },
    { ...cache, after: { ...cache.after, n_returned: 2 } },
    { ...cache, after: { ...cache.after, axis_total: 2 } },
    { ...cache, after: { ...cache.after, x_unit: "cm^-1" } },
  ]) {
    expect(parseCalSpectra(invalid, "search_a", "exp_a")).toBeNull();
  }
});

test("Cal comparison renders cached real curves through the read-only experiment route", () => {
  const summary = parseSummary({
    schema_version: 1,
    session_id: "search_a",
    retained_experiment_id: "exp_a",
    spectra_experiments: [
      {
        experiment_id: "exp_a",
        decision_id: "decision_a",
        round_index: 1,
        method: "svr",
      },
    ],
  })!;
  rs.mocked(useProcessResource).mockReturnValue({ data: cache } as never);
  const html = renderToStaticMarkup(
    createElement(CalSpectraComparison, {
      threadId: "thread_a",
      summary,
      zh: true,
    }),
  );
  expect(html).toContain("Cal 原始与预处理光谱");
  expect(html).toContain("不会读取 Test");
  expect(html.match(/<svg/g)?.length).toBe(2);
  expect(html).toContain("48");
  expect(html).toContain("700");
  expect(html).toContain("snv");
  expect(useProcessResource).toHaveBeenLastCalledWith(
    "thread_a",
    "autonomous/search_a/spectra/exp_a",
  );
  expect(html).not.toContain("private_path");
});

test("historical missing Cal cache is explicit and never requests recomputation", () => {
  const summary = parseSummary({ schema_version: 1, session_id: "search_a" })!;
  rs.mocked(useProcessResource).mockReturnValue({
    data: undefined,
    isPending: false,
  } as never);
  const html = renderToStaticMarkup(
    createElement(CalSpectraComparison, {
      threadId: "thread_a",
      summary,
      zh: true,
    }),
  );
  expect(html).toContain("不为历史运行重新计算");
  expect(html).not.toContain("<svg");
  expect(useProcessResource).toHaveBeenLastCalledWith("thread_a", null);
});
