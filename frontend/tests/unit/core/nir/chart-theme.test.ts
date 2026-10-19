import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "@rstest/core";

import {
  chartColors,
  chartInk,
  chartSeries,
  seriesColor,
} from "@/core/nir/chart-theme";

const CHART_COMPONENTS = [
  "src/components/workspace/nir/process-chart.tsx",
  "src/components/workspace/nir/nir-process-content.tsx",
  "src/components/workspace/nir/process-workspace.tsx",
  "src/components/workspace/nir-library/model-report-charts.tsx",
  "src/components/workspace/nir-library/dataset-workspace.tsx",
];

const chartLiterals = [
  ...chartSeries,
  ...Object.values(chartColors),
  ...Object.values(chartInk),
];

describe("series colors", () => {
  it("keeps a stable, unique categorical palette", () => {
    expect(chartSeries).toHaveLength(12);
    expect(new Set(chartSeries).size).toBe(chartSeries.length);
    for (const color of chartSeries) expect(color).toMatch(/^#[0-9a-f]{6}$/);
    expect(chartSeries[0]).toBe("#007eb8");
    expect(chartSeries[11]).toBe("#9333ea");
  });

  it("resolves the same data index to the same color across views", () => {
    expect(seriesColor(0)).toBe(chartSeries[0]);
    expect(seriesColor(11)).toBe(chartSeries[11]);
    expect(seriesColor(12)).toBe(chartSeries[0]);
    expect(seriesColor(-1)).toBe(chartSeries[11]);
    expect(seriesColor(1.5)).toBe(chartSeries[0]);
  });
});

describe("semantic colors", () => {
  it("keeps the meaning of each token, including the diagnostic select color", () => {
    expect(chartColors).toEqual({
      raw: "#64748b",
      processed: "#2563eb",
      retained: "#059669",
      stopped: "#dc2626",
      diagnostic: "#d97706",
      reference: "#ef2323",
    });
    expect(chartInk).toEqual({
      axis: "#111827",
      tick: "#334155",
      border: "#e2e8f0",
    });
  });
});

describe("single source for chart color", () => {
  for (const file of CHART_COMPONENTS)
    it(`${file} imports tokens instead of inlining colors`, () => {
      const source = readFileSync(join(process.cwd(), file), "utf-8");
      expect(
        chartLiterals.filter((literal) => source.includes(literal)),
      ).toEqual([]);
    });
});
