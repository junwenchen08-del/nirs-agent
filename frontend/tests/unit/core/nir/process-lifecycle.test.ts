import { expect, test } from "@rstest/core";

import { parseSummary } from "@/core/nir/autonomous";
import { processLifecycle } from "@/core/nir/process-lifecycle";
import { modelDisplayName } from "@/core/nir-library/model-report";
import { type NIRModelVersion } from "@/core/nir-library/types";

const summary = (extra = {}) =>
  parseSummary({
    schema_version: 1,
    session_id: "s_current",
    stage: "finalized",
    ...extra,
  })!;

test("finalization never fabricates intake, freeze, approval or registration receipts", () => {
  const nodes = processLifecycle(summary(), {});
  expect(nodes.map((node) => node.id)).toEqual([
    "input",
    "inspect",
    "prepare",
    "freeze",
    "test",
    "delivery",
    "approval",
    "registration",
  ]);
  expect(nodes.find((node) => node.id === "input")?.status).toBe(
    "not_recorded",
  );
  expect(nodes.find((node) => node.id === "freeze")?.status).toBe("pending");
  expect(nodes.find((node) => node.id === "approval")?.status).toBe(
    "not_recorded",
  );
  expect(nodes.find((node) => node.id === "registration")?.status).toBe(
    "not_recorded",
  );
});

test("workflow receipts from a different search session cannot enter lifecycle details", () => {
  const nodes = processLifecycle(summary(), {
    search_session_id: "s_other",
    registered_model_id: "private-model",
    approval_status: "approved",
    ingestion_evidence: { status: "succeeded" },
  });
  expect(JSON.stringify(nodes)).not.toContain("private-model");
  expect(nodes.find((node) => node.id === "approval")?.status).toBe(
    "not_recorded",
  );
});

test("counts and freeze configuration retain exact values without touching the snapshot", () => {
  const value = summary({
    data_context: { partition_counts: { cal: 48, val: 16, test: 16 } },
    freeze: { parameters: { alpha: 0.004 }, fit_scope: "calibration" },
  });
  const before = JSON.stringify(value);
  const nodes = processLifecycle(value, {});
  expect(
    nodes.find((node) => node.id === "prepare")?.facts.partition_counts,
  ).toEqual({ cal: 48, val: 16, test: 16 });
  expect(nodes.find((node) => node.id === "freeze")?.facts.parameters).toEqual({
    alpha: 0.004,
  });
  expect(JSON.stringify(value)).toBe(before);
});

test("intake projection excludes paths, arrays and unbounded arbitrary fields", () => {
  const nodes = processLifecycle(summary(), {
    search_session_id: "s_current",
    ingestion_evidence: {
      status: "failed",
      code: "nir_ingestion_invalid",
      message: "C:/private/input.csv",
      raw_spectra: [[2]],
      source_path: "/mnt/private",
      password: "secret",
    },
  });
  expect(nodes[0]?.facts).toEqual({
    status: "failed",
    code: "nir_ingestion_invalid",
  });
});

test("a failed final Test remains failed with its consumed lease", () => {
  const node = processLifecycle(
    summary({ stage: "final_test_failed", test_status: "consumed" }),
    {},
  ).find((item) => item.id === "test");
  expect(node?.status).toBe("failed");
  expect(node?.facts.test_status).toBe("consumed");
});

test("model display names use recorded targets and never parse scientific claims from IDs", () => {
  const model = {
    model_id: "corn_moisture_snv_pls",
    method: "pls",
  } as NIRModelVersion;
  expect(modelDisplayName(model, true)).toBe(model.model_id);
  expect(
    modelDisplayName(
      {
        ...model,
        report_summary: {
          target: "水分",
          unit: null,
          sample_count: null,
          feature_count: null,
          metrics: [],
        },
      },
      true,
    ),
  ).toBe("水分 · 偏最小二乘回归（PLS）");
});
