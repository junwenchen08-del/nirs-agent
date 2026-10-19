import { expect, test } from "@rstest/core";

import {
  processCandidateLabel,
  candidateLabel,
  protocolLabel,
  selectedLabel,
  selectionExplanation,
} from "@/core/nir/presentation";
import type { SelectionView } from "@/core/nir/process-selectors";

const selection: SelectionView = {
  selected: "82cb8f406e10b404",
  reasonCode: "simplest_within_one_percent_rmsecv",
  comparisonReasonCode: null,
  reason: null,
  rule: "rmsecv_1pct",
  improvement: null,
  minImprovement: null,
  candidates: [
    {
      name: "82cb8f406e10b404",
      steps: ["snv"],
      score: 0.46,
      scoreKey: "cv_rmse",
      selected: true,
      reason: null,
    },
  ],
};

test("durable candidates display the recorded ordered methods instead of a hash", () => {
  expect(
    processCandidateLabel(
      {
        candidate_id: "82cb8f406e10b404",
        steps: [
          { method: "snv" },
          { method: "sg_smooth", params: { window: 11 } },
        ],
      },
      0,
      true,
    ),
  ).toBe("标准正态变量变换（SNV） → 光谱平滑");
  expect(
    processCandidateLabel({ candidate_id: "82cb8f406e10b404" }, 50, true),
  ).toBe("候选方案 51");
});

test("shows a named preprocessing option and a plain-language reason", () => {
  expect(selectedLabel(selection, "preprocessing", true)).toBe(
    "标准正态变量变换（SNV）",
  );
  expect(selectionExplanation("preprocessing", selection, true)).toContain(
    "差距不超过 1%",
  );
  expect(selectionExplanation("preprocessing", selection, false)).toContain(
    "simpler option",
  );
});

test("does not expose a candidate hash as the primary label", () => {
  expect(
    candidateLabel(
      { ...selection.candidates[0]!, steps: [] },
      "preprocessing",
      0,
      true,
    ),
  ).toBe("候选方案 1");
});

test("keeps unknown reasons and protocols out of the primary explanation", () => {
  expect(
    selectionExplanation(
      "model",
      {
        ...selection,
        reasonCode: "future_rule",
        reason: "Opaque internal note",
      },
      true,
    ),
  ).toContain("原始记录");
  expect(protocolLabel("future_protocol", true)).toBe("验证方案已记录");
});
