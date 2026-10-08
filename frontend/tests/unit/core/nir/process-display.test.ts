import { expect, test } from "@rstest/core";

import {
  candidateConfiguration,
  factLabel,
  factValue,
  formatProcessNumber,
  isTechnicalFact,
  resolveProcessCandidate,
} from "@/core/nir/process-display";

test("links recorded configuration by exact candidate ID without changing scores", () => {
  const candidate = {
    candidate_id: "82cb8f406e10b404",
    cv_rmse: 2.127873160451022,
    selected: false,
  };
  const steps = [
    { method: "snv", params: {} },
    { method: "sg_smooth", params: { window: 7, order: 2 } },
  ];
  const resolved = resolveProcessCandidate(candidate, {
    recommendation: {
      candidates: [{ candidate_id: candidate.candidate_id, steps }],
    },
  });
  expect(resolved.steps).toEqual(steps);
  expect(resolved.cv_rmse).toBe(candidate.cv_rmse);
  expect(resolved.selected).toBe(false);
  expect(candidateConfiguration(resolved, true)).toContain("窗口：7");
  expect(candidateConfiguration(resolved, true)).toContain("多项式阶数：2");
});

test("does not invent or substitute a missing configuration", () => {
  const candidate = { candidate_id: "opaque" };
  expect(
    resolveProcessCandidate(candidate, {
      recommendation: {
        candidates: [{ candidate_id: "different", steps: [{ method: "snv" }] }],
      },
    }),
  ).toEqual(candidate);
  expect(
    resolveProcessCandidate(candidate, {
      candidates: [
        { candidate_id: "opaque", steps: [{ method: "snv" }] },
        { candidate_id: "opaque", steps: [{ method: "msc" }] },
      ],
    }),
  ).toEqual(candidate);
  expect(
    resolveProcessCandidate(
      { ...candidate, steps: [] },
      { candidates: [{ candidate_id: "opaque", steps: [{ method: "snv" }] }] },
    ).steps,
  ).toEqual([]);
});

test("translates recorded metric names, workflow stages and evaluation scopes", () => {
  expect(factLabel("best_n_comp", true)).toBe("最佳潜变量数");
  expect(factLabel("cv_r2", true)).toBe("交叉验证决定系数（R²）");
  expect(factLabel("val_r2", true)).toBe("调参集决定系数（R²）");
  expect(factValue("evaluation_kind", "cross_validation", true)).toBe(
    "训练内交叉验证",
  );
  expect(factValue("stage", "completed", true)).toBe("已完成");
  expect(factValue("source_tool", "nir_train_auto_split_model", true)).toBe(
    "自动划分数据并训练模型",
  );
  expect(factValue("evaluation_kind", "future_scope", true)).toBe(
    "future_scope",
  );
});

test("formats finite metrics without rounding tiny nonzero values to zero", () => {
  expect(formatProcessNumber(2.127873160451022)).toBe("2.1279");
  expect(formatProcessNumber(-1.232223678456475)).toBe("-1.2322");
  expect(formatProcessNumber(0)).toBe("0");
  expect(formatProcessNumber(1.2e-8)).toBe("1.2e-8");
  expect(formatProcessNumber(Infinity)).toBe("—");
  expect(formatProcessNumber(null)).toBe("—");
});

test("keeps identifiers and unknown fields in technical details", () => {
  expect(isTechnicalFact("candidate_id")).toBe(true);
  expect(isTechnicalFact("split_hash")).toBe(true);
  expect(isTechnicalFact("future_internal_field")).toBe(true);
  expect(isTechnicalFact("cv_rmse")).toBe(false);
});
