import { expect, test } from "@rstest/core";

import {
  appendEvaluationHistory,
  evaluationStorageKey,
} from "@/core/nir-evaluations/history";

test("scopes NIR evaluation storage keys by user", () => {
  const baseKey = "deerflow:nir-evaluation-history:v1";

  expect(evaluationStorageKey(baseKey, "user-a")).not.toBe(
    evaluationStorageKey(baseKey, "user-b"),
  );
  expect(evaluationStorageKey(baseKey, "user/a")).toBe(
    `${baseKey}:user:user%2Fa`,
  );
});

test("keeps newest NIR evaluation summaries first and caps history", () => {
  const existing = Array.from({ length: 12 }, (_, index) => ({
    generatedAt: `2026-01-${String(index + 1).padStart(2, "0")}T00:00:00Z`,
    total: 1,
    passed: 1,
    passRate: 1,
    averageScore: 100,
    policyViolationCount: 0,
  }));
  const next = {
    generatedAt: "2026-02-01T00:00:00Z",
    total: 2,
    passed: 1,
    passRate: 0.5,
    averageScore: 75,
    policyViolationCount: 1,
  };

  const result = appendEvaluationHistory(existing, next);

  expect(result).toHaveLength(12);
  expect(result[0]).toEqual(next);
  expect(result).not.toContain(existing[11]);
});
