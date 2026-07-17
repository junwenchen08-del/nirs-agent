import type { NIREvaluationHistoryEntry, NIREvaluationSummary } from "./types";

const HISTORY_LIMIT = 12;

export function evaluationStorageKey(baseKey: string, userId: string): string {
  return `${baseKey}:user:${encodeURIComponent(userId)}`;
}

export function historyEntryFromSummary(
  summary: NIREvaluationSummary,
): NIREvaluationHistoryEntry {
  return {
    generatedAt: summary.generated_at,
    total: summary.total,
    passed: summary.passed,
    passRate: summary.pass_rate,
    averageScore: summary.average_score,
    policyViolationCount: summary.policy_violation_count,
  };
}

export function appendEvaluationHistory(
  history: NIREvaluationHistoryEntry[],
  entry: NIREvaluationHistoryEntry,
): NIREvaluationHistoryEntry[] {
  return [entry, ...history].slice(0, HISTORY_LIMIT);
}
