import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import type {
  NIRBatchEvaluationResponse,
  NIREvaluationEntry,
  NIREvaluationScenarioCatalog,
} from "./types";

async function errorMessage(response: Response, fallback: string) {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
  } catch {
    // Fall through to the stable local message for non-JSON gateway errors.
  }
  return fallback;
}

export async function loadNirEvaluationScenarios(
  signal?: AbortSignal,
): Promise<NIREvaluationScenarioCatalog> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/nir/evaluations/scenarios`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(
      await errorMessage(response, "Failed to load NIR evaluation scenarios"),
    );
  }
  return (await response.json()) as NIREvaluationScenarioCatalog;
}

export async function runNirEvaluation(
  entries: NIREvaluationEntry[],
  signal?: AbortSignal,
): Promise<NIRBatchEvaluationResponse> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/nir/evaluations/run`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries }),
      signal,
    },
  );
  if (!response.ok) {
    throw new Error(
      await errorMessage(response, "Failed to run NIR evaluation"),
    );
  }
  return (await response.json()) as NIRBatchEvaluationResponse;
}
