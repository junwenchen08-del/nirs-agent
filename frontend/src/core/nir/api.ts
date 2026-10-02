import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export async function loadNIRWorkflowState(
  threadId: string,
  signal?: AbortSignal,
): Promise<unknown | null> {
  const response = await fetchWithAuth(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/state`,
    { method: "GET", signal },
  );

  if (response.status === 403 || response.status === 404) {
    return null;
  }
  if (!response.ok) {
    throw new Error("Failed to load the NIR workflow state.");
  }

  const payload: unknown = await response.json();
  if (!isRecord(payload) || !isRecord(payload.values)) {
    return null;
  }
  return payload.values.nir_workflow ?? null;
}
