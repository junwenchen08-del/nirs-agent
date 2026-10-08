import { fetch as fetchWithAuth } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import { type ProcessSummary } from "./process-contract";

export class ProcessAPIError extends Error {
  constructor(public status: number) {
    super(`Process request failed (${status})`);
  }
}

export async function loadProcess<T>(
  threadId: string,
  resource: string,
  signal?: AbortSignal,
  previous?: ProcessSummary,
): Promise<T> {
  const response = await fetchWithAuth(
    `${getBackendBaseURL()}/api/threads/${encodeURIComponent(threadId)}/nir-process/${resource}`,
    {
      signal,
      headers: previous
        ? {
            "If-None-Match": `"${previous.run_id}-${previous.offset ?? 0}-${previous.revision}"`,
          }
        : undefined,
    },
  );
  if (response.status === 304 && previous) return previous as T;
  if (!response.ok) throw new ProcessAPIError(response.status);
  const result: unknown = await response.json();
  if (!result || typeof result !== "object")
    throw new Error("Invalid process response");
  if (
    previous &&
    "revision" in result &&
    typeof result.revision === "number" &&
    result.revision < previous.revision
  )
    return previous as T;
  return result as T;
}
