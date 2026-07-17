import { beforeEach, describe, expect, test, rs } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/config", () => ({
  getBackendBaseURL: () => "",
}));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  loadNirEvaluationScenarios,
  runNirEvaluation,
} from "@/core/nir-evaluations/api";

const mockedFetch = rs.mocked(fetcher);

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  mockedFetch.mockReset();
});

describe("NIR evaluation API", () => {
  test("loads the versioned scenario catalog", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, { version: 1, scenarios: [{ id: "inspection" }] }),
    );

    await expect(loadNirEvaluationScenarios()).resolves.toMatchObject({
      version: 1,
      scenarios: [{ id: "inspection" }],
    });
    expect(mockedFetch).toHaveBeenCalledWith(
      "/api/nir/evaluations/scenarios",
      expect.objectContaining({ signal: undefined }),
    );
  });

  test("posts only thread and scenario identifiers for a batch run", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, {
        version: 1,
        summary: { total: 1, results: [] },
        traces: [],
      }),
    );

    await runNirEvaluation([
      { thread_id: "thread-1", scenario_id: "inspection-complete" },
    ]);

    expect(mockedFetch).toHaveBeenCalledWith(
      "/api/nir/evaluations/run",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          entries: [
            {
              thread_id: "thread-1",
              scenario_id: "inspection-complete",
            },
          ],
        }),
      }),
    );
  });

  test("surfaces the gateway detail when evaluation fails", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(404, { detail: "Thread has no NIR workflow trace" }),
    );

    await expect(
      runNirEvaluation([
        { thread_id: "thread-1", scenario_id: "inspection-complete" },
      ]),
    ).rejects.toThrow("Thread has no NIR workflow trace");
  });
});
