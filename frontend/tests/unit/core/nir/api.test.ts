import { afterEach, expect, test, rs } from "@rstest/core";

import { loadNIRWorkflowState } from "@/core/nir/api";

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

test("loads only the NIR workflow from the owner-checked thread state", async () => {
  globalThis.fetch = rs.fn(
    async () =>
      new Response(
        JSON.stringify({
          values: {
            messages: [{ content: "private message" }],
            nir_workflow: {
              project_id: "project-one",
              task_type: "analysis",
              stage: "review",
              revision: 4,
            },
          },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
  ) as typeof fetch;

  await expect(loadNIRWorkflowState("thread one")).resolves.toMatchObject({
    project_id: "project-one",
    revision: 4,
  });
  expect(globalThis.fetch).toHaveBeenCalledWith(
    "/api/threads/thread%20one/state",
    expect.objectContaining({ credentials: "include" }),
  );
});

test("treats missing or inaccessible thread state as unavailable", async () => {
  globalThis.fetch = rs.fn(
    async () => new Response(null, { status: 404 }),
  ) as typeof fetch;
  await expect(loadNIRWorkflowState("missing")).resolves.toBeNull();

  globalThis.fetch = rs.fn(
    async () => new Response(null, { status: 403 }),
  ) as typeof fetch;
  await expect(loadNIRWorkflowState("forbidden")).resolves.toBeNull();
});
