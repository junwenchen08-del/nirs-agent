import { beforeEach, describe, expect, test, rs } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/config", () => ({
  getBackendBaseURL: () => "",
}));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  archiveDataset,
  attachModel,
  deleteDataset,
  fetchNirLibraryEnabled,
  listDatasets,
  listModels,
} from "@/core/nir-library/api";

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

describe("NIR library API", () => {
  test("reads the feature flag", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, {
        agents_api: { enabled: true },
        nir_library: { enabled: true },
      }),
    );
    await expect(fetchNirLibraryEnabled()).resolves.toBe(true);
  });

  test("lists bounded dataset and model records", async () => {
    mockedFetch
      .mockResolvedValueOnce(
        jsonResponse(200, { datasets: [{ id: "ds_1" }], count: 1 }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, { models: [{ id: "nmv_1" }], count: 1 }),
      );
    await expect(listDatasets()).resolves.toHaveLength(1);
    await expect(listModels()).resolves.toHaveLength(1);
    expect(mockedFetch).toHaveBeenNthCalledWith(
      2,
      "/api/nir/models?include_archived=true",
      { signal: undefined },
    );
  });

  test("uses CSRF-aware state-changing requests", async () => {
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { status: "archived" }))
      .mockResolvedValueOnce(jsonResponse(200, { status: "deleted" }))
      .mockResolvedValueOnce(jsonResponse(200, { status: "attached" }));
    await archiveDataset("ds_1");
    await deleteDataset("ds_1", "ds_1");
    await attachModel("tablet", "v1", "thread-2");
    expect(mockedFetch).toHaveBeenNthCalledWith(
      1,
      "/api/nir/datasets/ds_1/archive",
      expect.objectContaining({ method: "POST" }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      2,
      "/api/nir/datasets/ds_1",
      expect.objectContaining({
        method: "DELETE",
        body: JSON.stringify({ confirmation: "ds_1" }),
      }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      3,
      "/api/nir/models/tablet/versions/v1/attach",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ thread_id: "thread-2" }),
      }),
    );
  });

  test("surfaces the stable server error message", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(409, {
        detail: {
          code: "referenced_by_models",
          message: "Delete referencing models first",
        },
      }),
    );
    await expect(deleteDataset("ds_1", "ds_1")).rejects.toThrow(
      "Delete referencing models first",
    );
  });
});
