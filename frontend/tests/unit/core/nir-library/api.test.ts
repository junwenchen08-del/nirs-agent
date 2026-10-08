import { beforeEach, describe, expect, test, rs } from "@rstest/core";

rs.mock("@/core/api/fetcher", () => ({
  fetch: rs.fn(),
}));

rs.mock("@/core/config", () => ({
  getBackendBaseURL: () => "",
  getLangGraphBaseURL: () => "/api/langgraph",
}));

import { fetch as fetcher } from "@/core/api/fetcher";
import {
  archiveDataset,
  attachDatasetForComposer,
  attachModelForComposer,
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
  test("prepares the owned chat before attaching an exact model version, with cancellation", async () => {
    const signal = new AbortController().signal;
    const result = {
      status: "attached",
      model_id: "corn",
      version: "v2",
      model_path: "/mnt/user-data/outputs/models/corn/v2/model.pkl",
    };
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { thread_id: "thread-1" }))
      .mockResolvedValueOnce(jsonResponse(200, result));
    await expect(
      attachModelForComposer("corn", "v2", "thread-1", signal),
    ).resolves.toEqual(result);
    expect(mockedFetch).toHaveBeenCalledTimes(2);
    expect(mockedFetch).toHaveBeenNthCalledWith(
      1,
      "/api/langgraph/threads",
      expect.objectContaining({
        signal,
        body: JSON.stringify({ thread_id: "thread-1", metadata: {} }),
      }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      2,
      "/api/nir/models/corn/versions/v2/attach",
      expect.objectContaining({
        method: "POST",
        signal,
        body: JSON.stringify({ thread_id: "thread-1" }),
      }),
    );
  });
  test("model mounting never proceeds when thread ownership preparation fails", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(403, { detail: "Access denied" }),
    );
    await expect(
      attachModelForComposer(
        "corn",
        "v2",
        "thread-1",
        new AbortController().signal,
      ),
    ).rejects.toThrow("Access denied");
    expect(mockedFetch).toHaveBeenCalledTimes(1);
  });
  test("prepares the owned thread then mounts without creating a run", async () => {
    const signal = new AbortController().signal;
    const mounted = {
      status: "attached",
      virtual_path: "/mnt/user-data/uploads/dataset-ds_1.mat",
    };
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { thread_id: "thread-1" }))
      .mockResolvedValueOnce(jsonResponse(200, mounted));
    await expect(
      attachDatasetForComposer("ds_1", "thread-1", signal),
    ).resolves.toEqual(mounted);
    expect(mockedFetch).toHaveBeenCalledTimes(2);
    expect(mockedFetch).toHaveBeenNthCalledWith(
      1,
      "/api/langgraph/threads",
      expect.objectContaining({
        method: "POST",
        signal,
        body: JSON.stringify({ thread_id: "thread-1", metadata: {} }),
      }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      2,
      "/api/nir/datasets/ds_1/attach",
      expect.objectContaining({ method: "POST", signal }),
    );
  });

  test("does not attach if preparing the thread fails", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(403, { detail: "Access denied" }),
    );
    await expect(
      attachDatasetForComposer(
        "ds_1",
        "thread-1",
        new AbortController().signal,
      ),
    ).rejects.toThrow("Access denied");
    expect(mockedFetch).toHaveBeenCalledTimes(1);
  });

  test("does not attach to a different thread returned by the server", async () => {
    mockedFetch.mockResolvedValueOnce(
      jsonResponse(200, { thread_id: "other-thread" }),
    );
    await expect(
      attachDatasetForComposer(
        "ds_1",
        "thread-1",
        new AbortController().signal,
      ),
    ).rejects.toThrow("does not match");
    expect(mockedFetch).toHaveBeenCalledTimes(1);
  });

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
