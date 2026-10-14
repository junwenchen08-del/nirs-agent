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
  uploadDataset,
  getDatasetParsing,
  previewDatasetParsing,
  saveDatasetParsing,
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
  test("visual parsing separates read-only preview from confirmed save", async () => {
    const schema = {
      dataset_id: "ds_1",
      source_sha256: "a".repeat(64),
      profile_id: "dsp_previous",
      format: "mat" as const,
      task_type: "inspection",
      current: null,
      suggested: null,
      layouts: [],
      columns: [],
    };
    const selection = {
      layout_id: "layout-b",
      target_id: "target-y",
      axis_id: "axis",
    };
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, schema))
      .mockResolvedValueOnce(jsonResponse(200, {}))
      .mockResolvedValueOnce(jsonResponse(200, { id: "dsp_new" }));
    const loaded = await getDatasetParsing("ds_1");
    await previewDatasetParsing(loaded, selection, "calibration");
    await saveDatasetParsing(loaded, selection, "calibration");
    expect(mockedFetch).toHaveBeenNthCalledWith(
      2,
      "/api/nir/datasets/ds_1/parsing/preview",
      expect.objectContaining({ method: "POST" }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      3,
      "/api/nir/datasets/ds_1/parsing",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({
          source_sha256: schema.source_sha256,
          profile_id: "dsp_previous",
          selection,
          task_type: "calibration",
        }),
      }),
    );
  });

  test("the current parsed group takes precedence over an older workspace attachment", async () => {
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { thread_id: "thread-1" }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          profiles: [
            {
              id: "old",
              profile_version: 1,
              profile_status: "confirmed",
              mapping: { workspace: {} },
            },
            {
              id: "parsed",
              profile_version: 2,
              profile_status: "confirmed",
              mapping: { library_parse: {} },
            },
          ],
        }),
      )
      .mockResolvedValueOnce(jsonResponse(200, {}));
    await attachDatasetForComposer(
      "ds_1",
      "thread-1",
      new AbortController().signal,
    );
    const body = JSON.parse(mockedFetch.mock.calls[2]![1]!.body as string);
    expect(body.profile_id).toBe("parsed");
  });
  test("uploads the local file with authenticated multipart fetch and no thread", async () => {
    const file = new File(["1000,1100\n1,2\n"], "玉米.csv", {
      type: "text/csv",
    });
    mockedFetch.mockResolvedValueOnce(jsonResponse(200, { id: "ds_upload" }));
    await expect(uploadDataset(file, "玉米光谱")).resolves.toEqual({
      id: "ds_upload",
    });
    expect(mockedFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockedFetch.mock.calls[0]!;
    expect(url).toBe("/api/nir/datasets/upload");
    expect(options?.method).toBe("POST");
    expect(options?.headers).toBeUndefined();
    const body = options?.body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.get("name")).toBe("玉米光谱");
    expect(body.has("thread_id")).toBe(false);
  });

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
      .mockResolvedValueOnce(jsonResponse(200, { profiles: [] }))
      .mockResolvedValueOnce(jsonResponse(200, mounted));
    await expect(
      attachDatasetForComposer("ds_1", "thread-1", signal),
    ).resolves.toEqual(mounted);
    expect(mockedFetch).toHaveBeenCalledTimes(3);
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
      "/api/nir/datasets/ds_1",
      expect.objectContaining({ signal }),
    );
    expect(mockedFetch).toHaveBeenNthCalledWith(
      3,
      "/api/nir/datasets/ds_1/attach",
      expect.objectContaining({ method: "POST", signal }),
    );
  });

  test("pins the latest saved workspace profile so later mounts cannot reuse an old version", async () => {
    const signal = new AbortController().signal;
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { thread_id: "thread-1" }))
      .mockResolvedValueOnce(
        jsonResponse(200, {
          profiles: [
            {
              id: "old",
              profile_version: 1,
              profile_status: "superseded",
              mapping: { workspace: {} },
            },
            {
              id: "draft",
              profile_version: 9,
              profile_status: "draft",
              mapping: { workspace: {} },
            },
            {
              id: "saved-v3",
              profile_version: 3,
              profile_status: "confirmed",
              mapping: { workspace: {} },
            },
          ],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse(200, {
          status: "attached",
          virtual_path: "/mnt/user-data/uploads/selected.csv",
          attachment_size_bytes: 1234,
        }),
      );
    const result = await attachDatasetForComposer("ds_1", "thread-1", signal);
    expect(result.attachment_size_bytes).toBe(1234);
    const body = JSON.parse(mockedFetch.mock.calls[2]![1]!.body as string);
    expect(body.profile_id).toBe("saved-v3");
    expect(body.idempotency_key).toBe("ui:ds_1:thread-1:saved-v3");
    expect(mockedFetch.mock.calls[2]![1]!.signal).toBe(signal);
  });

  test("metadata read failure never attaches an unchecked workspace version", async () => {
    mockedFetch
      .mockResolvedValueOnce(jsonResponse(200, { thread_id: "thread-1" }))
      .mockResolvedValueOnce(
        jsonResponse(404, { detail: "Dataset unavailable" }),
      );
    await expect(
      attachDatasetForComposer(
        "ds_1",
        "thread-1",
        new AbortController().signal,
      ),
    ).rejects.toThrow("Dataset unavailable");
    expect(mockedFetch).toHaveBeenCalledTimes(2);
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
