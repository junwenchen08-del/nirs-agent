import { expect, test } from "@rstest/core";

import {
  mountedDatasetFiles,
  type MountedDataset,
} from "@/core/uploads/mounted-dataset";

const mounted: MountedDataset = {
  threadId: "thread1",
  datasetId: "ds_a",
  name: "光谱",
  virtualPath: "/mnt/user-data/uploads/dataset-ds_a.mat",
  sizeBytes: 16,
};

test("uses the mounted server file as message metadata without creating a browser upload", () => {
  expect(mountedDatasetFiles(mounted, "thread1")).toEqual([
    {
      filename: "dataset-ds_a.mat",
      size: 16,
      path: mounted.virtualPath,
      status: "uploaded",
    },
  ]);
});

test("does not carry a selected dataset into another thread", () => {
  expect(mountedDatasetFiles(mounted, "thread2")).toEqual([]);
  expect(mountedDatasetFiles(null, "thread1")).toEqual([]);
});

test("rejects malformed attachment paths or metadata", () => {
  for (const virtualPath of [
    "/mnt/user-data/outputs/data.mat",
    "/mnt/user-data/uploads/sub/data.mat",
    "/mnt/user-data/uploads/../data.mat",
    "C:\\data.mat",
  ]) {
    expect(mountedDatasetFiles({ ...mounted, virtualPath }, "thread1")).toEqual(
      [],
    );
  }
  expect(
    mountedDatasetFiles({ ...mounted, sizeBytes: NaN }, "thread1"),
  ).toEqual([]);
});
