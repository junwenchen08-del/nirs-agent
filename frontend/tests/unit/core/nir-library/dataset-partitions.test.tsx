import { expect, test } from "@rstest/core";
import { renderToStaticMarkup } from "react-dom/server";

import { DatasetPartitions } from "@/components/workspace/nir-library/dataset-partitions";
import {
  datasetChatPath,
  datasetVersionPath,
  validPartition,
} from "@/core/nir-library/dataset-partitions";
import type {
  DatasetPartition,
  NIRDatasetProfile,
} from "@/core/nir-library/types";

const partition: DatasetPartition = {
  profile_id: "split_1",
  version: 3,
  method: "random",
  counts: { calibration: 60, validation: 20, test: 20 },
  parent_profile_id: "parse_2",
  session_id: "search_1",
  source_thread_id: "thread_1",
};

test("dataset and chat paths carry the exact encoded partition version", () => {
  expect(datasetVersionPath("数据 / 1", "划分 & 3")).toBe(
    `/workspace/nir/datasets/${encodeURIComponent("数据 / 1")}?profile=${encodeURIComponent("划分 & 3")}`,
  );
  expect(datasetChatPath("ds1", "split_1")).toBe(
    "/workspace/chats/new?dataset=ds1&datasetProfile=split_1",
  );
  expect(datasetVersionPath("ds1")).toBe("/workspace/nir/datasets/ds1");
  expect(validPartition(partition, "other")).toBe(false);
  expect(
    validPartition(
      { ...partition, counts: { ...partition.counts, test: -1 } },
      "split_1",
    ),
  ).toBe(false);
});

test("saved partition cards show recorded membership counts and exact-version reuse links", () => {
  const profile = {
    id: "split_1",
    dataset_id: "ds1",
    profile_version: 3,
    profile_status: "split",
    partition,
  } as NIRDatasetProfile;
  const html = renderToStaticMarkup(
    <DatasetPartitions datasetId="ds1" profiles={[profile]} zh />,
  );
  expect(html).toContain("划分版本");
  expect(html).toContain("训练集");
  expect(html).toContain("60");
  expect(html).toContain("?profile=split_1");
  expect(html).toContain("datasetProfile=split_1");
  expect(html).toContain("复用此划分");
});
