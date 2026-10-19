import { describe, expect, it } from "@rstest/core";

import type { UploadedFileInfo } from "@/core/uploads/api";
import { summarizeDatasetSaves } from "@/core/uploads/dataset-library";

function file(result?: UploadedFileInfo["dataset_library"]): UploadedFileInfo {
  return {
    filename: "spectrum.csv",
    size: 12,
    path: "",
    virtual_path: "",
    artifact_url: "",
    dataset_library: result,
  };
}

describe("dataset upload save feedback", () => {
  it("counts unique saved and reused assets without double-counting a duplicate in one batch", () => {
    expect(
      summarizeDatasetSaves([
        file({ status: "saved", dataset_id: "ds_a" }),
        file({ status: "reused", dataset_id: "ds_a" }),
        file({ status: "reused", dataset_id: "ds_b" }),
        file({ status: "reused", dataset_id: "ds_b" }),
      ]),
    ).toEqual({ saved: 1, reused: 1, failures: [] });
  });
  it("does not infer a save from an ordinary upload or missing asset ID", () => {
    expect(
      summarizeDatasetSaves([
        file(),
        file({ status: "saved", dataset_id: null }),
      ]),
    ).toEqual({ saved: 0, reused: 0, failures: [] });
  });
  it("keeps failed saves explicit while allowing other successful files", () => {
    expect(
      summarizeDatasetSaves([
        file({ status: "failed", error_code: "quota_exceeded" }),
        file({ status: "saved", dataset_id: "ds_a" }),
      ]),
    ).toEqual({
      saved: 1,
      reused: 0,
      failures: [{ filename: "spectrum.csv", errorCode: "quota_exceeded" }],
    });
  });
});
