import { expect, test } from "@rstest/core";

import {
  groupModels,
  modelLibraryPath,
  modelTask,
} from "@/core/nir-library/model-catalog";
import type { NIRModelVersion } from "@/core/nir-library/types";

function record(
  id: string,
  dataset: string | null,
  date: string | null,
  profile = "profile",
) {
  return {
    id,
    source_dataset_id: dataset,
    source_profile_id: profile,
    created_at: date,
  } as NIRModelVersion;
}

test("grouping preserves dataset identity, profile and unknown lineage instead of ranking test scores", () => {
  const original = [
    record("older", "one", "2026-10-01"),
    record("newer", "one", "2026-10-03", "different-profile"),
    record("other", "two", "2026-10-02"),
    record("unknown", null, "2026-10-05"),
    record("invalid-date", "two", "not-a-date"),
  ];
  const groups = groupModels(original);
  expect(groups.map((group) => group.datasetId)).toEqual(["one", "two", null]);
  expect(groups[0]!.models.map((model) => model.id)).toEqual([
    "newer",
    "older",
  ]);
  expect(groups[0]!.models[0]!.source_profile_id).toBe("different-profile");
  expect(original[0]!.id).toBe("older");
  expect(groups[1]!.models.map((model) => model.id)).toEqual([
    "other",
    "invalid-date",
  ]);
});

test("model task uses known model families and dataset filter links encode the exact identity", () => {
  expect(modelTask({ method: "pls" } as NIRModelVersion)).toBe("regression");
  expect(modelTask({ method: "rbf_svm" } as NIRModelVersion)).toBe(
    "classification",
  );
  expect(modelTask({ method: "new_method" } as NIRModelVersion)).toBe(
    "unknown",
  );
  expect(modelLibraryPath("数据 / 一&二")).toBe(
    "/workspace/nir/models?dataset=" + encodeURIComponent("数据 / 一&二"),
  );
});
