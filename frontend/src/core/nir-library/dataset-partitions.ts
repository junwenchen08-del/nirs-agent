import type { DatasetPartition } from "./types";

export function datasetVersionPath(
  datasetId: string,
  profileId?: string | null,
) {
  return `/workspace/nir/datasets/${encodeURIComponent(datasetId)}${profileId ? `?profile=${encodeURIComponent(profileId)}` : ""}`;
}

export function datasetChatPath(datasetId: string, profileId?: string | null) {
  return `/workspace/chats/new?dataset=${encodeURIComponent(datasetId)}${profileId ? `&datasetProfile=${encodeURIComponent(profileId)}` : ""}`;
}

export function validPartition(
  value: DatasetPartition,
  profileId: string | null,
) {
  return (
    value.profile_id === profileId &&
    Number.isSafeInteger(value.version) &&
    value.version > 0 &&
    ["random", "fixed"].includes(value.method) &&
    ["calibration", "validation", "test"].every((key) => {
      const count = value.counts?.[key as keyof DatasetPartition["counts"]];
      return Number.isSafeInteger(count) && count > 0;
    })
  );
}
