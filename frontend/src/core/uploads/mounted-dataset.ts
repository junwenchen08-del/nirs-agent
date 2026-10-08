import type { PromptInputMessage } from "@/components/ai-elements/prompt-input";
import type { FileInMessage } from "@/core/messages/utils";
import type { MountedModel } from "@/core/nir-library/mounted-model";

export type MountedDataset = {
  threadId: string;
  datasetId: string;
  name: string;
  virtualPath: string;
  sizeBytes: number;
};

export type ThreadInputMessage = PromptInputMessage & {
  mountedDataset?: MountedDataset;
  mountedModel?: MountedModel;
};

export function mountedDatasetFiles(
  dataset: MountedDataset | null | undefined,
  threadId: string,
): FileInMessage[] {
  if (dataset?.threadId !== threadId) return [];
  const match = /^\/mnt\/user-data\/uploads\/([^/\\]+)$/.exec(
    dataset.virtualPath,
  );
  const filename = match?.[1];
  if (
    !filename ||
    filename.length > 256 ||
    !/\.(csv|txt|mat)$/i.test(filename) ||
    !Number.isSafeInteger(dataset.sizeBytes) ||
    dataset.sizeBytes < 0
  )
    return [];
  return [
    {
      filename,
      size: dataset.sizeBytes,
      path: dataset.virtualPath,
      status: "uploaded",
    },
  ];
}
