import type { UploadedFileInfo } from "./api";

export function summarizeDatasetSaves(files: UploadedFileInfo[]) {
  const saved = new Set<string>();
  const reused = new Set<string>();
  const failures: { filename: string; errorCode: string }[] = [];
  for (const file of files) {
    const result = file.dataset_library;
    if (!result) continue;
    if (result.status === "failed") {
      failures.push({
        filename: file.filename,
        errorCode: result.error_code ?? "storage_error",
      });
    } else if (result.dataset_id) {
      if (result.status === "saved") saved.add(result.dataset_id);
      else if (result.status === "reused") reused.add(result.dataset_id);
    }
  }
  for (const id of saved) reused.delete(id);
  return { saved: saved.size, reused: reused.size, failures };
}
