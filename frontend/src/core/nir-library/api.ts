import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL, getLangGraphBaseURL } from "@/core/config";

import { parseModelDetail } from "./model-report";
import type {
  DatasetParsing,
  DatasetParsingPreview,
  DatasetParsingSelection,
  DatasetWorkspace,
  DatasetWorkspaceRows,
  DatasetWorkspaceSpectra,
  NIRDataset,
  NIRDatasetDetail,
  NIRDatasetProfile,
  NIRDatasetPreview,
  NIRDatasetUse,
  NIRDeleteResult,
  NIRModelVersion,
  NIRModelDetail,
  NIRStorageUsage,
} from "./types";

async function errorMessage(response: Response, fallback: string) {
  try {
    const payload = (await response.json()) as {
      detail?: string | { message?: string };
    };
    if (typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail;
    }
    if (
      payload.detail &&
      typeof payload.detail === "object" &&
      typeof payload.detail.message === "string" &&
      payload.detail.message.trim()
    ) {
      return payload.detail.message;
    }
  } catch {
    // Use the stable local fallback for non-JSON gateway responses.
  }
  return fallback;
}

async function requestJSON<T>(
  path: string,
  init: RequestInit | undefined,
  fallback: string,
): Promise<T> {
  const response = await fetch(`${getBackendBaseURL()}${path}`, init);
  if (!response.ok) throw new Error(await errorMessage(response, fallback));
  return (await response.json()) as T;
}

function encoded(value: string) {
  return encodeURIComponent(value);
}

export async function fetchNirLibraryEnabled(): Promise<boolean> {
  const response = await requestJSON<{ nir_library?: { enabled?: boolean } }>(
    "/api/features",
    undefined,
    "Failed to load NIR library availability",
  );
  return response.nir_library?.enabled === true;
}

export async function listDatasets(
  signal?: AbortSignal,
): Promise<NIRDataset[]> {
  const response = await requestJSON<{ datasets: NIRDataset[] }>(
    "/api/nir/datasets",
    { signal },
    "Failed to load datasets",
  );
  return response.datasets;
}

export function getDataset(datasetId: string, signal?: AbortSignal) {
  return requestJSON<NIRDatasetDetail>(
    `/api/nir/datasets/${encoded(datasetId)}`,
    { signal },
    "Failed to load dataset details",
  );
}

export function getDatasetPreview(
  datasetId: string,
  signal?: AbortSignal,
  profileId?: string | null,
) {
  return requestJSON<NIRDatasetPreview>(
    `/api/nir/datasets/${encoded(datasetId)}/preview${profileId ? `?profile_id=${encoded(profileId)}` : ""}`,
    { signal },
    "暂时无法读取数据预览，请重试。",
  );
}

export function getDatasetParsing(datasetId: string, signal?: AbortSignal) {
  return requestJSON<DatasetParsing>(
    `/api/nir/datasets/${encoded(datasetId)}/parsing`,
    { signal },
    "无法识别文件结构，请重试。",
  );
}

export function previewDatasetParsing(
  schema: DatasetParsing,
  selection: DatasetParsingSelection,
  taskType: string,
  signal?: AbortSignal,
) {
  return requestJSON<DatasetParsingPreview>(
    `/api/nir/datasets/${encoded(schema.dataset_id)}/parsing/preview`,
    {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_sha256: schema.source_sha256,
        profile_id: schema.profile_id,
        selection,
        task_type: taskType,
      }),
    },
    "无法预览当前选择，请调整解析设置。",
  );
}

export function saveDatasetParsing(
  schema: DatasetParsing,
  selection: DatasetParsingSelection,
  taskType: string,
) {
  return requestJSON<NIRDatasetProfile>(
    `/api/nir/datasets/${encoded(schema.dataset_id)}/parsing`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_sha256: schema.source_sha256,
        profile_id: schema.profile_id,
        selection,
        task_type: taskType,
      }),
    },
    "保存解析设置失败，当前选择仍保留。",
  );
}

export function getDatasetWorkspace(
  datasetId: string,
  signal?: AbortSignal,
  profileId?: string | null,
) {
  return requestJSON<DatasetWorkspace>(
    `/api/nir/datasets/${encoded(datasetId)}/workspace${profileId ? `?profile_id=${encoded(profileId)}` : ""}`,
    { signal },
    "无法读取数据工作台，请重试。",
  );
}

export function getDatasetWorkspaceRows(
  datasetId: string,
  offset: number,
  query: string,
  signal?: AbortSignal,
  profileId?: string | null,
) {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: "50",
    query,
  });
  if (profileId) params.set("profile_id", profileId);
  return requestJSON<DatasetWorkspaceRows>(
    `/api/nir/datasets/${encoded(datasetId)}/workspace/rows?${params}`,
    { signal },
    "无法读取样本表格，请重试。",
  );
}

export function getDatasetWorkspaceSpectra(
  datasetId: string,
  indices: number[] | null,
  signal?: AbortSignal,
  profileId?: string | null,
) {
  const params = new URLSearchParams();
  if (indices !== null) params.set("indices", indices.join(","));
  if (profileId) params.set("profile_id", profileId);
  const query = params.size ? `?${params}` : "";
  return requestJSON<DatasetWorkspaceSpectra>(
    `/api/nir/datasets/${encoded(datasetId)}/workspace/spectra${query}`,
    { signal },
    "无法读取光谱图，请重试。",
  );
}

export function saveDatasetWorkspace(
  workspace: DatasetWorkspace,
  usages: number[],
  range: [number, number],
  signal?: AbortSignal,
) {
  return requestJSON<DatasetWorkspace>(
    `/api/nir/datasets/${encoded(workspace.dataset_id)}/workspace`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        source_sha256: workspace.source_sha256,
        profile_id: workspace.profile_id,
        usages,
        range,
      }),
    },
    "保存失败，请重试。当前修改仍保留。",
  );
}

export async function listDatasetUses(
  datasetId: string,
  signal?: AbortSignal,
): Promise<NIRDatasetUse[]> {
  const response = await requestJSON<{ uses: NIRDatasetUse[] }>(
    `/api/nir/datasets/${encoded(datasetId)}/uses`,
    { signal },
    "Failed to load dataset history",
  );
  return response.uses;
}

export function getStorageUsage(signal?: AbortSignal) {
  return requestJSON<NIRStorageUsage>(
    "/api/nir/storage/usage",
    { signal },
    "Failed to load storage usage",
  );
}

export function saveDataset(input: {
  threadId: string;
  virtualPath: string;
  name: string;
}) {
  return requestJSON<NIRDataset>(
    "/api/nir/datasets",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        thread_id: input.threadId,
        virtual_path: input.virtualPath,
        name: input.name,
        save_confirmed: true,
      }),
    },
    "Failed to save dataset",
  );
}

export function uploadDataset(file: File, name: string) {
  const body = new FormData();
  body.append("file", file);
  body.append("name", name);
  return requestJSON<NIRDataset & { reused_existing: boolean }>(
    "/api/nir/datasets/upload",
    { method: "POST", body },
    "上传失败，请重试。",
  );
}

export function renameDataset(datasetId: string, name: string) {
  return requestJSON<NIRDataset>(
    `/api/nir/datasets/${encoded(datasetId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    },
    "Failed to rename dataset",
  );
}

export function createDatasetProfile(
  datasetId: string,
  input: {
    taskType: string;
    schemaStatus: string;
    mapping: Record<string, unknown>;
  },
) {
  return requestJSON<NIRDatasetProfile>(
    `/api/nir/datasets/${encoded(datasetId)}/profiles`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        task_type: input.taskType,
        schema_status: input.schemaStatus,
        mapping: input.mapping,
      }),
    },
    "Failed to create dataset profile",
  );
}

export function confirmDatasetProfile(datasetId: string, profileId: string) {
  return requestJSON<NIRDatasetProfile>(
    `/api/nir/datasets/${encoded(datasetId)}/profiles/${encoded(profileId)}/confirm`,
    { method: "POST" },
    "Failed to confirm dataset profile",
  );
}

export function attachDataset(
  datasetId: string,
  input: {
    threadId: string;
    profileId?: string;
    desiredFilename?: string;
    signal?: AbortSignal;
  },
) {
  return requestJSON<{
    status: string;
    dataset_id: string;
    profile_id: string | null;
    virtual_path: string;
    attachment_size_bytes?: number;
    workspace_version?: number | null;
  }>(
    `/api/nir/datasets/${encoded(datasetId)}/attach`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: input.signal,
      body: JSON.stringify({
        thread_id: input.threadId,
        profile_id: input.profileId ?? null,
        desired_filename: input.desiredFilename ?? null,
        idempotency_key: `ui:${datasetId}:${input.threadId}:${input.profileId ?? "none"}`,
      }),
    },
    "Failed to attach dataset",
  );
}

async function prepareComposerThread(threadId: string, signal: AbortSignal) {
  const response = await fetch(
    `${getLangGraphBaseURL().replace(/\/+$/, "")}/threads`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({ thread_id: threadId, metadata: {} }),
    },
  );
  if (!response.ok)
    throw new Error(
      await errorMessage(
        response,
        "Failed to prepare this chat for attachment",
      ),
    );
  const thread = (await response.json()) as { thread_id: string };
  if (thread.thread_id !== threadId)
    throw new Error("Attachment chat does not match the current chat");
}

export async function attachDatasetForComposer(
  datasetId: string,
  threadId: string,
  signal: AbortSignal,
  profileId?: string,
) {
  await prepareComposerThread(threadId, signal);
  const detail = await getDataset(datasetId, signal);
  const latestProfile = (detail.profiles ?? [])
    .filter((profile) => profile.profile_status === "confirmed")
    .sort((a, b) => b.profile_version - a.profile_version)[0];
  const workspaceProfile =
    latestProfile &&
    (Boolean(latestProfile.mapping?.workspace) ||
      Boolean(latestProfile.mapping?.library_parse))
      ? latestProfile
      : undefined;
  if (
    profileId &&
    !detail.profiles.some(
      (profile) =>
        profile.id === profileId &&
        profile.dataset_id === datasetId &&
        ["split", "confirmed"].includes(profile.profile_status),
    )
  )
    throw new Error("所选划分版本已不可用，请回到数据集库重新选择。");
  const attached = await attachDataset(datasetId, {
    threadId,
    signal,
    profileId: profileId ?? workspaceProfile?.id,
  });
  if (
    profileId &&
    (attached.dataset_id !== datasetId || attached.profile_id !== profileId)
  )
    throw new Error("挂载的数据划分与所选版本不一致。");
  return attached;
}

export function archiveDataset(datasetId: string) {
  return requestJSON<NIRDataset>(
    `/api/nir/datasets/${encoded(datasetId)}/archive`,
    { method: "POST" },
    "Failed to archive dataset",
  );
}

export function deleteDataset(datasetId: string, confirmation: string) {
  return requestJSON<NIRDeleteResult>(
    `/api/nir/datasets/${encoded(datasetId)}`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation }),
    },
    "Failed to delete dataset",
  );
}

export async function listModels(
  signal?: AbortSignal,
): Promise<NIRModelVersion[]> {
  const response = await requestJSON<{ models: NIRModelVersion[] }>(
    "/api/nir/models?include_archived=true",
    { signal },
    "Failed to load models",
  );
  return response.models;
}

export function attachModel(
  modelId: string,
  version: string,
  threadId: string,
  signal?: AbortSignal,
) {
  return requestJSON<{
    status: string;
    model_id: string;
    version: string;
    model_path: string;
  }>(
    `/api/nir/models/${encoded(modelId)}/versions/${encoded(version)}/attach`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({ thread_id: threadId }),
    },
    "Failed to attach model",
  );
}

export function getModel(
  modelId: string,
  version: string,
  signal?: AbortSignal,
) {
  return requestJSON<NIRModelVersion>(
    `/api/nir/models/${encoded(modelId)}?version=${encoded(version)}`,
    { signal },
    "暂时无法读取模型，请重试。",
  );
}

export async function getModelReport(
  modelId: string,
  version: string,
  signal?: AbortSignal,
): Promise<NIRModelDetail> {
  const payload = await requestJSON<unknown>(
    `/api/nir/models/${encoded(modelId)}/versions/${encoded(version)}/report`,
    { signal },
    "暂时无法读取模型详情，请重试。",
  );
  return parseModelDetail(payload, modelId, version);
}

export async function attachModelForComposer(
  modelId: string,
  version: string,
  threadId: string,
  signal: AbortSignal,
) {
  await prepareComposerThread(threadId, signal);
  return attachModel(modelId, version, threadId, signal);
}

export function archiveModel(modelId: string, version: string) {
  return requestJSON<NIRModelVersion>(
    `/api/nir/models/${encoded(modelId)}/versions/${encoded(version)}/archive`,
    { method: "POST" },
    "Failed to archive model",
  );
}

export function deleteModel(
  modelId: string,
  version: string,
  confirmation: string,
) {
  return requestJSON<NIRDeleteResult>(
    `/api/nir/models/${encoded(modelId)}/versions/${encoded(version)}`,
    {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirmation }),
    },
    "Failed to delete model",
  );
}
