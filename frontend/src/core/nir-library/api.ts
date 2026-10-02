import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

import type {
  NIRDataset,
  NIRDatasetDetail,
  NIRDatasetProfile,
  NIRDatasetUse,
  NIRDeleteResult,
  NIRModelVersion,
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
  input: { threadId: string; profileId?: string; desiredFilename?: string },
) {
  return requestJSON<{ status: string; virtual_path: string }>(
    `/api/nir/datasets/${encoded(datasetId)}/attach`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
) {
  return requestJSON<{ status: string; model_path: string }>(
    `/api/nir/models/${encoded(modelId)}/versions/${encoded(version)}/attach`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId }),
    },
    "Failed to attach model",
  );
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
