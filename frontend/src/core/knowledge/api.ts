import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

export interface KnowledgeBatchUploadResult {
  filename: string;
  success: boolean;
  chunks_added?: number;
  doc_id?: string;
  error?: string;
}

export type KnowledgeReviewStatus =
  | "draft"
  | "needs_review"
  | "published"
  | "retired";

export type KnowledgeQualityTier = "A" | "B" | "C" | "D" | "E";

export interface KnowledgeStatusResult {
  doc_id: string;
  review_status: KnowledgeReviewStatus;
}

export interface KnowledgeDocumentMetadata {
  title: string;
  authors: string[];
  year: number | null;
  doi: string | null;
  language: string;
  domains: string[];
  quality_tier: KnowledgeQualityTier;
}

export interface KnowledgeMetadataResult extends KnowledgeDocumentMetadata {
  doc_id: string;
  review_status: KnowledgeReviewStatus;
}

async function readErrorDetail(response: Response): Promise<string> {
  const data = (await response.json().catch(() => ({}))) as { detail?: string };
  return data.detail ?? `HTTP ${response.status}: ${response.statusText}`;
}

export async function uploadKnowledgeDocumentsSequentially(
  files: File[],
  title: string,
  year: string,
): Promise<KnowledgeBatchUploadResult[]> {
  const results: KnowledgeBatchUploadResult[] = [];

  // BGE-M3 CPU ingestion can take several minutes per PDF. Give every file its
  // own HTTP request so a multi-file selection does not share one proxy timeout.
  for (const file of files) {
    const form = new FormData();
    form.append("files", file, file.name);
    if (title) form.append("title", title);
    if (year) form.append("year", year);

    try {
      const response = await fetch(
        `${getBackendBaseURL()}/api/knowledge/documents/batch`,
        { method: "POST", body: form },
      );
      if (!response.ok) {
        results.push({
          filename: file.name,
          success: false,
          error: await readErrorDetail(response),
        });
        continue;
      }

      const payload = (await response.json()) as {
        results: KnowledgeBatchUploadResult[];
      };
      results.push(...payload.results);
    } catch (error) {
      results.push({
        filename: file.name,
        success: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  return results;
}

export async function setKnowledgeDocumentStatus(
  docId: string,
  reviewStatus: KnowledgeReviewStatus,
): Promise<KnowledgeStatusResult> {
  const encoded = encodeURIComponent(docId);
  const response = await fetch(
    `${getBackendBaseURL()}/api/knowledge/documents/${encoded}/status`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_status: reviewStatus }),
    },
  );
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as KnowledgeStatusResult;
}

export async function updateKnowledgeDocumentMetadata(
  docId: string,
  metadata: KnowledgeDocumentMetadata,
): Promise<KnowledgeMetadataResult> {
  const encoded = encodeURIComponent(docId);
  const response = await fetch(
    `${getBackendBaseURL()}/api/knowledge/documents/${encoded}/metadata`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(metadata),
    },
  );
  if (!response.ok) {
    throw new Error(await readErrorDetail(response));
  }
  return (await response.json()) as KnowledgeMetadataResult;
}
