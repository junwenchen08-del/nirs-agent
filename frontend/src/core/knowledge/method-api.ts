import { fetch } from "@/core/api/fetcher";
import { getBackendBaseURL } from "@/core/config";

export type ParameterValue = string | number | boolean;
export type CandidateParams = Record<string, ParameterValue>;
export interface MethodParameter {
  type: "integer" | "number" | "string" | "boolean";
  description_zh?: string;
  default?: ParameterValue;
  minimum?: number;
  maximum?: number;
  odd?: boolean;
  choices?: ParameterValue[];
}
export interface MethodCardChanges {
  title?: string;
  summary_zh?: string;
  source_url?: string;
  source_section?: string;
  keywords?: string;
  problem_tags?: string[];
  avoid_tags?: string[];
  planning_notes_zh?: string;
  candidate_params?: CandidateParams[];
  priority?: number;
  review_status?: "draft" | "published" | "retired";
  applicability_zh?: string;
  limitations_zh?: string;
  parameter_guidance_zh?: string;
}
export interface MethodCard extends Required<MethodCardChanges> {
  method_id: string;
  method_kind: "preprocessing" | "modeling" | "mcp_reference";
  category?: string;
  mcp_capability_id?: string;
  mcp_available?: boolean;
  mcp_execution_supported?: boolean;
  mcp_provider_version?: string;
  mcp_kind?: string;
  mcp_operations?: string[];
  mcp_parameters?: {
    required?: string[];
    properties?: Record<
      string,
      {
        type?: string;
        "x-python-type"?: string;
        description?: string;
        default?: unknown;
        default_repr?: string;
        [key: string]: unknown;
      }
    >;
    [key: string]: unknown;
  };
  official_documentation?: boolean;
  source_documented_signature?: string;
  source_documented_parameters?: Record<string, string>;
  documentation_runtime_difference?: string;
  provider: "chemotools" | "native" | "scikit-learn";
  provider_class: string;
  runtime_provider_version: string;
  runtime_parameters: Record<string, MethodParameter>;
  auto_eligible: boolean;
  eligibility_reason: string | null;
  regular_axis_required: boolean;
  has_default: boolean;
  content_sha256: string;
  evidence_id: string;
  retrieved_on: string;
}
export interface MethodCatalog {
  can_edit: boolean;
  revision: number;
  index_version: string;
  cards: MethodCard[];
  count: number;
  problem_tags: Record<string, string>;
  category_labels?: Record<string, string>;
  coverage?: {
    official_document_count: number;
    official_covered_count: number;
    mcp_capability_count: number;
    mcp_covered_count: number;
    documentation_only_ids: string[];
    runtime_provider_version: string;
    retrieved_on: string;
  };
  available_methods: Array<
    Pick<
      MethodCard,
      | "method_id"
      | "title"
      | "provider_class"
      | "runtime_provider_version"
      | "runtime_parameters"
      | "regular_axis_required"
      | "method_kind"
      | "provider"
    >
  >;
  history: Array<{
    revision: number;
    method_id: string;
    action: string;
    actor: string;
    changed_at: string;
  }>;
}
export interface MethodSearchResult {
  results: MethodCard[];
  abstained: boolean;
  reason: string;
  index_version: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(
    `${getBackendBaseURL()}/api/method-knowledge${path}`,
    init,
  );
  if (!response.ok) {
    const data = (await response.json().catch(() => ({}))) as {
      detail?: unknown;
    };
    throw new Error(
      typeof data.detail === "string" ? data.detail : `HTTP ${response.status}`,
    );
  }
  return (await response.json()) as T;
}
export const listMethodCards = () => request<MethodCatalog>("/cards");
export const saveMethodCard = (
  methodId: string,
  changes: MethodCardChanges,
  revision: number,
) =>
  request<MethodCatalog>(`/cards/${encodeURIComponent(methodId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: revision, changes }),
  });
export const resetMethodCard = (methodId: string, revision: number) =>
  request<MethodCatalog>(`/cards/${encodeURIComponent(methodId)}/reset`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ expected_revision: revision }),
  });
export const searchMethodCards = (query: string) =>
  request<MethodSearchResult>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 6 }),
  });

export function setCandidateValue(
  params: CandidateParams,
  name: string,
  raw: string,
  schema: MethodParameter,
): CandidateParams {
  const next = { ...params };
  if (!raw.trim()) {
    delete next[name];
    return next;
  }
  let value: ParameterValue = raw;
  if (schema.type === "integer" || schema.type === "number") {
    value = Number(raw);
    if (
      !Number.isFinite(value) ||
      (schema.type === "integer" && !Number.isInteger(value))
    )
      throw new Error(`${name}: invalid number / 数值无效`);
  } else if (schema.type === "boolean") {
    if (raw !== "true" && raw !== "false")
      throw new Error(`${name}: invalid boolean`);
    value = raw === "true";
  }
  next[name] = value;
  return next;
}
