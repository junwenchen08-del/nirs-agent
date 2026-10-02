export interface NIRDataset {
  id: string;
  name: string;
  original_filename: string | null;
  sha256: string | null;
  size_bytes: number;
  media_type: string | null;
  status: string;
  created_at: string | null;
}

export interface NIRDatasetProfile {
  id: string;
  dataset_id: string;
  profile_version: number;
  profile_status: string;
  task_type: string | null;
  schema_status: string;
  mapping: Record<string, unknown>;
  mapping_sha256: string | null;
  confirmed_at: string | null;
  created_at: string | null;
}

export interface NIRDatasetDetail extends NIRDataset {
  profiles: NIRDatasetProfile[];
}

export interface NIRDatasetUse {
  attachment_id: string;
  dataset_id: string;
  profile_id: string | null;
  thread_id: string;
  run_id: string | null;
  workflow_project_id: string | null;
  source_sha256: string;
  virtual_path: string | null;
  status: string;
  reused_existing: boolean;
}

export interface NIRModelVersion {
  id: string;
  model_id: string;
  version: string;
  status: string;
  method: string | null;
  preprocessing: { steps?: Array<Record<string, unknown>> };
  validation_scope: string;
  artifact_size_bytes: number;
  artifact_sha256: string;
  metrics_sha256: string;
  training_data_sha256: string;
  metrics_summary: Record<string, unknown>;
  source_dataset_id: string | null;
  source_profile_id: string | null;
  source_thread_id: string;
  source_run_id: string | null;
  source_attempt: number;
  created_at: string | null;
  updated_at: string | null;
  last_used_at: string | null;
  reused_existing: boolean;
}

export interface NIRStorageUsage {
  dataset_bytes: number;
  model_bytes: number;
  thread_bytes: number;
  accounted_total_bytes: number;
  max_user_dataset_bytes: number;
  max_user_model_bytes: number;
  library_write_admission_bytes: number;
  strict_total_quota: boolean;
  disk_free_bytes: number;
  min_free_disk_bytes: number;
}

export interface NIRDeleteResult {
  status: "deleted";
  already_deleted: boolean;
  reclaimed_bytes: number;
  attached_thread_copies_retained: boolean;
}
