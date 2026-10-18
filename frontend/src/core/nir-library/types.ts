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
  partition?: DatasetPartition | null;
}

export interface DatasetPartition {
  profile_id: string;
  version: number;
  method: "random" | "fixed";
  counts: { calibration: number; validation: number; test: number };
  parent_profile_id: string | null;
  source_thread_id: string;
  session_id: string;
}

export interface NIRDatasetDetail extends NIRDataset {
  profiles: NIRDatasetProfile[];
}

export type DatasetAnalysisType = "regression" | "classification" | "unknown";

export interface DatasetParsingSelection {
  layout_id?: string | null;
  target_id?: string | null;
  axis_id?: string | null;
  spectral_start?: number | null;
  spectral_end?: number | null;
  target_columns?: number[];
  sample_column?: number | null;
  coordinate_column?: number | null;
  transpose?: boolean;
  has_header?: boolean | null;
}

export interface DatasetParsing {
  dataset_id: string;
  source_sha256: string;
  profile_id: string | null;
  task_type: string;
  format: "mat" | "table";
  current: DatasetParsingSelection | null;
  suggested: DatasetParsingSelection | null;
  columns: Array<{ index: number; name: string }>;
  layouts: Array<{
    id: string;
    group: string;
    label: string;
    sample_count: number;
    feature_count: number;
    targets: Array<{ id: string; label: string }>;
    axes: Array<{ id: string; label: string }>;
    default_target: string | null;
    default_axis: string | null;
  }>;
}

export interface DatasetParsingPreview extends NIRDatasetPreview {
  samples: Array<{ name: string; values: Array<string | number | null> }>;
}

export interface DatasetWorkspace {
  dataset_id: string;
  source_sha256: string;
  profile_id: string | null;
  version: number;
  saved: boolean;
  sample_count: number;
  feature_count: number;
  axis_kind: "axis" | "index";
  axis: number[];
  fields: string[];
  usages: number[];
  range: [number, number];
  selected_feature_count: number;
  read_only?: boolean;
  partition?: DatasetPartition | null;
}

export interface DatasetWorkspaceRows {
  dataset_id: string;
  source_sha256: string;
  profile_id: string | null;
  total: number;
  offset: number;
  rows: Array<{
    index: number;
    name: string;
    values: Array<string | number | null>;
  }>;
}

export interface DatasetWorkspaceSpectra {
  dataset_id: string;
  source_sha256: string;
  profile_id: string | null;
  axis: number[];
  total: number;
  point_count: number;
  curves: Array<{ index: number; name: string; values: Array<number | null> }>;
}

export interface NIRDatasetPreview {
  dataset_id: string;
  source_sha256: string;
  profile_id?: string | null;
  status: "ready" | "needs_mapping";
  message: string;
  analysis_type: DatasetAnalysisType;
  sample_count: number | null;
  feature_count: number | null;
  preview_sample_count: number;
  missing_ratio: number | null;
  axis_kind: "axis" | "index";
  spectrum: Array<{
    x: number;
    mean: number | null;
    low: number | null;
    high: number | null;
  }>;
  targets: Array<{
    name: string;
    kind: "regression" | "classification";
    bars: Array<{ label: string; count: number }>;
  }>;
  fields: string[];
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
  quality_status?: "below_threshold_user_approved" | null;
  report_summary?: Pick<
    NIRModelReport,
    "target" | "unit" | "sample_count" | "feature_count" | "metrics"
  > | null;
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

export interface ModelMetric {
  scope: "calibration" | "validation" | "final_test";
  R2: number | null;
  RMSE: number | null;
  MAE: number | null;
  n: number | null;
}

export interface ModelChart {
  type: "prediction" | "spectra";
  scope: string;
  x: number[];
  y?: number[];
  series?: Array<{ id: string; values: number[] }>;
  x_label?: string | null;
  x_unit?: string | null;
  y_unit?: string | null;
  n_total: number;
  n_returned: number;
  axis_total?: number;
}

export interface NIRModelReport {
  schema_version: 1;
  target: string | null;
  unit: string | null;
  method: string | null;
  sample_count: number | null;
  feature_count: number | null;
  model_feature_count: number | null;
  parameters: Record<string, string | number | boolean | null>;
  preprocessing: Array<{
    method: string;
    params: Record<string, string | number | boolean | null>;
  }> | null;
  metrics: ModelMetric[];
  charts: ModelChart[];
}

export interface NIRModelDetail {
  model: NIRModelVersion;
  report: NIRModelReport;
  dataset: {
    id: string;
    name: string;
    source_sha256: string;
    profile_id: string | null;
    partition?: DatasetPartition | null;
  } | null;
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
