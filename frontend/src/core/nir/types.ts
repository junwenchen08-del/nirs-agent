export type NIRWorkflowStage =
  | "intake"
  | "data_audit"
  | "clarification"
  | "planning"
  | "execution"
  | "knowledge"
  | "evaluation"
  | "review"
  | "approved"
  | "registered"
  | "completed"
  | "blocked";

export interface NIRWorkflowState extends Record<string, unknown> {
  project_id: string;
  task_type: string;
  stage: NIRWorkflowStage | string;
  revision: number;
  attempt?: number;
  max_attempts?: number;
  validation_goal?: string | null;
  approval_status?: string | null;
  next_action?: string | null;
  dataset_id?: string | null;
  dataset_profile_id?: string | null;
  dataset_sha256?: string | null;
  attempt_evidence?: Record<string, unknown> | null;
  attempts?: Record<string, unknown>[];
}

export interface NIRMetricView {
  key: string;
  value: string;
}

export interface NIRWorkflowView {
  projectId: string;
  taskType: string;
  stage: string;
  revision: number;
  attempt: number;
  maxAttempts: number;
  validationGoal: string | null;
  validationScope: string | null;
  approvalStatus: string | null;
  nextAction: string | null;
  method: string | null;
  preprocessing: string[];
  metrics: NIRMetricView[];
  datasetId: string | null;
  datasetProfileId: string | null;
  datasetHashShort: string | null;
}
