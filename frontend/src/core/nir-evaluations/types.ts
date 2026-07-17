export type NIREvaluationScenario = {
  id: string;
  description: string;
  task_type: string;
  user_message: string;
  expected_route: string | null;
  expected_final_stages: string[];
  required_workflow_actions: string[];
  required_tools: string[];
  expected_approval_status: string | null;
  retry_outcome: string | null;
  required_trace_fields: string[];
  tags: string[];
};

export type NIREvaluationScenarioCatalog = {
  version: 1;
  scenarios: NIREvaluationScenario[];
};

export type NIREvaluationEntry = {
  thread_id: string;
  scenario_id: string;
};

export type NIREvaluationCheck = {
  name: string;
  passed: boolean;
  value: number;
  weight: number;
  details: string;
};

export type NIREvaluationResult = {
  scenario_id: string;
  passed: boolean;
  score: number;
  policy_violation_count: number;
  duration_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  trace_id: string | null;
  checks: NIREvaluationCheck[];
};

export type NIREvaluationSummary = {
  generated_at: string;
  total: number;
  passed: number;
  pass_rate: number;
  average_score: number;
  policy_violation_count: number;
  total_duration_ms: number;
  total_input_tokens: number;
  total_output_tokens: number;
  results: NIREvaluationResult[];
};

export type NIRBatchEvaluationResponse = {
  version: 1;
  summary: NIREvaluationSummary;
  traces: Record<string, unknown>[];
};

export type NIREvaluationMapping = {
  id: string;
  threadId: string;
  scenarioId: string;
};

export type NIREvaluationHistoryEntry = {
  generatedAt: string;
  total: number;
  passed: number;
  passRate: number;
  averageScore: number;
  policyViolationCount: number;
};
