/** Read-only lifecycle projection. A later stage never proves an earlier receipt. */
import {
  autonomousLifecycleFacts,
  record,
  text,
  type AutonomousSummary,
} from "./autonomous";

export type LifecycleKey =
  | "input"
  | "inspect"
  | "prepare"
  | "freeze"
  | "test"
  | "delivery"
  | "approval"
  | "registration";
export type LifecycleNode = {
  id: LifecycleKey;
  title: [string, string];
  stage: "data" | "final" | "delivery";
  status: string;
  facts: Record<string, unknown>;
  method: [string, string];
};

export function processLifecycle(
  summary: AutonomousSummary,
  workflow: unknown,
): LifecycleNode[] {
  const raw = record(workflow);
  const current = raw.search_session_id === summary.session_id ? raw : {};
  const rawIngestion = record(current.ingestion_evidence);
  const ingestion: Record<string, string | number> = {};
  for (const key of [
    "status",
    "source_tool",
    "code",
    "message",
    "source_sha256",
    "dataset_id",
    "profile_id",
    "sample_count",
    "feature_count",
  ]) {
    const value = rawIngestion[key];
    if (typeof value === "string" && text(value)) ingestion[key] = text(value);
    else if (typeof value === "number" && Number.isFinite(value))
      ingestion[key] = value;
  }
  const lifecycle = autonomousLifecycleFacts(summary, workflow);
  const approval = text(lifecycle.user_approval_status);
  const registered = text(lifecycle.registered_model_id);
  return [
    {
      id: "input",
      title: ["文件进入", "Data intake"],
      stage: "data",
      status: text(ingestion.status) || "not_recorded",
      method: ["读取已有输入记录", "Recorded data intake"],
      facts: ingestion,
    },
    {
      id: "inspect",
      title: ["数据解析与确认", "Data parsing and confirmation"],
      stage: "data",
      status: Object.keys(summary.data_context).length
        ? "recorded"
        : "not_recorded",
      method: ["数据结构与目标确认", "Structure and target confirmation"],
      facts: {
        task: summary.task,
        calibration_samples: summary.data_context.calibration_samples,
        n_features: summary.data_context.n_features,
        unit: summary.data_context.unit,
      },
    },
    {
      id: "prepare",
      title: ["分区与搜索准备", "Partitions and search preparation"],
      stage: "data",
      status: Object.keys(record(summary.data_context.partition_counts)).length
        ? "recorded"
        : "not_recorded",
      method: [
        "固定分区与训练内交叉验证",
        "Fixed partitions and calibration CV",
      ],
      facts: summary.data_context,
    },
    {
      id: "freeze",
      title: ["冻结模型与输入", "Freeze model and inputs"],
      stage: "final",
      status: summary.freeze ? "completed" : "pending",
      method: [
        "冻结明确保留的实验",
        "Freeze the explicitly retained experiment",
      ],
      facts: summary.freeze ?? {},
    },
    {
      id: "test",
      title: ["最终测试评价", "Final Test evaluation"],
      stage: "final",
      status:
        summary.stage === "finalized"
          ? "completed"
          : summary.stage === "final_test_failed"
            ? "failed"
            : summary.test_status === "consumed"
              ? "consumed"
              : "pending",
      method: ["独立测试集最终评价", "Final independent Test evaluation"],
      facts: { test_status: summary.test_status, quality: summary.quality },
    },
    {
      id: "delivery",
      title: ["报告与模型交付", "Reports and model delivery"],
      stage: "delivery",
      status: summary.export_status || "pending",
      method: [
        "生成并发布已有模型与报告",
        "Publish the existing model and reports",
      ],
      facts: {
        export_status: summary.export_status,
        automatic_delivery: summary.automatic_delivery,
      },
    },
    {
      id: "approval",
      title: ["保存授权", "Save approval"],
      stage: "delivery",
      status:
        approval === "not_recorded_in_session" ? "not_recorded" : approval,
      method: ["等待真实用户授权", "Genuine user approval"],
      facts: {
        user_approval_status: lifecycle.user_approval_status,
        lifecycle_at_delivery: lifecycle.lifecycle_at_delivery,
      },
    },
    {
      id: "registration",
      title: ["模型入库", "Model registration"],
      stage: "delivery",
      status: registered ? "completed" : "not_recorded",
      method: [
        "登记已授权的精确模型产物",
        "Register the approved exact model artifact",
      ],
      facts: {
        registered_model_id: registered || null,
        lifecycle_at_delivery: lifecycle.lifecycle_at_delivery,
      },
    },
  ];
}

export function lifecycleStatus(status: string, zh: boolean) {
  return (
    {
      recorded: ["已记录", "Recorded"],
      completed: ["已完成", "Completed"],
      succeeded: ["已完成", "Completed"],
      failed: ["失败", "Failed"],
      pending: ["尚未完成", "Pending"],
      not_recorded: ["记录未提供", "Not recorded"],
      waiting: ["等待回复", "Waiting"],
      pending_user: ["等待授权", "Awaiting approval"],
      approved: ["已授权", "Approved"],
      rejected: ["未授权", "Denied"],
      consumed: ["已执行，结果待确认", "Executed; result unconfirmed"],
    }[status]?.[zh ? 0 : 1] ?? status
  );
}
