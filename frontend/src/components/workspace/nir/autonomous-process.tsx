"use client";

import { useEffect, useState } from "react";

import { urlOfArtifact } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import {
  autonomousIntakeStatus,
  autonomousLifecycleFacts,
  autonomousTestStatusLabel,
  isAutonomousTerminal,
  parseRejectionPage,
  parseSummary,
  record,
  stopExplanation,
  text,
} from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { useProcessResource } from "@/core/nir/process-hooks";

import { Metrics } from "./autonomous-evidence";
import { FinalCharts, Trends } from "./autonomous-final-views";
import { AutonomousRoundExplorer } from "./autonomous-round-explorer";
import { CalSpectraComparison } from "./cal-spectra";
import { ProcessFacts } from "./process-facts";

const stageLabels: Record<string, [string, string]> = {
  searching: ["Cal/Val 搜索中", "Cal/Val search"],
  budget_exhausted: [
    "搜索预算耗尽，等待保留选择",
    "Budget exhausted, awaiting retention",
  ],
  stopped: ["停止选择已记录，等待冻结", "Choice recorded, awaiting freeze"],
  frozen: ["模型已冻结，等待独立 Test", "Frozen, awaiting independent Test"],
  testing: ["独立 Test 评价中", "Independent Test in progress"],
  finalized: ["独立 Test 已完成", "Independent Test completed"],
  final_test_failed: [
    "Test 评价失败，禁止自动重试",
    "Test failed, no automatic replay",
  ],
  provider_failed: ["模型服务失败", "Provider failure"],
  cancelled: ["人工取消", "Cancelled"],
  no_deliverable_model: ["没有可交付模型", "No deliverable model"],
  decision_invalid: ["提案修正次数耗尽", "Proposal repairs exhausted"],
  safety_closed: [
    "安全终止，查看已记录原因",
    "Safety stop; inspect recorded cause",
  ],
  finalization_unavailable: [
    "终结资源不可用",
    "Finalization resources unavailable",
  ],
  stopped_pending_confirmation: [
    "模型保留已记录，等待确认",
    "Retention recorded, awaiting confirmation",
  ],
};

const deliveryLabels: Record<string, [string, string]> = {
  queued: ["正在确认最终模型", "Confirming retained model"],
  freezing: ["正在冻结模型", "Freezing model"],
  frozen: ["模型已冻结", "Model frozen"],
  final_test_pending: ["正在执行最终评价", "Running final evaluation"],
  final_test_completed: ["最终评价已完成", "Final evaluation completed"],
  publishing: ["正在生成报告与下载文件", "Building report and downloads"],
  delivered: ["已完成交付，等待用户审批", "Delivered, awaiting approval"],
  failed: [
    "自动交付失败，保留已完成证据",
    "Delivery failed; completed evidence preserved",
  ],
  audit_required: [
    "安全关闭，待人工审计",
    "Closed safely; operator audit required",
  ],
  cancelled: [
    "交付已取消，保留已有证据",
    "Delivery cancelled; evidence preserved",
  ],
};

function Rejections({
  threadId,
  sid,
  zh,
  poll,
}: {
  threadId: string;
  sid: string;
  zh: boolean;
  poll: boolean;
}) {
  const [offset, setOffset] = useState(0);
  const query = useProcessResource<{
    data: unknown[];
    next_offset: number | null;
  }>(threadId, `autonomous/${sid}/rejections?offset=${offset}`, poll);
  const page = parseRejectionPage(query.data, offset);
  return (
    <section className="rounded border p-3 text-sm">
      <h4 className="font-medium">
        {zh ? "被拒绝的提案（未执行）" : "Rejected proposals (not executed)"}
      </h4>
      {query.isError ? (
        <p role="alert">
          {zh
            ? "提案校验记录读取失败，请关闭后重新读取。"
            : "Proposal validation records failed to load; reopen to retry."}
        </p>
      ) : query.isPending ? (
        <p>
          {zh
            ? "正在读取提案校验记录…"
            : "Reading proposal validation records…"}
        </p>
      ) : page.data.length ? (
        page.data.map((raw, index) => (
          <div key={index} className="mt-2 border-t pt-2">
            <ProcessFacts zh={zh} value={record(raw)} />
          </div>
        ))
      ) : (
        <p className="text-muted-foreground mt-2">
          {zh
            ? "未记录提案拒绝事件。旧运行不补造记录。"
            : "No recorded rejection events. Old runs are not reconstructed."}
        </p>
      )}
      <div className="mt-2 flex justify-between">
        <button
          type="button"
          disabled={!offset}
          onClick={() => setOffset(Math.max(0, offset - 10))}
        >
          {zh ? "上一页" : "Previous"}
        </button>
        <button
          type="button"
          disabled={page.next_offset == null}
          onClick={() => {
            if (page.next_offset !== null) setOffset(page.next_offset);
          }}
        >
          {zh ? "下一页" : "Next"}
        </button>
      </div>
    </section>
  );
}

export function AutonomousProcess({
  threadId,
  sessionId,
  workflow,
  compact = false,
}: {
  threadId: string;
  sessionId?: string;
  workflow?: unknown;
  compact?: boolean;
}) {
  const { locale } = useI18n(),
    zh = locale === "zh-CN";
  const sessions = useProcessResource<{ data: { session_id: string }[] }>(
    threadId,
    sessionId ? null : "autonomous",
  );
  const sid = sessionId ?? sessions.data?.data[0]?.session_id;
  const summaryQuery = useProcessResource<unknown>(
    threadId,
    sid ? `autonomous/${sid}` : null,
    true,
  );
  const summary = parseSummary(summaryQuery.data);
  const [showRejections, setShowRejections] = useState(false);
  useEffect(() => {
    setShowRejections(false);
  }, [threadId, sid]);
  const [tab, setTab] = useState<"timeline" | "charts" | "delivery">(
    "timeline",
  );
  if (summaryQuery.isError || sessions.isError)
    return (
      <div className="p-4" role="alert">
        <p>
          {zh
            ? "自主建模证据读取失败或没有访问权限。"
            : "Autonomous evidence unavailable or access denied."}
        </p>
        <button
          type="button"
          onClick={() => {
            void summaryQuery.refetch();
            void sessions.refetch();
          }}
        >
          {zh ? "重新读取" : "Retry read"}
        </button>
      </div>
    );
  if (!summary || !sid) {
    const intake = autonomousIntakeStatus(workflow, zh);
    return (
      <div className="p-4" role={intake.failed ? "alert" : "status"}>
        <p>{intake.message}</p>
        {intake.code && (
          <p className="text-muted-foreground mt-2 text-xs">{intake.code}</p>
        )}
      </div>
    );
  }
  const deliveryStage = text(record(summary.automatic_delivery).status);
  if (compact) {
    const recent = summary.trends.slice(-4).reverse();
    const latest = recent[0];
    return (
      <div
        className="space-y-5 overflow-y-auto p-4"
        data-testid="modeling-progress-summary"
      >
        <p role="status" className="font-medium">
          {deliveryLabels[deliveryStage]?.[zh ? 0 : 1] ??
            stageLabels[summary.stage]?.[zh ? 0 : 1] ??
            summary.stage}
        </p>
        {latest && (
          <>
            <p className="text-sm">
              {zh
                ? `最近完成：第 ${latest.round_index ?? "—"} 轮`
                : `Latest completed: round ${latest.round_index ?? "—"}`}{" "}
              · {latest.method.toUpperCase()}
            </p>
            <Metrics
              value={{
                metrics: { val: record(latest.metrics).val },
                stage: "searching",
              }}
            />
          </>
        )}
        <h3 className="text-sm font-medium">
          {zh ? "最近轮次" : "Recent rounds"}
        </h3>
        <div className="space-y-2">
          {recent.map((row) => (
            <a
              key={row.decision_id}
              href={`/workspace/nir/process/${encodeURIComponent(threadId)}?nir_session=${encodeURIComponent(summary.session_id)}&nir_decision=${encodeURIComponent(row.decision_id)}`}
              className="hover:bg-accent flex flex-wrap justify-between gap-2 rounded-lg border p-3 text-sm"
            >
              <span>
                {zh
                  ? `第 ${row.round_index ?? "—"} 轮`
                  : `Round ${row.round_index ?? "—"}`}{" "}
                · {row.method.toUpperCase()}
              </span>
              <span>Val RMSE {fmt(record(record(row.metrics).val).RMSE)}</span>
            </a>
          ))}
        </div>
        {record(record(workflow).user_question).status === "waiting" &&
          record(workflow).search_session_id === summary.session_id && (
            <div role="status" className="bg-accent rounded-lg p-3 text-sm">
              <strong>{zh ? "等待你的回复" : "Waiting for your reply"}</strong>
              <p className="mt-2 whitespace-pre-wrap">
                {text(record(record(workflow).user_question).question, 4000)}
              </p>
            </div>
          )}
        <p className="text-muted-foreground text-xs">
          {stopExplanation(summary, zh)}
        </p>
        <p className="text-muted-foreground text-xs">
          {zh
            ? "完整过程保留方法、参数、各范围图表和 LLM 下一步建议。"
            : "The full process includes methods, parameters, scoped plots and original LLM advice."}
        </p>
      </div>
    );
  }
  const limits = record(summary.budget.limits);
  return (
    <div
      className="mx-auto min-h-0 w-full max-w-[1600px] flex-1 space-y-4 overflow-y-auto p-4 lg:p-6"
      data-testid="autonomous-process"
    >
      <section className="rounded-xl border p-4">
        <h2 className="text-lg font-semibold">
          {zh ? "建模全过程" : "LLM autonomous modeling"}
        </h2>
        <p className="mt-1 text-sm" role="status">
          {deliveryLabels[deliveryStage]?.[zh ? 0 : 1] ??
            stageLabels[summary.stage]?.[zh ? 0 : 1] ??
            summary.stage}
        </p>
        <details className="mt-3 text-xs">
          <summary className="text-muted-foreground cursor-pointer">
            {zh ? "资源用量与执行状态" : "Usage and execution status"}
          </summary>
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            {[
              [
                zh ? "Cal/Val 搜索" : "Cal/Val search",
                `${fmt(summary.budget.experiments_used)}/${fmt(limits.max_experiments)}`,
              ],
              [
                zh ? "模型拟合" : "Model fits",
                `${fmt(summary.budget.fits_used)}/${fmt(limits.max_fits)}`,
              ],
              [
                zh ? "高层决策" : "Decisions",
                `${fmt(summary.budget.actions_used)}/${fmt(limits.max_decisions)}`,
              ],
              ["Test", autonomousTestStatusLabel(summary, zh)],
              [
                zh ? "模型调用" : "Model calls",
                fmt(
                  Number(record(record(summary.budget.llm).search).calls ?? 0) +
                    Number(
                      record(record(summary.budget.llm).finalization).calls ??
                        0,
                    ),
                ),
              ],
              [
                zh ? "冻结" : "Freeze",
                summary.freeze
                  ? zh
                    ? "已冻结"
                    : "Frozen"
                  : zh
                    ? "尚未冻结"
                    : "Pending",
              ],
              [
                zh ? "交付" : "Delivery",
                summary.export_status === "succeeded"
                  ? zh
                    ? "已发布"
                    : "Published"
                  : summary.export_status || "—",
              ],
            ].map(([label, value]) => (
              <span key={label} className="rounded border px-2 py-1">
                {label}: {value}
              </span>
            ))}
          </div>
        </details>
        <p className="mt-3 text-sm">{stopExplanation(summary, zh)}</p>
        {(summary.stop_source || summary.terminal_failure_source) && (
          <details className="mt-2 text-xs">
            <summary>
              {zh ? "停止与资源依据" : "Stop and resource evidence"}
            </summary>
            <ProcessFacts
              zh={zh}
              value={{
                stop_source: summary.stop_source,
                stop_reason: summary.stop_reason,
                retained_experiment_id: summary.retained_experiment_id,
                terminal_failure_source: summary.terminal_failure_source,
                terminal_failure_reason: summary.terminal_failure_reason,
                budget: summary.budget,
              }}
            />
          </details>
        )}
        <p className="text-muted-foreground mt-2 text-xs">
          {summary.spectra_experiments.length
            ? zh
              ? "Cal 原始与预处理光谱已记录，可在“验证与图表”中查看。"
              : "Raw and preprocessed Cal spectra are available in Validation and charts."
            : zh
              ? "本运行未记录 Cal 光谱曲线；页面不读取原始文件或 Test 来补画。"
              : "No Cal spectra recorded for this run. The page does not reopen raw data or Test."}
        </p>
      </section>
      <nav
        className="flex gap-2"
        aria-label={zh ? "自主建模视图" : "Autonomous views"}
      >
        {(["timeline", "charts", "delivery"] as const).map((key) => (
          <button
            type="button"
            key={key}
            className={`rounded-lg border px-3 py-2 text-sm ${tab === key ? "bg-accent font-semibold" : ""}`}
            aria-pressed={tab === key}
            onClick={() => setTab(key)}
          >
            {
              {
                timeline: ["建模过程", "Modeling process"],
                charts: ["验证与图表", "Validation and charts"],
                delivery: ["最终模型与下载", "Model and downloads"],
              }[key][zh ? 0 : 1]
            }
          </button>
        ))}
      </nav>
      {tab === "timeline" && (
        <section className="space-y-3">
          <details
            open={showRejections}
            onToggle={(event) => setShowRejections(event.currentTarget.open)}
          >
            <summary className="cursor-pointer text-sm">
              {zh
                ? "查看提案校验错误与修正"
                : "Proposal validation errors and corrections"}
            </summary>
            {showRejections && (
              <Rejections
                threadId={threadId}
                sid={sid}
                zh={zh}
                poll={!isAutonomousTerminal(summary.stage)}
              />
            )}
          </details>
          <AutonomousRoundExplorer
            key={`${threadId}:${sid}`}
            threadId={threadId}
            summary={summary}
            workflow={workflow}
            zh={zh}
          />
        </section>
      )}
      {tab === "charts" && (
        <>
          <Trends summary={summary} zh={zh} />
          <Metrics value={summary} />
          <CalSpectraComparison threadId={threadId} summary={summary} zh={zh} />
          <FinalCharts threadId={threadId} summary={summary} zh={zh} />
        </>
      )}
      {tab === "delivery" && (
        <section className="space-y-4 rounded-xl border p-4">
          <h3 className="font-medium">
            {zh ? "最终保留模型" : "Final retained model"}
          </h3>
          {!summary.freeze ? (
            <p>
              {zh
                ? "尚无冻结模型；成功实验并不自动成为最终模型。"
                : "No frozen model; success does not automatically select the final model."}
            </p>
          ) : (
            <ProcessFacts zh={zh} value={summary.freeze} />
          )}
          <Metrics value={summary} />
          <p className="text-sm">
            {zh
              ? "科学协议、性能门禁、用户审批与注册分别记录。"
              : "Scientific protocol, quality, user approval and registration are separate."}
          </p>
          <ProcessFacts
            zh={zh}
            value={autonomousLifecycleFacts(summary, workflow)}
          />
          {summary.downloads.length ? (
            <div className="flex flex-wrap gap-2">
              {summary.downloads.map((file) => (
                <a
                  key={file.path}
                  href={urlOfArtifact({
                    threadId,
                    filepath: `/${file.path}`,
                    download: true,
                  })}
                  className="rounded-lg border px-3 py-2 text-sm"
                  download
                >
                  {zh
                    ? ({
                        "model.joblib": "预测模型（model.joblib）",
                        "report.md": "中文图文报告（Markdown）",
                        "report.html": "离线图文报告（HTML）",
                        "delivery.zip": "完整下载包（ZIP）",
                        "metrics.json": "完整指标证据（JSON）",
                        "decisions.json": "原始决策证据（JSON）",
                      }[file.name] ?? file.name)
                    : file.name}
                </a>
              ))}
            </div>
          ) : (
            <p className="text-muted-foreground text-sm">
              {zh
                ? "尚无已发布的模型与报告文件。导出失败不自动重新运行 Test。"
                : "No published model/report files. Export failure never automatically reruns Test."}
            </p>
          )}
          <details>
            <summary className="cursor-pointer text-xs">
              {zh ? "任务与交付身份" : "Task and delivery identity"}
            </summary>
            <ProcessFacts
              zh={zh}
              value={{
                session_id: sid,
                task: summary.task,
                retained_experiment_id: summary.retained_experiment_id,
                best_val_experiment_id: summary.best_val_experiment_id,
              }}
            />
          </details>
        </section>
      )}
      <button
        type="button"
        className="rounded border px-3 py-2 text-sm"
        onClick={() => {
          void summaryQuery.refetch();
        }}
      >
        {zh ? "刷新进展（只读）" : "Refresh progress (read-only)"}
      </button>
    </div>
  );
}
