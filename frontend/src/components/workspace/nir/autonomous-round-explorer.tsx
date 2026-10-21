"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode } from "react";

import {
  isAutonomousTerminal,
  parseDecision,
  parseCalSpectra,
  record,
  sameRecordedValue,
  stopExplanation,
  text,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import { useProcessResource } from "@/core/nir/process-hooks";
import {
  processLifecycle,
  lifecycleStatus,
  type LifecycleKey,
} from "@/core/nir/process-lifecycle";
import { useRoundHistory } from "@/core/nir/round-hooks";
import { roundConfiguration } from "@/core/nir/round-process";

import { AutonomousRoundDetail } from "./autonomous-round-detail";
import { LifecycleNodeDetail } from "./lifecycle-node-detail";
import { type NodeDetailTab } from "./node-detail-tabs";
import { ProcessRoundNavigation } from "./process-round-navigation";
import { RoundNodeCard } from "./round-node-card";
import { RoundOptimizationTrend } from "./round-optimization-trend";
import { RoundPipelineRow } from "./round-pipeline-row";

type Stage = "data" | "search" | "final" | "delivery";
const stages: Stage[] = ["data", "search", "final", "delivery"];

export function AutonomousRoundExplorer({
  threadId,
  summary,
  workflow,
  zh,
  finalContent,
  deliveryContent,
  searchNotes,
}: {
  threadId: string;
  summary: AutonomousSummary;
  workflow?: unknown;
  zh: boolean;
  finalContent?: ReactNode;
  deliveryContent?: ReactNode;
  searchNotes?: ReactNode;
}) {
  const terminal = isAutonomousTerminal(summary.stage);
  const history = useRoundHistory(threadId, summary.session_id, !terminal);
  const [selected, setSelected] = useState<DecisionView | null>(null);
  const [restoredId, setRestoredId] = useState("");
  const [stage, setStage] = useState<Stage>("search");
  const [lifecycleId, setLifecycleId] = useState<LifecycleKey | null>(null);
  const [detailTab, setDetailTab] = useState<NodeDetailTab>("evaluation");
  const [step, setStep] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(true);
  const canvasRef = useRef<HTMLDivElement>(null);
  const lifecycle = processLifecycle(summary, workflow);
  const revision = useRef(summary.revision);
  const { refetch } = history;
  useEffect(() => {
    if (revision.current !== summary.revision) {
      revision.current = summary.revision;
      void refetch();
    }
  }, [summary.revision, refetch]);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("nir_session") !== summary.session_id) return;
    const did = params.get("nir_decision") ?? "";
    if (/^[A-Za-z0-9_-]{1,128}$/.test(did)) {
      setRestoredId(did);
      setExpanded(true);
    }
    const savedStage = params.get("nir_stage");
    if (stages.includes(savedStage as Stage)) setStage(savedStage as Stage);
    const node = lifecycle.find((item) => item.id === params.get("nir_node"));
    if (node) {
      setLifecycleId(node.id);
      setStage(node.stage);
    }
    const savedStep = params.get("nir_step") ?? "";
    if (
      /^(raw|preprocess-\d{1,2}|preprocessing|model|validation|decision)$/.test(
        savedStep,
      )
    )
      setStep(savedStep);
    const tab = params.get("nir_tab");
    if (["overview", "config", "evaluation", "reason"].includes(tab ?? "")) {
      setDetailTab(tab as NodeDetailTab);
      if (!savedStep && !node)
        setStep(
          tab === "reason"
            ? "decision"
            : tab === "config"
              ? "model"
              : "validation",
        );
    }
    // Lifecycle membership is stable for a session; later revisions never reset selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary.session_id]);
  useEffect(() => {
    if (
      history.hasNextPage &&
      !history.isFetching &&
      !history.isError &&
      history.rows.length < 100
    )
      void history.fetchNextPage();
  }, [history]);
  const rows = history.rows;
  const retained = rows.find(
    (row) => row.experiment_id === summary.retained_experiment_id,
  );
  const retainedRef = summary.trends.find(
    (row) => row.experiment_id === summary.retained_experiment_id,
  );
  const fallback =
    retained ??
    (retainedRef
      ? parseDecision({
          ...retainedRef,
          status: "succeeded",
          action: "EXPLORE",
        })
      : null) ??
    rows.at(-1);
  const choice =
    rows.find(
      (row) => row.decision_id === (selected?.decision_id ?? restoredId),
    ) ??
    selected ??
    (restoredId ? parseDecision({ decision_id: restoredId }) : null) ??
    fallback;
  const spectraQuery = useProcessResource<unknown>(
    threadId,
    expanded &&
      stage === "search" &&
      choice?.status === "succeeded" &&
      choice.experiment_id &&
      !["STOP", "RETAIN"].includes(choice.action)
      ? `autonomous/${summary.session_id}/spectra/${choice.experiment_id}`
      : null,
  );
  const curves = choice?.experiment_id
    ? parseCalSpectra(
        spectraQuery.data,
        summary.session_id,
        choice.experiment_id,
      )
    : null;
  const config = choice ? roundConfiguration(choice) : null;
  const spectra =
    curves &&
    curves.decision_id === choice?.decision_id &&
    config?.actual &&
    curves.configuration.method === config.method &&
    sameRecordedValue(curves.configuration.steps, config.steps)
      ? curves
      : null;
  const persist = (values: Record<string, string | null>) => {
    const params = new URLSearchParams(window.location.search);
    params.set("nir_session", summary.session_id);
    params.set("nir_view", "pipeline");
    for (const [key, value] of Object.entries(values)) {
      if (value === null) params.delete(key);
      else params.set(key, value);
    }
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}?${params}`,
    );
  };
  const onSelect = (decision: DecisionView, focus?: string) => {
    setStage("search");
    setLifecycleId(null);
    setSelected(decision);
    setRestoredId("");
    setExpanded(true);
    const stopping = ["STOP", "RETAIN"].includes(decision.action);
    const focusedStep = focus ?? (stopping ? "decision" : "validation");
    const tab =
      focusedStep === "decision"
        ? "reason"
        : ["raw", "model", "preprocessing"].includes(focusedStep) ||
            focusedStep.startsWith("preprocess-")
          ? "config"
          : "evaluation";
    setDetailTab(tab);
    setStep(focusedStep);
    persist({
      nir_stage: "search",
      nir_decision: decision.decision_id,
      nir_node: null,
      nir_step: focusedStep,
      nir_tab: tab,
    });
    requestAnimationFrame(() => {
      const selector = focus
        ? `[data-testid="round-detail"]`
        : `[data-flow-id="${decision.decision_id}"]`;
      (
        canvasRef.current?.querySelector(selector) ?? canvasRef.current
      )?.scrollIntoView({
        block: focus ? "nearest" : "start",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
      });
    });
  };
  const selectLifecycle = (id: LifecycleKey) => {
    const node = lifecycle.find((item) => item.id === id)!;
    setStage(node.stage);
    setLifecycleId(id);
    setStep(null);
    setDetailTab("overview");
    persist({
      nir_stage: node.stage,
      nir_node: id,
      nir_step: null,
      nir_tab: "overview",
    });
  };
  const changeTab = (tab: NodeDetailTab) => {
    setDetailTab(tab);
    const nextStep =
      stage === "search"
        ? tab === "reason"
          ? "decision"
          : tab === "config"
            ? "model"
            : "validation"
        : null;
    setStep(nextStep);
    persist({ nir_tab: tab, nir_step: nextStep });
  };
  const pendingQuestion = record(record(workflow).user_question);
  const waiting =
    pendingQuestion.status === "waiting" &&
    record(workflow).search_session_id === summary.session_id;
  const selectedLifecycle =
    lifecycle.find((node) => node.id === lifecycleId && node.stage === stage) ??
    (stage === "data"
      ? lifecycle.find((node) => node.id === "input")
      : undefined);
  return (
    <section
      className="min-w-0 space-y-4"
      data-testid="autonomous-round-explorer"
    >
      {waiting && (
        <section
          className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-4"
          role="status"
        >
          <h3 className="font-medium">
            {zh ? "等待你的回复" : "Waiting for your reply"}
          </h3>
          <p className="my-2 text-sm whitespace-pre-wrap">
            {text(pendingQuestion.question, 4000)}
          </p>
          <Link
            className="text-primary text-sm underline"
            href={`/workspace/chats/${encodeURIComponent(threadId)}`}
          >
            {zh ? "返回原对话回复" : "Reply in the original chat"}
          </Link>
        </section>
      )}
      <div className="process-explorer-layout">
        <ProcessRoundNavigation
          stage={stage}
          selectedId={choice?.decision_id ?? ""}
          lifecycle={lifecycle}
          lifecycleId={selectedLifecycle?.id ?? null}
          onLifecycle={selectLifecycle}
          rows={rows}
          summary={summary}
          zh={zh}
          total={history.total}
          loading={history.isPending}
          failed={history.isError}
          hasMore={Boolean(history.hasNextPage)}
          loadingMore={history.isFetchingNextPage}
          onStage={(key) => {
            setStage(key);
            setLifecycleId(null);
            setStep(null);
            persist({ nir_stage: key, nir_node: null, nir_step: null });
            if (key === "data") selectLifecycle("input");
          }}
          onRound={(row) => onSelect(row)}
          onMore={() => void history.fetchNextPage()}
          onRefresh={() => void history.refetch()}
        />
        <div
          className="process-selected-content min-w-0"
          data-testid="process-selected-content"
        >
          {stage === "search" ? (
            <div
              className="min-w-0 space-y-4"
              ref={canvasRef}
              data-testid="process-workbench"
            >
              <details
                className="rounded-lg border"
                data-testid="round-trend-disclosure"
              >
                <summary className="cursor-pointer px-4 py-3 text-sm font-medium">
                  {zh
                    ? "轮次对比与优化趋势"
                    : "Round comparison and optimization trend"}
                </summary>
                <RoundOptimizationTrend
                  summary={summary}
                  selectedId={choice?.decision_id ?? ""}
                  zh={zh}
                  onSelect={(id) => {
                    const row =
                      rows.find((item) => item.decision_id === id) ??
                      parseDecision({ decision_id: id });
                    if (row) {
                      onSelect(row);
                    }
                  }}
                />
              </details>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-muted-foreground text-xs">
                  {zh
                    ? "左侧切换轮次；点击管线步骤查看行动与结果。"
                    : "Select a round on the left to inspect its pipeline and step spectra."}
                </p>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  {retained && (
                    <button
                      className="rounded border px-3 py-2"
                      onClick={() => {
                        onSelect(retained);
                      }}
                    >
                      {zh ? "查看保留轮次" : "Retained round"}
                    </button>
                  )}
                </div>
              </div>
              {history.isError && (
                <p role="alert" className="text-sm">
                  {zh
                    ? "轮次读取失败，已加载内容仍可查看。"
                    : "History read failed; loaded records remain available."}
                  <button
                    className="ml-2 underline"
                    onClick={() => void history.refetch()}
                  >
                    {zh ? "重新读取" : "Retry"}
                  </button>
                </p>
              )}
              <div className="space-y-5" data-testid="selected-round-evidence">
                {rows
                  .filter((row) => row.decision_id === choice?.decision_id)
                  .map((row) => {
                    const active =
                      expanded && choice?.decision_id === row.decision_id;
                    const stopping = ["STOP", "RETAIN"].includes(row.action);
                    const base = rows.find(
                      (item) =>
                        item.experiment_id &&
                        item.experiment_id === row.base_experiment_id,
                    );
                    return (
                      <article
                        key={row.decision_id}
                        data-flow-id={row.decision_id}
                        className="scroll-mt-4 space-y-2"
                      >
                        {stopping ? (
                          <RoundNodeCard
                            decision={row}
                            rows={rows}
                            summary={summary}
                            zh={zh}
                            selected={active}
                            onSelect={onSelect}
                          />
                        ) : (
                          <RoundPipelineRow
                            decision={row}
                            base={base}
                            summary={summary}
                            zh={zh}
                            expanded={active}
                            selectedStep={step ?? "validation"}
                            onSelect={(focus) => {
                              if (!focus && active) {
                                setExpanded(false);
                                setSelected(row);
                                setRestoredId("");
                              } else onSelect(row, focus);
                            }}
                          />
                        )}
                        {active && (
                          <div
                            id={`expanded-${row.decision_id}`}
                            className="pipeline-expanded-detail"
                          >
                            <AutonomousRoundDetail
                              embedded
                              key={row.decision_id}
                              threadId={threadId}
                              summary={summary}
                              decision={row}
                              zh={zh}
                              onSelect={onSelect}
                              tab={detailTab}
                              onTabChange={changeTab}
                              focusStep={step}
                              spectra={spectra}
                              spectraStatus={
                                spectraQuery.isError
                                  ? "error"
                                  : spectraQuery.isPending &&
                                      choice?.status === "succeeded"
                                    ? "loading"
                                    : spectra
                                      ? "available"
                                      : record(spectraQuery.data)
                                            .evidence_status === "available"
                                        ? "mismatch"
                                        : record(spectraQuery.data)
                                              .evidence_status === "failed"
                                          ? "failed"
                                          : record(spectraQuery.data)
                                                .evidence_status ===
                                              "unavailable"
                                            ? "unavailable"
                                            : "missing"
                              }
                              onRetrySpectra={() => void spectraQuery.refetch()}
                            />
                          </div>
                        )}
                      </article>
                    );
                  })}
                {expanded &&
                  choice &&
                  !rows.some(
                    (row) => row.decision_id === choice.decision_id,
                  ) && (
                    <AutonomousRoundDetail
                      embedded
                      key={choice.decision_id}
                      threadId={threadId}
                      summary={summary}
                      decision={choice}
                      zh={zh}
                      onSelect={onSelect}
                      tab={detailTab}
                      onTabChange={changeTab}
                      focusStep={step}
                    />
                  )}
              </div>
              {!rows.length && (
                <p className="text-muted-foreground p-4 text-sm">
                  {history.isPending
                    ? zh
                      ? "正在读取轮次记录…"
                      : "Reading history…"
                    : zh
                      ? "尚无已接受的建模决策。"
                      : "No accepted decisions yet."}
                </p>
              )}
              {history.hasNextPage && (
                <p className="text-muted-foreground text-xs">
                  {zh
                    ? "左侧可继续加载历史轮次。"
                    : "Load more rounds in the sidebar."}
                </p>
              )}
              {terminal &&
                !rows.some((row) =>
                  ["STOP", "RETAIN"].includes(row.action),
                ) && (
                  <p className="text-muted-foreground text-sm">
                    {stopExplanation(summary, zh)}
                  </p>
                )}
              {searchNotes}
            </div>
          ) : (
            <div className="space-y-4">
              {stage !== "data" && (
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {lifecycle
                    .filter((node) => node.stage === stage)
                    .map((node) => (
                      <button
                        type="button"
                        key={node.id}
                        data-testid="lifecycle-node"
                        data-node-id={node.id}
                        aria-pressed={lifecycleId === node.id}
                        className={`rounded-xl border p-4 text-left ${lifecycleId === node.id ? "border-primary bg-accent" : "bg-card hover:bg-accent/40"}`}
                        onClick={() => selectLifecycle(node.id)}
                      >
                        <span className="flex flex-wrap justify-between gap-2 text-sm">
                          <strong>{node.title[zh ? 0 : 1]}</strong>
                          <span className="text-muted-foreground text-xs">
                            {lifecycleStatus(node.status, zh)}
                          </span>
                        </span>
                        <span className="text-muted-foreground mt-2 block text-xs">
                          {node.method[zh ? 0 : 1]}
                        </span>
                      </button>
                    ))}
                </div>
              )}
              {selectedLifecycle ? (
                <div className="pipeline-expanded-detail">
                  <LifecycleNodeDetail
                    node={selectedLifecycle}
                    threadId={threadId}
                    summary={summary}
                    zh={zh}
                    tab={detailTab}
                    onTabChange={changeTab}
                  />
                </div>
              ) : stage === "final" ? (
                finalContent
              ) : (
                deliveryContent
              )}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}
