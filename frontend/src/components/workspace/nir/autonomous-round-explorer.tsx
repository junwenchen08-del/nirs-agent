"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";

import {
  isAutonomousTerminal,
  parseDecision,
  record,
  stopExplanation,
  text,
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import {
  processLifecycle,
  lifecycleStatus,
  type LifecycleKey,
} from "@/core/nir/process-lifecycle";
import { useRoundHistory } from "@/core/nir/round-hooks";

import { AutonomousRoundDetail } from "./autonomous-round-detail";
import { LifecycleNodeDetail } from "./lifecycle-node-detail";
import { type NodeDetailTab } from "./node-detail-tabs";
import { RoundNodeCard } from "./round-node-card";

const RoundGraph = dynamic(
  () =>
    import("./autonomous-round-graph").then(
      (module) => module.AutonomousRoundGraph,
    ),
  {
    ssr: false,
    // Without a placeholder the graph column stays blank while the chunk loads.
    loading: () => (
      <div
        className="bg-muted/20 h-[640px] animate-pulse rounded-xl border"
        aria-hidden
      />
    ),
  },
);

export function AutonomousRoundExplorer({
  threadId,
  summary,
  workflow,
  zh,
}: {
  threadId: string;
  summary: AutonomousSummary;
  workflow?: unknown;
  zh: boolean;
}) {
  const history = useRoundHistory(
    threadId,
    summary.session_id,
    !isAutonomousTerminal(summary.stage),
  );
  const [selected, setSelected] = useState<DecisionView | null>(null);
  const [restoredId, setRestoredId] = useState("");
  const [layout, setLayout] = useState<"flow" | "timeline" | "graph">("flow");
  const [lifecycleId, setLifecycleId] = useState<LifecycleKey | null>(null);
  const [detailTab, setDetailTab] = useState<NodeDetailTab>("evaluation");
  const tabPinned = useRef(false);
  const canvasRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
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
    if (params.get("nir_session") === summary.session_id) {
      const did = params.get("nir_decision") ?? "";
      if (/^[A-Za-z0-9_-]{1,128}$/.test(did)) setRestoredId(did);
      if (params.get("nir_view") === "graph") setLayout("graph");
      if (params.get("nir_view") === "timeline") setLayout("timeline");
      const node = params.get("nir_node");
      if (
        [
          "input",
          "inspect",
          "prepare",
          "freeze",
          "test",
          "delivery",
          "approval",
          "registration",
        ].includes(node ?? "")
      )
        setLifecycleId(node as LifecycleKey);
      const tab = params.get("nir_tab");
      if (["overview", "config", "evaluation", "reason"].includes(tab ?? "")) {
        setDetailTab(tab as NodeDetailTab);
        tabPinned.current = true;
      }
    }
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
  const persist = (decisionId: string, view: string) => {
    const params = new URLSearchParams(window.location.search);
    params.set("nir_session", summary.session_id);
    params.set("nir_decision", decisionId);
    params.set("nir_view", view);
    params.delete("nir_node");
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}?${params.toString()}`,
    );
  };
  const onSelect = (decision: DecisionView) => {
    setLifecycleId(null);
    if (!tabPinned.current)
      setDetailTab(decision.status === "succeeded" ? "evaluation" : "overview");
    setSelected(decision);
    setRestoredId("");
    persist(decision.decision_id, layout);
  };
  const pendingQuestion = record(record(workflow).user_question);
  const waiting =
    pendingQuestion.status === "waiting" &&
    record(workflow).search_session_id === summary.session_id;
  const terminal = isAutonomousTerminal(summary.stage);
  const changeTab = (value: NodeDetailTab) => {
    tabPinned.current = true;
    setDetailTab(value);
    const params = new URLSearchParams(window.location.search);
    params.set("nir_tab", value);
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}?${params}`,
    );
  };
  return (
    <section className="space-y-4" data-testid="autonomous-round-explorer">
      {waiting && (
        <section
          className="rounded-xl border border-amber-500/40 bg-amber-500/5 p-4"
          role="status"
        >
          <h3 className="font-medium">
            {zh ? "等待你的回复" : "Waiting for your reply"}
          </h3>
          <p className="mt-2 text-sm whitespace-pre-wrap">
            {text(pendingQuestion.question, 4000)}
          </p>
          <p className="text-muted-foreground mt-2 text-xs">
            {zh ? "请在聊天中回复后继续。" : "Reply in the chat to continue."}
          </p>
        </section>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div
          className="flex flex-wrap gap-2"
          role="group"
          aria-label={zh ? "过程布局" : "Process layout"}
        >
          {(["flow", "timeline", "graph"] as const).map((view) => (
            <button
              type="button"
              key={view}
              aria-pressed={layout === view}
              className={`rounded-lg border px-3 py-2 text-sm ${layout === view ? "bg-accent text-accent-foreground border-primary" : "bg-card"}`}
              onClick={() => {
                setLayout(view);
                if (view === "graph") {
                  setLifecycleId(null);
                  if (choice) persist(choice.decision_id, view);
                } else if (!lifecycleId && choice)
                  persist(choice.decision_id, view);
                else {
                  const params = new URLSearchParams(window.location.search);
                  params.set("nir_view", view);
                  window.history.replaceState(
                    window.history.state,
                    "",
                    `${window.location.pathname}?${params}`,
                  );
                }
              }}
            >
              {
                {
                  flow: ["全过程画布", "Full process"],
                  timeline: ["轮次时间线", "Round timeline"],
                  graph: ["方案关系图", "Recorded relations"],
                }[view][zh ? 0 : 1]
              }
            </button>
          ))}
        </div>
        <span className="text-muted-foreground text-xs">
          {zh
            ? `已加载 ${rows.length} / ${history.total ?? "—"} 次决策`
            : `${rows.length} / ${history.total ?? "—"} decisions loaded`}
        </span>
      </div>
      {history.isError && (
        <p role="alert" className="text-sm">
          {zh
            ? "轮次读取失败，已加载内容仍可查看。"
            : "History read failed; loaded records remain available."}
          <button
            type="button"
            className="ml-2 underline"
            onClick={() => void history.refetch()}
          >
            {zh ? "重新读取" : "Retry"}
          </button>
        </p>
      )}
      <div className="process-workbench" data-testid="process-workbench">
        <nav
          className="process-node-pane"
          aria-label={zh ? "建模轮次节点" : "Modeling round nodes"}
        >
          <div className="mb-3 flex shrink-0 flex-wrap items-center justify-between gap-2">
            <h3 className="font-semibold">
              {zh ? "流程节点" : "Process nodes"}
            </h3>
            <span className="text-muted-foreground text-xs">
              {rows.length + lifecycle.length} {zh ? "个节点" : "nodes"}
            </span>
          </div>
          {layout !== "graph" && (
            <div
              className="mb-3 flex shrink-0 flex-wrap gap-1"
              role="group"
              aria-label={zh ? "阶段定位" : "Jump to stage"}
            >
              {[
                ["input", "数据准备", "Data"],
                [rows[0]?.decision_id ?? "prepare", "方案搜索", "Search"],
                ["freeze", "最终评价", "Final"],
                ["delivery", "交付入库", "Delivery"],
              ].map(([key, cn, en]) => (
                <button
                  type="button"
                  key={key}
                  className="hover:bg-accent rounded px-2 py-2 text-xs"
                  onClick={() => {
                    canvasRef.current
                      ?.querySelector(`[data-flow-id="${key}"]`)
                      ?.scrollIntoView({
                        block: "start",
                        behavior: window.matchMedia(
                          "(prefers-reduced-motion: reduce)",
                        ).matches
                          ? "auto"
                          : "smooth",
                      });
                  }}
                >
                  {zh ? cn : en}
                </button>
              ))}
            </div>
          )}
          <div
            ref={canvasRef}
            className="bg-card min-h-0 flex-1 overflow-auto rounded-xl border"
            data-testid="process-flow-canvas"
            style={
              layout === "flow"
                ? {
                    backgroundImage:
                      "radial-gradient(var(--border) 1px, transparent 1px)",
                    backgroundSize: "18px 18px",
                  }
                : undefined
            }
          >
            {layout === "graph" ? (
              <RoundGraph
                summary={summary}
                rows={rows}
                selectedId={choice?.decision_id ?? ""}
                zh={zh}
                onSelect={onSelect}
              />
            ) : (
              <ol
                className={`mx-auto max-w-lg px-5 py-6 ${layout === "flow" ? "space-y-8" : "space-y-3"}`}
                style={{ zoom }}
              >
                {[
                  ...lifecycle.filter((node) => node.stage === "data"),
                  ...rows,
                  ...lifecycle.filter((node) => node.stage !== "data"),
                ].map((node, index, all) => {
                  const decision = "decision_id" in node;
                  const key = decision ? node.decision_id : node.id;
                  return (
                    <li
                      key={key}
                      data-flow-id={key}
                      className="relative scroll-mt-5"
                    >
                      {layout === "flow" && index < all.length - 1 && (
                        <span
                          aria-hidden
                          className="border-border absolute top-full left-1/2 h-8 border-l"
                        />
                      )}
                      {decision ? (
                        <RoundNodeCard
                          decision={node}
                          rows={rows}
                          summary={summary}
                          zh={zh}
                          selected={
                            !lifecycleId &&
                            choice?.decision_id === node.decision_id
                          }
                          onSelect={onSelect}
                        />
                      ) : (
                        <button
                          type="button"
                          data-testid="lifecycle-node"
                          data-node-id={node.id}
                          aria-pressed={lifecycleId === node.id}
                          className={`w-full rounded-xl border p-4 text-left text-sm ${lifecycleId === node.id ? "border-primary bg-accent" : "bg-card hover:bg-accent/40"} ${["pending", "not_recorded"].includes(node.status) ? "border-dashed" : ""}`}
                          onClick={() => {
                            setLifecycleId(node.id);
                            if (!tabPinned.current) setDetailTab("overview");
                            const params = new URLSearchParams(
                              window.location.search,
                            );
                            params.set("nir_session", summary.session_id);
                            params.set("nir_node", node.id);
                            params.set("nir_view", layout);
                            window.history.replaceState(
                              window.history.state,
                              "",
                              `${window.location.pathname}?${params}`,
                            );
                          }}
                        >
                          <span className="flex flex-wrap justify-between gap-2">
                            <strong className="font-medium">
                              {node.title[zh ? 0 : 1]}
                            </strong>
                            <span className="text-muted-foreground text-xs">
                              {lifecycleStatus(node.status, zh)}
                            </span>
                          </span>
                          <span className="text-muted-foreground mt-2 block text-xs">
                            {node.method[zh ? 0 : 1]}
                          </span>
                        </button>
                      )}
                    </li>
                  );
                })}
                {!rows.length && (
                  <li className="text-muted-foreground text-sm">
                    {history.isPending
                      ? zh
                        ? "正在读取轮次记录…"
                        : "Reading history…"
                      : zh
                        ? "尚无已接受的建模决策。"
                        : "No accepted modeling decisions yet."}
                  </li>
                )}
                {history.hasNextPage && (
                  <li>
                    <button
                      type="button"
                      disabled={history.isFetchingNextPage}
                      className="w-full rounded-md border px-3 py-2 text-sm"
                      onClick={() => void history.fetchNextPage()}
                    >
                      {zh ? "加载更多轮次" : "Load more rounds"}
                    </button>
                  </li>
                )}
              </ol>
            )}
          </div>
          {layout === "graph" && history.hasNextPage && (
            <button
              type="button"
              disabled={history.isFetchingNextPage}
              className="my-2 rounded border p-2 text-sm"
              onClick={() => void history.fetchNextPage()}
            >
              {zh ? "加载更多轮次" : "Load more rounds"}
            </button>
          )}
          <div className="mt-3 flex shrink-0 flex-wrap items-center gap-2">
            <button
              type="button"
              className="bg-card rounded-md border px-3 py-2 text-sm"
              aria-label={zh ? "放大流程" : "Zoom in"}
              disabled={zoom >= 1.3 || layout === "graph"}
              onClick={() => setZoom((v) => Math.min(1.3, v + 0.1))}
            >
              +
            </button>
            <button
              type="button"
              className="bg-card rounded-md border px-3 py-2 text-sm"
              aria-label={zh ? "缩小流程" : "Zoom out"}
              disabled={zoom <= 0.7 || layout === "graph"}
              onClick={() => setZoom((v) => Math.max(0.7, v - 0.1))}
            >
              −
            </button>
            <button
              type="button"
              className="bg-card rounded-md border px-3 py-2 text-xs"
              onClick={() =>
                canvasRef.current
                  ?.querySelector(
                    `[data-flow-id="${lifecycleId ?? choice?.decision_id}"]`,
                  )
                  ?.scrollIntoView({
                    block: "center",
                    behavior: window.matchMedia(
                      "(prefers-reduced-motion: reduce)",
                    ).matches
                      ? "auto"
                      : "smooth",
                  })
              }
            >
              {zh ? "定位选中" : "Locate selection"}
            </button>
            <button
              type="button"
              className="bg-card rounded-md border px-3 py-2 text-xs"
              onClick={() =>
                canvasRef.current?.scrollTo({
                  top: 0,
                  behavior: window.matchMedia(
                    "(prefers-reduced-motion: reduce)",
                  ).matches
                    ? "auto"
                    : "smooth",
                })
              }
            >
              {zh ? "回到开始" : "Back to start"}
            </button>
            <span className="text-muted-foreground ml-auto text-xs">
              {Math.round(zoom * 100)}%
            </span>
          </div>
          <p className="text-muted-foreground mt-2 shrink-0 text-xs">
            {layout === "graph"
              ? zh
                ? "仅连接记录的反馈与保留关系。"
                : "Recorded feedback and retention links only."
              : zh
                ? "连线表示展示顺序；虚线节点表示未完成或记录未提供。"
                : "Lines show display order; dashed nodes are pending or unrecorded."}
          </p>
        </nav>
        {lifecycleId ? (
          <LifecycleNodeDetail
            node={lifecycle.find((node) => node.id === lifecycleId)!}
            threadId={threadId}
            summary={summary}
            zh={zh}
            tab={detailTab}
            onTabChange={changeTab}
          />
        ) : choice ? (
          <AutonomousRoundDetail
            key={choice.decision_id}
            threadId={threadId}
            summary={summary}
            decision={choice}
            zh={zh}
            onSelect={onSelect}
            tab={detailTab}
            onTabChange={changeTab}
          />
        ) : (
          <div className="process-detail-pane bg-card rounded-xl border p-6 text-sm">
            {zh ? "等待记录建模决策。" : "Waiting for a recorded decision."}
          </div>
        )}
      </div>
      {terminal &&
        !rows.some((row) => ["STOP", "RETAIN"].includes(row.action)) && (
          <p className="text-muted-foreground text-sm">
            {stopExplanation(summary, zh)}
          </p>
        )}
      <button
        type="button"
        className="rounded-md border px-3 py-2 text-sm"
        onClick={() => void history.refetch()}
      >
        {zh ? "刷新轮次" : "Refresh rounds"}
      </button>
    </section>
  );
}
