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
import { useRoundHistory } from "@/core/nir/round-hooks";

import { AutonomousRoundDetail } from "./autonomous-round-detail";
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
  const [layout, setLayout] = useState<"timeline" | "graph">("timeline");
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
    window.history.replaceState(
      window.history.state,
      "",
      `${window.location.pathname}?${params.toString()}`,
    );
  };
  const onSelect = (decision: DecisionView) => {
    setSelected(decision);
    setRestoredId("");
    persist(decision.decision_id, layout);
  };
  const pendingQuestion = record(record(workflow).user_question);
  const waiting =
    pendingQuestion.status === "waiting" &&
    record(workflow).search_session_id === summary.session_id;
  const terminal = isAutonomousTerminal(summary.stage);
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
          className="flex gap-2"
          role="group"
          aria-label={zh ? "过程布局" : "Process layout"}
        >
          {(["timeline", "graph"] as const).map((view) => (
            <button
              type="button"
              key={view}
              aria-pressed={layout === view}
              className={`rounded-lg border px-3 py-2 text-sm ${layout === view ? "bg-primary text-primary-foreground" : "bg-background"}`}
              onClick={() => {
                setLayout(view);
                if (choice) persist(choice.decision_id, view);
              }}
            >
              {view === "timeline"
                ? zh
                  ? "轮次时间线"
                  : "Round timeline"
                : zh
                  ? "方案关系图"
                  : "Experiment graph"}
            </button>
          ))}
        </div>
        <span className="text-muted-foreground text-xs">
          {zh
            ? `已加载 ${rows.length} / ${history.total ?? "—"} 次决策`
            : `${rows.length} / ${history.total ?? "—"} decisions loaded`}
        </span>
      </div>
      <p className="text-muted-foreground text-xs">
        {layout === "graph"
          ? zh
            ? "实线表示记录的反馈来源；虚线表示停止决策明确保留的方案。没有记录的分支不会补画。默认聚焦当前轮次，可拖动平移或用画布控制查看全图。"
            : "Solid edges show recorded feedback sources; dashed edges show explicit retention. Unrecorded branches are not inferred. The view focuses the current round; drag to pan or use the canvas controls for the whole graph."
          : zh
            ? "按决策发生顺序展示。点击轮次查看详情；选择历史轮次后，新结果不会切走当前查看内容。"
            : "Decisions in execution order. Selecting a historical round keeps it open as new results arrive."}
      </p>
      {history.isError && (
        <p role="alert" className="text-sm">
          {zh
            ? "轮次读取失败，已加载内容仍可查看。"
            : "Could not update round history. Loaded records remain available."}
          <button
            type="button"
            className="ml-2 underline"
            onClick={() => void history.refetch()}
          >
            {zh ? "重新读取" : "Retry"}
          </button>
        </p>
      )}
      <div
        className={`grid min-w-0 gap-4 ${
          layout === "graph"
            ? "lg:grid-cols-[minmax(320px,1fr)_minmax(0,1.25fr)]"
            : "lg:grid-cols-[minmax(260px,.8fr)_minmax(0,1.4fr)]"
        }`}
      >
        <nav
          className="min-w-0"
          aria-label={zh ? "建模轮次节点" : "Modeling round nodes"}
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
            <ol className="space-y-3 border-l pl-4">
              {rows.map((decision) => (
                <li key={decision.decision_id}>
                  <RoundNodeCard
                    decision={decision}
                    rows={rows}
                    summary={summary}
                    zh={zh}
                    selected={choice?.decision_id === decision.decision_id}
                    onSelect={onSelect}
                  />
                </li>
              ))}
            </ol>
          )}
          {!rows.length && (
            <p className="p-4 text-sm">
              {history.isPending
                ? zh
                  ? "正在读取轮次记录…"
                  : "Reading round history…"
                : zh
                  ? "尚无已接受的建模决策。"
                  : "No accepted modeling decisions yet."}
            </p>
          )}
          {history.hasNextPage && (
            <button
              type="button"
              disabled={history.isFetchingNextPage}
              className="mt-3 rounded-md border px-3 py-2 text-sm"
              onClick={() => void history.fetchNextPage()}
            >
              {zh ? "加载更多轮次" : "Load more rounds"}
            </button>
          )}
          {terminal &&
            !rows.some((row) => ["STOP", "RETAIN"].includes(row.action)) && (
              <section className="mt-3 rounded-xl border p-3 text-sm">
                <h3 className="font-medium">{zh ? "运行结束" : "Run ended"}</h3>
                <p className="mt-2">{stopExplanation(summary, zh)}</p>
              </section>
            )}
        </nav>
        {choice ? (
          <AutonomousRoundDetail
            key={choice.decision_id}
            threadId={threadId}
            summary={summary}
            decision={choice}
            zh={zh}
            onSelect={onSelect}
          />
        ) : (
          <p className="text-muted-foreground rounded-xl border border-dashed p-6 text-sm">
            {zh
              ? "建模决策记录后，在这里查看方案、指标与决策依据。"
              : "Recorded configuration, metrics and decisions will appear here."}
          </p>
        )}
      </div>
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
