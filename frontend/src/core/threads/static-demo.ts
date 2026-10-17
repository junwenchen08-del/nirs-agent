import type { ThreadState } from "@langchain/langgraph-sdk";
import type { ThreadsClient } from "@langchain/langgraph-sdk/client";

import type { AgentThread, AgentThreadState } from "./types";

export const DEMO_THREAD_IDS = [
  "8c2ec2e1-4c63-4d81-8c3d-cc3150267d04",
] as const;

export type ThreadSearchParams = NonNullable<
  Parameters<ThreadsClient["search"]>[0]
>;

export async function loadStaticDemoThreads(
  params: ThreadSearchParams = {},
): Promise<AgentThread[]> {
  const threads = await Promise.all(
    DEMO_THREAD_IDS.map((threadId) => loadStaticDemoThread(threadId)),
  );

  const sortBy = params.sortBy ?? "updated_at";
  const sortOrder = params.sortOrder ?? "desc";
  const sortedThreads = [...threads].sort((a, b) => {
    const aTimestamp = (a as unknown as Record<string, unknown>)[sortBy];
    const bTimestamp = (b as unknown as Record<string, unknown>)[sortBy];
    const aParsed = typeof aTimestamp === "string" ? Date.parse(aTimestamp) : 0;
    const bParsed = typeof bTimestamp === "string" ? Date.parse(bTimestamp) : 0;
    const aValue = Number.isNaN(aParsed) ? 0 : aParsed;
    const bValue = Number.isNaN(bParsed) ? 0 : bParsed;
    return sortOrder === "asc" ? aValue - bValue : bValue - aValue;
  });

  const offset = Math.max(0, Math.floor(params.offset ?? 0));
  const limit =
    typeof params.limit === "number"
      ? Math.max(0, Math.floor(params.limit))
      : sortedThreads.length;
  return sortedThreads.slice(offset, offset + limit);
}

export async function loadStaticDemoThread(
  threadId: string,
): Promise<AgentThread> {
  const response = await globalThis.fetch(
    `/demo/threads/${encodeURIComponent(threadId)}/thread.json`,
  );
  if (!response.ok) {
    throw new Error(`Failed to load demo thread ${threadId}`);
  }
  const thread = (await response.json()) as AgentThread;
  return {
    ...thread,
    thread_id: threadId,
    updated_at: thread.updated_at ?? thread.created_at,
  };
}

export function staticDemoThreadState(
  thread: AgentThread,
): ThreadState<AgentThreadState> {
  return {
    values: thread.values,
    next: [],
    checkpoint: {
      thread_id: thread.thread_id,
      checkpoint_ns: "",
      checkpoint_id: null,
      checkpoint_map: null,
    },
    metadata: thread.metadata ?? null,
    created_at: thread.updated_at ?? thread.created_at ?? null,
    parent_checkpoint: null,
    tasks: [],
  };
}
