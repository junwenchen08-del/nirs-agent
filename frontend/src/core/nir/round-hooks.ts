"use client";

import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { useAuth } from "@/core/auth/AuthProvider";

import { loadProcess, ProcessAPIError } from "./process-api";
import { parseRoundPage, roundHistoryPageSize } from "./round-process";

export function useRoundHistory(threadId: string, sid: string, poll: boolean) {
  const { user } = useAuth();
  const client = useQueryClient();
  const query = useInfiniteQuery({
    queryKey: [
      "nir-process",
      user?.id,
      threadId,
      `autonomous/${sid}/round-history`,
    ],
    initialPageParam: 0,
    queryFn: async ({ pageParam, signal }) =>
      parseRoundPage(
        await loadProcess(
          threadId,
          `autonomous/${sid}/decisions?offset=${pageParam}&limit=${roundHistoryPageSize}`,
          signal,
        ),
        pageParam,
      ),
    getNextPageParam: (last) => last.next_offset ?? undefined,
    enabled: Boolean(user && sid),
    refetchInterval: (current) =>
      poll
        ? Math.min(30000, 3000 * 2 ** current.state.fetchFailureCount)
        : false,
    refetchIntervalInBackground: false,
    retry: (count, error) =>
      !(error instanceof ProcessAPIError && error.status < 500) && count < 3,
    retryDelay: (count) => Math.min(30000, 3000 * 2 ** count),
  });
  useEffect(
    () => () => {
      void client.cancelQueries({
        queryKey: [
          "nir-process",
          user?.id,
          threadId,
          `autonomous/${sid}/round-history`,
        ],
      });
    },
    [client, user?.id, threadId, sid],
  );
  const rows = [
    ...new Map(
      query.data?.pages
        .flatMap((page) => page.data)
        .map((row) => [row.decision_id, row]) ?? [],
    ).values(),
  ];
  return { ...query, rows, total: query.data?.pages[0]?.total ?? null };
}
