"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

import { useAuth } from "@/core/auth/AuthProvider";

import { loadProcess, ProcessAPIError } from "./process-api";
import { type ProcessSummary } from "./process-contract";

export function useProcessResource<T>(
  threadId: string,
  resource: string | null,
  poll = false,
) {
  const { user } = useAuth();
  const client = useQueryClient();
  const key = ["nir-process", user?.id, threadId, resource];
  const query = useQuery<T>({
    queryKey: key,
    queryFn: ({ signal }) =>
      loadProcess<T>(
        threadId,
        resource!,
        signal,
        resource?.match(/^runs\/[^/?]+(?:\?.*)?$/)
          ? client.getQueryData<ProcessSummary>(key)
          : undefined,
      ),
    enabled: Boolean(user && resource),
    staleTime: resource?.includes("charts/") ? Infinity : 2000,
    retry: (count, error) =>
      !(error instanceof ProcessAPIError && error.status < 500) && count < 3,
    retryDelay: (count) => Math.min(30000, 3000 * 2 ** count),
    refetchInterval: (current) => {
      if (!poll) return false;
      const data = current.state.data as ProcessSummary | undefined;
      if (
        data?.stage &&
        ["completed", "blocked", "registered"].includes(data.stage) &&
        !data.attempts?.some(
          (attempt) => attempt.execution_status === "running",
        )
      )
        return false;
      return Math.min(30000, 3000 * 2 ** current.state.fetchFailureCount);
    },
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });
  useEffect(
    () => () => {
      void client.cancelQueries({
        queryKey: ["nir-process", user?.id, threadId, resource],
      });
    },
    [client, user?.id, threadId, resource],
  );
  return query;
}
