"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { useAuth } from "@/core/auth/AuthProvider";

import { autonomousPollingInterval } from "./autonomous";
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
  const publicationPolling = useRef<{ key: string; startedAt: number } | null>(
    null,
  );
  const query = useQuery<T>({
    queryKey: key,
    queryFn: ({ signal }) =>
      loadProcess<T>(
        threadId,
        resource!,
        signal,
        resource?.match(/^(?:runs|autonomous)\/[^/?]+(?:\?.*)?$/)
          ? client.getQueryData<
              ProcessSummary | { session_id: string; revision: string }
            >(key)
          : undefined,
      ),
    enabled: Boolean(user && resource),
    staleTime:
      resource?.includes("charts/") || resource?.includes("spectra/")
        ? Infinity
        : 2000,
    retry: (count, error) =>
      !(error instanceof ProcessAPIError && error.status < 500) && count < 3,
    retryDelay: (count) => Math.min(30000, 3000 * 2 ** count),
    refetchInterval: (current) => {
      if (!poll) return false;
      const data = current.state.data as ProcessSummary | undefined;
      const pollingKey = JSON.stringify(key);
      const interval = autonomousPollingInterval(data);
      if (interval === 5000) {
        if (publicationPolling.current?.key !== pollingKey)
          publicationPolling.current = {
            key: pollingKey,
            startedAt: Date.now(),
          };
        return (
          autonomousPollingInterval(
            data,
            Date.now() - publicationPolling.current.startedAt,
          ) ?? false
        );
      }
      publicationPolling.current = null;
      if (interval === false) return false;
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
