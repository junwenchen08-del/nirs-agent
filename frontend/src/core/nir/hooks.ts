"use client";

import { useQuery } from "@tanstack/react-query";

import { loadNIRWorkflowState } from "./api";
import { selectLatestNIRWorkflow, selectNIRWorkflow } from "./selectors";

export function useNIRWorkflow(
  threadId: string | undefined,
  liveValue: unknown,
) {
  const liveWorkflow = selectNIRWorkflow(liveValue);
  const recoveredState = useQuery({
    queryKey: ["nir", "workflow-state", threadId],
    queryFn: ({ signal }) => loadNIRWorkflowState(threadId!, signal),
    enabled: Boolean(threadId) && liveWorkflow === null,
    retry: false,
    refetchOnWindowFocus: false,
    staleTime: 30_000,
  });

  return selectLatestNIRWorkflow(liveWorkflow, recoveredState.data);
}
