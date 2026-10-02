"use client";

import { useQuery } from "@tanstack/react-query";

import { fetchNirLibraryEnabled } from "./api";

export function useNirLibraryEnabled() {
  const { data, isPending } = useQuery({
    queryKey: ["features", "nir_library"],
    queryFn: fetchNirLibraryEnabled,
    staleTime: 0,
    refetchOnMount: true,
    retry: false,
  });
  return { enabled: data === true, isLoading: isPending };
}
