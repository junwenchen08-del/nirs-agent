"use client";

import { useQuery } from "@tanstack/react-query";

import { useAuth } from "@/core/auth/AuthProvider";

import { loadModelCatalog } from "./model-catalog";

export function useModelCatalog(enabled: boolean) {
  const { user } = useAuth();
  return useQuery({
    queryKey: ["nir-library", "model-catalog", user?.id ?? null],
    queryFn: ({ signal }) => loadModelCatalog(signal),
    enabled,
    staleTime: 0,
    retry: false,
  });
}
