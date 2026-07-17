"use client";

import {
  ActivityIcon,
  AlertTriangleIcon,
  ArrowUpRightIcon,
  CheckCircle2Icon,
  DownloadIcon,
  FlaskConicalIcon,
  PlusIcon,
  ShieldCheckIcon,
  Trash2Icon,
  XCircleIcon,
} from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { useAuth } from "@/core/auth/AuthProvider";
import { useI18n } from "@/core/i18n/hooks";
import {
  appendEvaluationHistory,
  evaluationStorageKey,
  historyEntryFromSummary,
  loadNirEvaluationScenarios,
  runNirEvaluation,
  type NIRBatchEvaluationResponse,
  type NIREvaluationHistoryEntry,
  type NIREvaluationMapping,
  type NIREvaluationScenario,
} from "@/core/nir-evaluations";
import { cn } from "@/lib/utils";

const MAPPINGS_KEY = "deerflow:nir-evaluation-mappings:v1";
const HISTORY_KEY = "deerflow:nir-evaluation-history:v1";

function createMapping(index = 0): NIREvaluationMapping {
  const randomId = globalThis.crypto?.randomUUID?.();
  return {
    id: randomId ?? `mapping-${Date.now()}-${index}`,
    threadId: "",
    scenarioId: "",
  };
}

function readStoredArray<T>(key: string): T[] {
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? "[]") as unknown;
    return Array.isArray(value) ? (value as T[]) : [];
  } catch {
    return [];
  }
}

function formatDuration(durationMs: number) {
  if (durationMs < 1000) return `${Math.round(durationMs)} ms`;
  return `${(durationMs / 1000).toFixed(1)} s`;
}

export function NIREvaluationDashboard() {
  const { user } = useAuth();
  return (
    <UserScopedNIREvaluationDashboard
      key={user?.id ?? "signed-out"}
      userId={user?.id ?? null}
    />
  );
}

function UserScopedNIREvaluationDashboard({
  userId,
}: {
  userId: string | null;
}) {
  const { t } = useI18n();
  const copy = t.nirEvaluations;
  const mappingsKey = userId
    ? evaluationStorageKey(MAPPINGS_KEY, userId)
    : null;
  const historyKey = userId ? evaluationStorageKey(HISTORY_KEY, userId) : null;
  const [scenarios, setScenarios] = useState<NIREvaluationScenario[]>([]);
  const [mappings, setMappings] = useState<NIREvaluationMapping[]>([
    createMapping(),
  ]);
  const [history, setHistory] = useState<NIREvaluationHistoryEntry[]>([]);
  const [result, setResult] = useState<NIRBatchEvaluationResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [isLoadingCatalog, setIsLoadingCatalog] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [storageReady, setStorageReady] = useState(false);

  useEffect(() => {
    setStorageReady(false);
    setMappings([createMapping()]);
    setHistory([]);
    if (!mappingsKey || !historyKey) return;

    const storedMappings = readStoredArray<NIREvaluationMapping>(mappingsKey)
      .filter(
        (mapping) =>
          typeof mapping.id === "string" &&
          typeof mapping.threadId === "string" &&
          typeof mapping.scenarioId === "string",
      )
      .slice(0, 100);
    const storedHistory = readStoredArray<NIREvaluationHistoryEntry>(
      historyKey,
    ).slice(0, 12);
    if (storedMappings.length) setMappings(storedMappings);
    setHistory(storedHistory);
    setStorageReady(true);
  }, [historyKey, mappingsKey]);

  useEffect(() => {
    if (!storageReady || !mappingsKey) return;
    localStorage.setItem(mappingsKey, JSON.stringify(mappings));
  }, [mappings, mappingsKey, storageReady]);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoadingCatalog(true);
    loadNirEvaluationScenarios(controller.signal)
      .then((catalog) => {
        setScenarios(catalog.scenarios);
        setCatalogError(null);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setCatalogError(
          error instanceof Error ? error.message : copy.loadError,
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setIsLoadingCatalog(false);
      });
    return () => controller.abort();
  }, [copy.loadError]);

  const validMappings = mappings.filter(
    (mapping) => mapping.threadId.trim() && mapping.scenarioId,
  );
  const hasDuplicateScenarios =
    new Set(validMappings.map((mapping) => mapping.scenarioId)).size !==
    validMappings.length;

  function updateMapping(
    id: string,
    field: "threadId" | "scenarioId",
    value: string,
  ) {
    setMappings((current) =>
      current.map((mapping) =>
        mapping.id === id ? { ...mapping, [field]: value } : mapping,
      ),
    );
  }

  async function runEvaluation() {
    setIsRunning(true);
    setRunError(null);
    try {
      const response = await runNirEvaluation(
        validMappings.map((mapping) => ({
          thread_id: mapping.threadId.trim(),
          scenario_id: mapping.scenarioId,
        })),
      );
      setResult(response);
      setHistory((current) => {
        const next = appendEvaluationHistory(
          current,
          historyEntryFromSummary(response.summary),
        );
        if (historyKey) localStorage.setItem(historyKey, JSON.stringify(next));
        return next;
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : copy.runError;
      setRunError(message);
      toast.error(message);
    } finally {
      setIsRunning(false);
    }
  }

  function exportEvidence() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `nir-evaluation-${result.summary.generated_at.replaceAll(":", "-")}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="relative min-h-full w-full overflow-hidden bg-[radial-gradient(circle_at_12%_8%,color-mix(in_oklab,var(--color-emerald-400)_14%,transparent),transparent_30%),radial-gradient(circle_at_88%_16%,color-mix(in_oklab,var(--color-amber-300)_12%,transparent),transparent_28%)] px-4 py-8 sm:px-6 lg:px-10">
      <div className="pointer-events-none absolute inset-0 [background-image:linear-gradient(to_right,var(--border)_1px,transparent_1px),linear-gradient(to_bottom,var(--border)_1px,transparent_1px)] [mask-image:linear-gradient(to_bottom,black,transparent_72%)] [background-size:32px_32px] opacity-35" />
      <div className="relative mx-auto flex w-full max-w-7xl flex-col gap-6">
        <header className="bg-background/85 flex flex-col gap-5 rounded-3xl border p-6 shadow-sm backdrop-blur-xl lg:flex-row lg:items-end lg:justify-between lg:p-8">
          <div className="max-w-3xl">
            <div className="mb-3 flex items-center gap-2 text-xs font-semibold tracking-[0.22em] text-emerald-700 uppercase dark:text-emerald-300">
              <FlaskConicalIcon className="size-4" />
              {copy.eyebrow}
            </div>
            <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
              {copy.title}
            </h1>
            <p className="text-muted-foreground mt-3 max-w-2xl text-sm leading-6 sm:text-base">
              {copy.description}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="h-8 gap-2 px-3">
              <ActivityIcon className="size-3.5 text-emerald-600" />
              {scenarios.length} {copy.scenarios}
            </Badge>
            <Badge variant="outline" className="h-8 gap-2 px-3">
              <ShieldCheckIcon className="size-3.5 text-amber-600" />
              {copy.deterministic}
            </Badge>
          </div>
        </header>

        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.55fr)]">
          <Card className="gap-0 overflow-hidden py-0">
            <CardHeader className="border-b py-6">
              <CardTitle>{copy.mappingTitle}</CardTitle>
              <CardDescription>{copy.mappingDescription}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 p-4 sm:p-6">
              {catalogError && (
                <div className="border-destructive/30 bg-destructive/5 text-destructive flex gap-3 rounded-xl border p-4 text-sm">
                  <AlertTriangleIcon className="mt-0.5 size-4 shrink-0" />
                  <span>{catalogError}</span>
                </div>
              )}
              {mappings.map((mapping, index) => (
                <div
                  key={mapping.id}
                  className="bg-muted/25 grid gap-3 rounded-2xl border p-3 sm:grid-cols-[minmax(180px,0.8fr)_minmax(240px,1.2fr)_auto] sm:items-center"
                >
                  <div>
                    <label
                      className="text-muted-foreground mb-1.5 block text-xs font-medium"
                      htmlFor={`thread-${mapping.id}`}
                    >
                      {copy.threadId} {index + 1}
                    </label>
                    <Input
                      id={`thread-${mapping.id}`}
                      value={mapping.threadId}
                      placeholder={copy.threadPlaceholder}
                      onChange={(event) =>
                        updateMapping(
                          mapping.id,
                          "threadId",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div>
                    <label
                      className="text-muted-foreground mb-1.5 block text-xs font-medium"
                      htmlFor={`scenario-${mapping.id}`}
                    >
                      {copy.scenario}
                    </label>
                    <select
                      id={`scenario-${mapping.id}`}
                      className="border-input bg-background focus-visible:border-ring focus-visible:ring-ring/50 h-9 w-full rounded-md border px-3 text-sm outline-none focus-visible:ring-[3px] disabled:opacity-50"
                      value={mapping.scenarioId}
                      disabled={isLoadingCatalog}
                      onChange={(event) =>
                        updateMapping(
                          mapping.id,
                          "scenarioId",
                          event.target.value,
                        )
                      }
                    >
                      <option value="">{copy.selectScenario}</option>
                      {scenarios.map((scenario) => (
                        <option key={scenario.id} value={scenario.id}>
                          {scenario.id} · {scenario.description}
                        </option>
                      ))}
                    </select>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="text-muted-foreground hover:text-destructive sm:mt-5"
                    aria-label={copy.remove}
                    disabled={mappings.length === 1}
                    onClick={() =>
                      setMappings((current) =>
                        current.filter((item) => item.id !== mapping.id),
                      )
                    }
                  >
                    <Trash2Icon />
                  </Button>
                </div>
              ))}
              {hasDuplicateScenarios && (
                <p className="text-sm text-amber-700 dark:text-amber-300">
                  {copy.duplicateScenario}
                </p>
              )}
              {runError && (
                <p className="text-destructive text-sm">{runError}</p>
              )}
              <div className="flex flex-col-reverse gap-3 pt-2 sm:flex-row sm:items-center sm:justify-between">
                <Button
                  type="button"
                  variant="outline"
                  disabled={mappings.length >= 100}
                  onClick={() =>
                    setMappings((current) => [
                      ...current,
                      createMapping(current.length),
                    ])
                  }
                >
                  <PlusIcon />
                  {copy.addCase}
                </Button>
                <Button
                  type="button"
                  className="bg-emerald-700 text-white hover:bg-emerald-800"
                  disabled={
                    isRunning ||
                    isLoadingCatalog ||
                    validMappings.length === 0 ||
                    hasDuplicateScenarios
                  }
                  onClick={runEvaluation}
                >
                  <ActivityIcon />
                  {isRunning ? copy.running : copy.run}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden border-emerald-900/10 bg-emerald-950 text-white dark:bg-emerald-950">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <ArrowUpRightIcon className="size-5 text-emerald-300" />
                {copy.recentTrend}
              </CardTitle>
              <CardDescription className="text-emerald-100/65">
                {copy.trendDescription}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length ? (
                <div className="flex h-48 items-end gap-2 rounded-2xl border border-white/10 bg-white/5 p-4">
                  {[...history].reverse().map((entry) => (
                    <div
                      key={entry.generatedAt}
                      className="group relative flex h-full min-w-0 flex-1 items-end"
                      title={`${entry.averageScore.toFixed(1)} / 100`}
                    >
                      <div
                        className={cn(
                          "w-full rounded-t-md transition-[height]",
                          entry.policyViolationCount
                            ? "bg-amber-400"
                            : entry.passRate === 1
                              ? "bg-emerald-300"
                              : "bg-cyan-300",
                        )}
                        style={{
                          height: `${Math.max(8, entry.averageScore)}%`,
                        }}
                      />
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex h-48 items-center justify-center rounded-2xl border border-dashed border-white/20 bg-white/5 text-center text-sm text-emerald-100/60">
                  {copy.noHistory}
                </div>
              )}
              <div className="mt-4 flex items-center justify-between text-xs text-emerald-100/55">
                <span>{copy.older}</span>
                <span>{copy.latest}</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {result ? (
          <section className="space-y-6">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
              <MetricCard
                label={copy.passRate}
                value={`${Math.round(result.summary.pass_rate * 100)}%`}
                accent="emerald"
              />
              <MetricCard
                label={copy.averageScore}
                value={result.summary.average_score.toFixed(1)}
                accent="cyan"
              />
              <MetricCard
                label={copy.policyViolations}
                value={String(result.summary.policy_violation_count)}
                accent={
                  result.summary.policy_violation_count ? "amber" : "emerald"
                }
              />
              <MetricCard
                label={copy.tokens}
                value={(
                  result.summary.total_input_tokens +
                  result.summary.total_output_tokens
                ).toLocaleString()}
                accent="slate"
              />
              <MetricCard
                label={copy.duration}
                value={formatDuration(result.summary.total_duration_ms)}
                accent="slate"
              />
            </div>

            <Card className="gap-0 overflow-hidden py-0">
              <CardHeader className="border-b py-6 sm:grid-cols-[1fr_auto]">
                <div>
                  <CardTitle>{copy.results}</CardTitle>
                  <CardDescription className="mt-2">
                    {result.summary.passed}/{result.summary.total} {copy.passed}
                  </CardDescription>
                </div>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={exportEvidence}
                  >
                    <DownloadIcon />
                    {copy.exportEvidence}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setResult(null)}
                  >
                    {copy.clearResult}
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="divide-y p-0">
                {result.summary.results.map((evaluation) => (
                  <details
                    key={evaluation.scenario_id}
                    className="group px-5 py-4 sm:px-6"
                  >
                    <summary className="flex cursor-pointer list-none flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="flex min-w-0 items-center gap-3">
                        {evaluation.passed ? (
                          <CheckCircle2Icon className="size-5 shrink-0 text-emerald-600" />
                        ) : (
                          <XCircleIcon className="text-destructive size-5 shrink-0" />
                        )}
                        <div className="min-w-0">
                          <p className="truncate font-medium">
                            {evaluation.scenario_id}
                          </p>
                          <p className="text-muted-foreground text-xs">
                            {evaluation.trace_id
                              ? `${copy.trace}: ${evaluation.trace_id}`
                              : copy.noTraceId}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge
                          variant={
                            evaluation.passed ? "secondary" : "destructive"
                          }
                        >
                          {evaluation.passed ? copy.passed : copy.failed}
                        </Badge>
                        <span className="font-mono text-lg font-semibold tabular-nums">
                          {evaluation.score.toFixed(1)}
                        </span>
                      </div>
                    </summary>
                    <div className="mt-4 space-y-3 pl-0 sm:pl-8">
                      <Progress value={evaluation.score} />
                      <div className="grid gap-2 md:grid-cols-2">
                        {evaluation.checks.map((check) => (
                          <div
                            key={check.name}
                            className={cn(
                              "rounded-xl border p-3 text-sm",
                              check.passed
                                ? "border-emerald-600/20 bg-emerald-600/5"
                                : "border-destructive/25 bg-destructive/5",
                            )}
                          >
                            <div className="flex items-center justify-between gap-3">
                              <span className="font-medium">{check.name}</span>
                              <span className="font-mono text-xs tabular-nums">
                                {(check.value * check.weight).toFixed(1)}/
                                {check.weight}
                              </span>
                            </div>
                            <p className="text-muted-foreground mt-1 text-xs leading-5 break-words">
                              {check.details}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </details>
                ))}
              </CardContent>
            </Card>
          </section>
        ) : (
          <div className="bg-background/50 text-muted-foreground rounded-2xl border border-dashed px-6 py-12 text-center text-sm">
            {copy.noResults}
          </div>
        )}
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: "emerald" | "cyan" | "amber" | "slate";
}) {
  const accentClass = {
    emerald: "from-emerald-500/15 to-emerald-500/0 border-emerald-500/20",
    cyan: "from-cyan-500/15 to-cyan-500/0 border-cyan-500/20",
    amber: "from-amber-500/15 to-amber-500/0 border-amber-500/20",
    slate: "from-slate-500/10 to-slate-500/0 border-slate-500/15",
  }[accent];
  return (
    <div
      className={cn("rounded-2xl border bg-gradient-to-br p-5", accentClass)}
    >
      <p className="text-muted-foreground text-xs font-medium tracking-wide uppercase">
        {label}
      </p>
      <p className="mt-2 font-mono text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </p>
    </div>
  );
}
