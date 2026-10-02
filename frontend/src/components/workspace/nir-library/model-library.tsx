"use client";

import {
  ArchiveIcon,
  BoxesIcon,
  LinkIcon,
  RefreshCwIcon,
  ShieldCheckIcon,
  Trash2Icon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/core/i18n/hooks";
import {
  archiveModel,
  attachModel,
  deleteModel,
  getStorageUsage,
  listModels,
  useNirLibraryEnabled,
  type NIRModelVersion,
  type NIRStorageUsage,
} from "@/core/nir-library";

import { FeatureDisabled } from "./dataset-library";
import { formatBytes, StorageUsageCard } from "./storage-usage-card";

type DialogKind = "attach" | "delete" | "details" | null;

function shortHash(value: string) {
  return `${value.slice(0, 10)}…${value.slice(-6)}`;
}

function dateText(value: string | null) {
  return value ? new Date(value).toLocaleString() : "—";
}

export function ModelLibrary() {
  const { t } = useI18n();
  const copy = t.nirLibrary;
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [models, setModels] = useState<NIRModelVersion[]>([]);
  const [usage, setUsage] = useState<NIRStorageUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [dialog, setDialog] = useState<DialogKind>(null);
  const [selected, setSelected] = useState<NIRModelVersion | null>(null);
  const [targetThread, setTargetThread] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(() => setRefreshKey((value) => value + 1), []);

  useEffect(() => {
    if (featureLoading) return;
    if (!enabled) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    Promise.all([
      listModels(controller.signal),
      getStorageUsage(controller.signal),
    ])
      .then(([modelRows, storage]) => {
        setModels(modelRows);
        setUsage(storage);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [enabled, featureLoading, refreshKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return models;
    return models.filter((model) =>
      [
        model.model_id,
        model.version,
        model.method ?? "",
        model.validation_scope,
      ].some((value) => value.toLowerCase().includes(needle)),
    );
  }, [models, query]);

  function open(kind: DialogKind, model: NIRModelVersion) {
    setSelected(model);
    setDialog(kind);
    setTargetThread("");
    setDeleteConfirmation("");
  }

  async function action(work: () => Promise<unknown>) {
    setBusy(true);
    try {
      await work();
      toast.success(copy.operationSuccess);
      setDialog(null);
      refresh();
    } catch (reason) {
      toast.error(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusy(false);
    }
  }

  if (!featureLoading && !enabled) {
    return (
      <FeatureDisabled
        title={copy.featureDisabledTitle}
        description={copy.featureDisabledDescription}
      />
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 p-4 sm:p-6">
      <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
        <div>
          <h1 className="text-2xl font-semibold">{copy.modelsTitle}</h1>
          <p className="text-muted-foreground mt-1 max-w-3xl text-sm">
            {copy.modelsDescription}
          </p>
        </div>
        <Button variant="outline" onClick={refresh} disabled={loading}>
          <RefreshCwIcon className="size-4" /> {copy.refresh}
        </Button>
      </div>
      <StorageUsageCard usage={usage} />
      <Input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={copy.search}
        className="max-w-lg"
      />
      {error && (
        <div className="border-destructive/40 bg-destructive/5 text-destructive rounded-lg border p-4 text-sm">
          {error}
        </div>
      )}
      {loading ? (
        <p className="text-muted-foreground py-8 text-center">{copy.loading}</p>
      ) : filtered.length === 0 ? (
        <p className="text-muted-foreground py-8 text-center">{copy.empty}</p>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {filtered.map((model) => {
            const external =
              model.validation_scope === "independent_external_validation";
            return (
              <Card key={model.id} data-testid="nir-model-card">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <CardTitle className="truncate text-base">
                        {model.model_id}
                      </CardTitle>
                      <CardDescription className="mt-1 font-mono text-xs">
                        {model.version}
                      </CardDescription>
                    </div>
                    <Badge variant="outline">
                      {copy.statuses[model.status] ?? model.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                    <span className="text-muted-foreground">{copy.method}</span>
                    <span>{model.method ?? "—"}</span>
                    <span className="text-muted-foreground">{copy.size}</span>
                    <span>{formatBytes(model.artifact_size_bytes)}</span>
                    <span className="text-muted-foreground">{copy.hash}</span>
                    <span className="font-mono text-xs">
                      {shortHash(model.artifact_sha256)}
                    </span>
                    <span className="text-muted-foreground">
                      {copy.created}
                    </span>
                    <span>{dateText(model.created_at)}</span>
                  </div>
                  <div className="bg-muted/50 rounded-md p-3">
                    <p className="flex items-center gap-2 font-medium">
                      <ShieldCheckIcon className="size-4" />
                      {copy.validation}
                    </p>
                    <p className="text-muted-foreground mt-1 text-xs">
                      {external
                        ? copy.validationExternal
                        : copy.validationInternal}
                    </p>
                    <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">
                      {copy.productionNotApproved}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {model.status === "ready" && (
                      <Button size="sm" onClick={() => open("attach", model)}>
                        <LinkIcon className="size-4" />
                        {copy.attach}
                      </Button>
                    )}
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => open("details", model)}
                    >
                      <BoxesIcon className="size-4" />
                      {copy.details}
                    </Button>
                    {model.status === "ready" && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() =>
                          action(() =>
                            archiveModel(model.model_id, model.version),
                          )
                        }
                      >
                        <ArchiveIcon className="size-4" />
                        {copy.archive}
                      </Button>
                    )}
                    {model.status === "archived" && (
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => open("delete", model)}
                      >
                        <Trash2Icon className="size-4" />
                        {copy.delete}
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}

      <Dialog
        open={dialog === "attach"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.attach}</DialogTitle>
            <DialogDescription>
              {selected ? `${selected.model_id} · ${selected.version}` : ""}
            </DialogDescription>
          </DialogHeader>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">{copy.targetThread}</span>
            <Input
              value={targetThread}
              onChange={(event) => setTargetThread(event.target.value)}
            />
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              disabled={busy || !selected || !targetThread.trim()}
              onClick={() =>
                action(() =>
                  attachModel(
                    selected!.model_id,
                    selected!.version,
                    targetThread.trim(),
                  ),
                )
              }
            >
              {copy.attach}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={dialog === "delete"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.permanentDelete}</DialogTitle>
            <DialogDescription>{copy.deleteWarning}</DialogDescription>
          </DialogHeader>
          <p className="text-muted-foreground text-sm">
            {copy.deleteCopiesWarning}
          </p>
          <label className="grid gap-1.5 text-sm">
            <span className="font-medium">
              {copy.typeExact}:{" "}
              {selected ? `${selected.model_id}:${selected.version}` : ""}
            </span>
            <Input
              value={deleteConfirmation}
              onChange={(event) => setDeleteConfirmation(event.target.value)}
            />
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              variant="destructive"
              disabled={
                busy ||
                !selected ||
                deleteConfirmation !==
                  `${selected.model_id}:${selected.version}`
              }
              onClick={() =>
                action(() =>
                  deleteModel(
                    selected!.model_id,
                    selected!.version,
                    deleteConfirmation,
                  ),
                )
              }
            >
              {copy.delete}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={dialog === "details"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent className="sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>{selected?.model_id}</DialogTitle>
            <DialogDescription>{selected?.version}</DialogDescription>
          </DialogHeader>
          {selected && (
            <div className="space-y-4 text-sm">
              <section>
                <h3 className="font-medium">{copy.metrics}</h3>
                <pre className="bg-muted mt-2 max-h-48 overflow-auto rounded-md p-3 text-xs">
                  {JSON.stringify(selected.metrics_summary, null, 2)}
                </pre>
              </section>
              <section>
                <h3 className="font-medium">{copy.source}</h3>
                <div className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md border p-3">
                  <span className="text-muted-foreground">dataset</span>
                  <span className="font-mono text-xs">
                    {selected.source_dataset_id ?? "—"}
                  </span>
                  <span className="text-muted-foreground">profile</span>
                  <span className="font-mono text-xs">
                    {selected.source_profile_id ?? "—"}
                  </span>
                  <span className="text-muted-foreground">thread</span>
                  <span className="font-mono text-xs">
                    {selected.source_thread_id}
                  </span>
                  <span className="text-muted-foreground">run</span>
                  <span className="font-mono text-xs">
                    {selected.source_run_id ?? "—"}
                  </span>
                  <span className="text-muted-foreground">attempt</span>
                  <span>{selected.source_attempt}</span>
                </div>
              </section>
              <section>
                <h3 className="font-medium">{copy.validation}</h3>
                <p className="text-muted-foreground mt-1">
                  {selected.validation_scope}
                </p>
                <p className="mt-1 text-xs text-amber-700 dark:text-amber-300">
                  {copy.productionNotApproved}
                </p>
              </section>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
