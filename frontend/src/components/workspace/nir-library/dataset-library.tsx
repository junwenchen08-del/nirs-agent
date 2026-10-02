"use client";

import {
  ArchiveIcon,
  DatabaseIcon,
  HistoryIcon,
  LinkIcon,
  PencilIcon,
  PlusIcon,
  RefreshCwIcon,
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
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/core/i18n/hooks";
import {
  archiveDataset,
  attachDataset,
  confirmDatasetProfile,
  createDatasetProfile,
  deleteDataset,
  getDataset,
  getStorageUsage,
  listDatasets,
  listDatasetUses,
  renameDataset,
  saveDataset,
  useNirLibraryEnabled,
  type NIRDataset,
  type NIRDatasetDetail,
  type NIRDatasetUse,
  type NIRStorageUsage,
} from "@/core/nir-library";

import { formatBytes, StorageUsageCard } from "./storage-usage-card";

type DialogKind = "save" | "attach" | "rename" | "delete" | "details" | null;

function shortHash(value: string | null) {
  return value ? `${value.slice(0, 10)}…${value.slice(-6)}` : "—";
}

function dateText(value: string | null) {
  return value ? new Date(value).toLocaleString() : "—";
}

export function DatasetLibrary() {
  const { t } = useI18n();
  const copy = t.nirLibrary;
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [datasets, setDatasets] = useState<NIRDataset[]>([]);
  const [usage, setUsage] = useState<NIRStorageUsage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [dialog, setDialog] = useState<DialogKind>(null);
  const [selected, setSelected] = useState<NIRDataset | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [threadId, setThreadId] = useState("");
  const [sourcePath, setSourcePath] = useState("/mnt/user-data/uploads/");
  const [saveConfirmed, setSaveConfirmed] = useState(false);
  const [profileId, setProfileId] = useState("");
  const [desiredFilename, setDesiredFilename] = useState("");
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [detail, setDetail] = useState<NIRDatasetDetail | null>(null);
  const [uses, setUses] = useState<NIRDatasetUse[]>([]);
  const [taskType, setTaskType] = useState("calibration");
  const [schemaStatus, setSchemaStatus] = useState("confirmed_mapping");
  const [mappingJson, setMappingJson] = useState("{}");

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
      listDatasets(controller.signal),
      getStorageUsage(controller.signal),
    ])
      .then(([datasetRows, storage]) => {
        setDatasets(datasetRows);
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

  useEffect(() => {
    if (dialog !== "details" || !selected) return;
    const controller = new AbortController();
    Promise.all([
      getDataset(selected.id, controller.signal),
      listDatasetUses(selected.id, controller.signal),
    ])
      .then(([datasetDetail, history]) => {
        setDetail(datasetDetail);
        setUses(history);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          toast.error(
            reason instanceof Error ? reason.message : String(reason),
          );
      });
    return () => controller.abort();
  }, [dialog, selected, refreshKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return datasets;
    return datasets.filter((dataset) =>
      [dataset.name, dataset.id, dataset.original_filename ?? ""].some(
        (value) => value.toLowerCase().includes(needle),
      ),
    );
  }, [datasets, query]);

  function open(kind: DialogKind, dataset: NIRDataset | null = null) {
    setSelected(dataset);
    setDialog(kind);
    setName(dataset?.name ?? "");
    setThreadId("");
    setProfileId("");
    setDesiredFilename("");
    setDeleteConfirmation("");
    setDetail(null);
    setUses([]);
  }

  async function action(work: () => Promise<unknown>, close = true) {
    setBusy(true);
    try {
      await work();
      toast.success(copy.operationSuccess);
      if (close) setDialog(null);
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
          <h1 className="text-2xl font-semibold">{copy.datasetsTitle}</h1>
          <p className="text-muted-foreground mt-1 max-w-3xl text-sm">
            {copy.datasetsDescription}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={refresh} disabled={loading}>
            <RefreshCwIcon className="size-4" /> {copy.refresh}
          </Button>
          <Button onClick={() => open("save")}>
            <PlusIcon className="size-4" /> {copy.saveDataset}
          </Button>
        </div>
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
          {filtered.map((dataset) => (
            <Card key={dataset.id} data-testid="nir-dataset-card">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <CardTitle className="truncate text-base">
                      {dataset.name}
                    </CardTitle>
                    <CardDescription className="mt-1 font-mono text-xs">
                      {dataset.id}
                    </CardDescription>
                  </div>
                  <Badge variant="outline">
                    {copy.statuses[dataset.status] ?? dataset.status}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                  <span className="text-muted-foreground">{copy.file}</span>
                  <span className="truncate">
                    {dataset.original_filename ?? "—"}
                  </span>
                  <span className="text-muted-foreground">{copy.size}</span>
                  <span>{formatBytes(dataset.size_bytes)}</span>
                  <span className="text-muted-foreground">{copy.hash}</span>
                  <span className="font-mono text-xs">
                    {shortHash(dataset.sha256)}
                  </span>
                  <span className="text-muted-foreground">{copy.created}</span>
                  <span>{dateText(dataset.created_at)}</span>
                </div>
                <div className="flex flex-wrap gap-2 pt-1">
                  {dataset.status === "ready" && (
                    <Button size="sm" onClick={() => open("attach", dataset)}>
                      <LinkIcon className="size-4" />
                      {copy.attach}
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => open("details", dataset)}
                  >
                    <HistoryIcon className="size-4" />
                    {copy.details}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => open("rename", dataset)}
                  >
                    <PencilIcon className="size-4" />
                    {copy.rename}
                  </Button>
                  {dataset.status === "ready" && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => action(() => archiveDataset(dataset.id))}
                    >
                      <ArchiveIcon className="size-4" />
                      {copy.archive}
                    </Button>
                  )}
                  {dataset.status === "archived" && (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => open("delete", dataset)}
                    >
                      <Trash2Icon className="size-4" />
                      {copy.delete}
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      <Dialog
        open={dialog === "save"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.saveDataset}</DialogTitle>
            <DialogDescription>{copy.datasetsDescription}</DialogDescription>
          </DialogHeader>
          <FormField label={copy.name}>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </FormField>
          <FormField label={copy.sourceThread}>
            <Input
              value={threadId}
              onChange={(event) => setThreadId(event.target.value)}
            />
          </FormField>
          <FormField label={copy.sourcePath}>
            <Input
              value={sourcePath}
              onChange={(event) => setSourcePath(event.target.value)}
            />
          </FormField>
          <label className="flex items-start gap-2 text-sm">
            <input
              className="mt-1"
              type="checkbox"
              checked={saveConfirmed}
              onChange={(event) => setSaveConfirmed(event.target.checked)}
            />
            <span>{copy.saveConfirmation}</span>
          </label>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              disabled={
                busy ||
                !name.trim() ||
                !threadId.trim() ||
                !sourcePath.trim() ||
                !saveConfirmed
              }
              onClick={() =>
                action(() =>
                  saveDataset({
                    name: name.trim(),
                    threadId: threadId.trim(),
                    virtualPath: sourcePath.trim(),
                  }),
                )
              }
            >
              {copy.save}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={dialog === "rename"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.rename}</DialogTitle>
          </DialogHeader>
          <FormField label={copy.name}>
            <Input
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </FormField>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              disabled={busy || !selected || !name.trim()}
              onClick={() =>
                action(() => renameDataset(selected!.id, name.trim()))
              }
            >
              {copy.save}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={dialog === "attach"}
        onOpenChange={(value) => !value && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{copy.attach}</DialogTitle>
            <DialogDescription>{selected?.name}</DialogDescription>
          </DialogHeader>
          <FormField label={copy.targetThread}>
            <Input
              value={threadId}
              onChange={(event) => setThreadId(event.target.value)}
            />
          </FormField>
          <FormField label={copy.profile}>
            <Input
              value={profileId}
              onChange={(event) => setProfileId(event.target.value)}
              placeholder="optional profile ID"
            />
          </FormField>
          <FormField label={copy.file}>
            <Input
              value={desiredFilename}
              onChange={(event) => setDesiredFilename(event.target.value)}
              placeholder="optional filename"
            />
          </FormField>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              disabled={busy || !selected || !threadId.trim()}
              onClick={() =>
                action(() =>
                  attachDataset(selected!.id, {
                    threadId: threadId.trim(),
                    profileId: profileId.trim() || undefined,
                    desiredFilename: desiredFilename.trim() || undefined,
                  }),
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
          <FormField label={`${copy.typeExact}: ${selected?.id ?? ""}`}>
            <Input
              value={deleteConfirmation}
              onChange={(event) => setDeleteConfirmation(event.target.value)}
            />
          </FormField>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDialog(null)}>
              {copy.cancel}
            </Button>
            <Button
              variant="destructive"
              disabled={busy || deleteConfirmation !== selected?.id}
              onClick={() => {
                if (selected) {
                  void action(() =>
                    deleteDataset(selected.id, deleteConfirmation),
                  );
                }
              }}
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
        <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle>{selected?.name}</DialogTitle>
            <DialogDescription>{selected?.id}</DialogDescription>
          </DialogHeader>
          <section className="space-y-3">
            <h3 className="font-medium">{copy.profiles}</h3>
            {!detail ? (
              <p className="text-muted-foreground text-sm">{copy.loading}</p>
            ) : detail.profiles.length === 0 ? (
              <p className="text-muted-foreground text-sm">{copy.noProfiles}</p>
            ) : (
              detail.profiles.map((item) => (
                <div key={item.id} className="rounded-lg border p-3 text-sm">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono text-xs">
                      v{item.profile_version} · {item.id}
                    </span>
                    <Badge variant="outline">
                      {copy.statuses[item.profile_status] ??
                        item.profile_status}
                    </Badge>
                  </div>
                  <p className="text-muted-foreground mt-1">
                    {item.task_type ?? "—"} · {item.schema_status}
                  </p>
                  {item.profile_status === "draft" &&
                    item.schema_status !== "needs_user_mapping" && (
                      <Button
                        className="mt-2"
                        size="sm"
                        variant="outline"
                        disabled={busy}
                        onClick={() =>
                          action(
                            () =>
                              confirmDatasetProfile(item.dataset_id, item.id),
                            false,
                          )
                        }
                      >
                        {copy.confirmProfile}
                      </Button>
                    )}
                </div>
              ))
            )}
            <div className="space-y-2 rounded-lg border p-3">
              <h4 className="text-sm font-medium">{copy.createProfile}</h4>
              <div className="grid gap-2 sm:grid-cols-2">
                <FormField label={copy.taskType}>
                  <select
                    className="border-input bg-background h-9 rounded-md border px-3 text-sm"
                    value={taskType}
                    onChange={(event) => setTaskType(event.target.value)}
                  >
                    <option value="calibration">calibration</option>
                    <option value="classification">classification</option>
                    <option value="inspection">inspection</option>
                  </select>
                </FormField>
                <FormField label={copy.schemaStatus}>
                  <select
                    className="border-input bg-background h-9 rounded-md border px-3 text-sm"
                    value={schemaStatus}
                    onChange={(event) => setSchemaStatus(event.target.value)}
                  >
                    <option value="confirmed_mapping">confirmed_mapping</option>
                    <option value="auto">auto</option>
                    <option value="legacy_auto_layout">
                      legacy_auto_layout
                    </option>
                    <option value="needs_user_mapping">
                      needs_user_mapping
                    </option>
                    <option value="spectra_only">spectra_only</option>
                  </select>
                </FormField>
              </div>
              <FormField label={copy.mappingJson}>
                <Textarea
                  value={mappingJson}
                  onChange={(event) => setMappingJson(event.target.value)}
                  rows={5}
                  className="font-mono text-xs"
                />
              </FormField>
              <Button
                size="sm"
                disabled={busy || !selected}
                onClick={() => {
                  let mapping: unknown;
                  try {
                    mapping = JSON.parse(mappingJson);
                  } catch {
                    toast.error(copy.invalidJson);
                    return;
                  }
                  if (
                    !mapping ||
                    typeof mapping !== "object" ||
                    Array.isArray(mapping)
                  ) {
                    toast.error(copy.invalidJson);
                    return;
                  }
                  void action(
                    () =>
                      createDatasetProfile(selected!.id, {
                        taskType,
                        schemaStatus,
                        mapping: mapping as Record<string, unknown>,
                      }),
                    false,
                  );
                }}
              >
                {copy.createProfile}
              </Button>
            </div>
          </section>
          <section className="space-y-3">
            <h3 className="font-medium">{copy.history}</h3>
            {uses.length === 0 ? (
              <p className="text-muted-foreground text-sm">{copy.noHistory}</p>
            ) : (
              uses.map((item) => (
                <div
                  key={item.attachment_id}
                  className="rounded-lg border p-3 text-sm"
                >
                  <div className="flex justify-between gap-3">
                    <span className="font-mono text-xs">{item.thread_id}</span>
                    <Badge variant="outline">
                      {copy.statuses[item.status] ?? item.status}
                    </Badge>
                  </div>
                  <p className="text-muted-foreground mt-1">
                    run: {item.run_id ?? "—"} · profile:{" "}
                    {item.profile_id ?? "—"}
                  </p>
                </div>
              ))
            )}
          </section>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function FormField({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="grid gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}

export function FeatureDisabled({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <Card className="max-w-lg">
        <CardHeader className="text-center">
          <DatabaseIcon className="text-muted-foreground mx-auto size-8" />
          <CardTitle>{title}</CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
