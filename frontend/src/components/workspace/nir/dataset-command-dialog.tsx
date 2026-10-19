"use client";

import { DatabaseIcon, LinkIcon, RefreshCwIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/core/i18n/hooks";
import { listDatasets, type NIRDataset } from "@/core/nir-library";
import { formatUploadSize } from "@/core/uploads";

export function DatasetCommandDialog({
  open,
  onOpenChange,
  onUse,
  canUse,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUse: (dataset: NIRDataset) => Promise<void>;
  canUse: boolean;
}) {
  const { t } = useI18n();
  const [datasets, setDatasets] = useState<NIRDataset[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    listDatasets(controller.signal)
      .then((rows) => {
        if (!controller.signal.aborted) setDatasets(rows);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : String(reason));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, refreshKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return datasets;
    return datasets.filter((dataset) =>
      [dataset.name, dataset.id, dataset.original_filename ?? ""].some(
        (value) => value.toLowerCase().includes(needle),
      ),
    );
  }, [datasets, query]);

  async function select(dataset: NIRDataset) {
    setBusyId(dataset.id);
    try {
      await onUse(dataset);
      onOpenChange(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <DatabaseIcon className="size-5" aria-hidden="true" />
            {t.nirLibrary.datasetsTitle}
          </DialogTitle>
          <DialogDescription>
            {t.inputBox.datasetPickerDescription}
          </DialogDescription>
        </DialogHeader>
        <div className="flex gap-2">
          <Input
            aria-label={t.nirLibrary.search}
            placeholder={t.nirLibrary.search}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <Button
            aria-label={t.nirLibrary.refresh}
            size="icon"
            type="button"
            variant="outline"
            onClick={() => setRefreshKey((value) => value + 1)}
          >
            <RefreshCwIcon className="size-4" />
          </Button>
        </div>
        {error && (
          <p role="alert" className="text-destructive text-sm">
            {error}
          </p>
        )}
        <div className="max-h-80 space-y-2 overflow-y-auto" aria-live="polite">
          {loading ? (
            <p className="text-muted-foreground py-6 text-center text-sm">
              {t.nirLibrary.loading}
            </p>
          ) : filtered.length === 0 ? (
            <p className="text-muted-foreground py-6 text-center text-sm">
              {t.nirLibrary.empty}
            </p>
          ) : (
            filtered.map((dataset) => (
              <div
                key={dataset.id}
                className="flex min-w-0 items-center gap-3 rounded-lg border p-3"
              >
                <div className="min-w-0 flex-1">
                  <p
                    className="truncate text-sm font-semibold"
                    title={dataset.name}
                  >
                    {dataset.name}
                  </p>
                  <p className="text-muted-foreground truncate text-xs">
                    {dataset.original_filename ?? dataset.id} ·{" "}
                    {formatUploadSize(dataset.size_bytes)}
                  </p>
                  {dataset.status !== "ready" && (
                    <p className="text-muted-foreground text-xs">
                      {t.nirLibrary.status}: {dataset.status}
                    </p>
                  )}
                </div>
                <Button
                  disabled={
                    !canUse || dataset.status !== "ready" || busyId !== null
                  }
                  size="sm"
                  type="button"
                  onClick={() => void select(dataset)}
                >
                  <LinkIcon className="size-4" aria-hidden="true" />
                  {t.nirLibrary.attach}
                </Button>
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
