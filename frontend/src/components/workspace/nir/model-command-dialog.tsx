"use client";

import { BoxesIcon, LinkIcon, RefreshCwIcon } from "lucide-react";
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
import { listModels, type NIRModelVersion } from "@/core/nir-library";

export function ModelCommandDialog({
  open,
  onOpenChange,
  onUse,
  canUse,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUse: (model: NIRModelVersion) => Promise<void>;
  canUse: boolean;
}) {
  const { t } = useI18n();
  const [models, setModels] = useState<NIRModelVersion[]>([]);
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
    listModels(controller.signal)
      .then((rows) => {
        if (!controller.signal.aborted) setModels(rows);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, refreshKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return models.filter((model) =>
      [model.model_id, model.version, model.method ?? ""].some((value) =>
        value.toLowerCase().includes(needle),
      ),
    );
  }, [models, query]);

  async function select(model: NIRModelVersion) {
    setBusyId(model.id);
    setError(null);
    try {
      await onUse(model);
      onOpenChange(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setBusyId(null);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <BoxesIcon className="size-5" aria-hidden="true" />
            {t.nirLibrary.modelsTitle}
          </DialogTitle>
          <DialogDescription>
            {t.inputBox.modelPickerDescription}
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
            filtered.map((model) => (
              <div
                key={model.id}
                data-testid="model-picker-row"
                className="flex min-w-0 items-center gap-3 rounded-lg border p-3"
              >
                <div className="min-w-0 flex-1">
                  <p
                    className="truncate text-sm font-semibold"
                    title={model.model_id}
                  >
                    {model.model_id}
                  </p>
                  <p
                    className="text-muted-foreground truncate text-xs"
                    title={model.version}
                  >
                    {model.version}
                  </p>
                  <p className="text-muted-foreground text-xs">
                    {t.nirLibrary.method}: {model.method?.toUpperCase() ?? "—"}{" "}
                    ·{" "}
                    {t.nirWorkflow.validationScopes[model.validation_scope] ??
                      model.validation_scope}
                  </p>
                  <p className="text-muted-foreground text-xs">
                    {t.nirLibrary.status}:{" "}
                    {t.nirLibrary.statuses[model.status] ?? model.status}
                  </p>
                </div>
                <Button
                  disabled={
                    !canUse || model.status !== "ready" || busyId !== null
                  }
                  size="sm"
                  type="button"
                  onClick={() => void select(model)}
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
