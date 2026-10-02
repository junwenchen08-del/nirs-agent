import { HardDriveIcon } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { useI18n } from "@/core/i18n/hooks";
import type { NIRStorageUsage } from "@/core/nir-library";

export function formatBytes(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(
    Math.floor(Math.log(value) / Math.log(1024)),
    units.length - 1,
  );
  return `${(value / 1024 ** exponent).toFixed(exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

export function StorageUsageCard({ usage }: { usage: NIRStorageUsage | null }) {
  const { t } = useI18n();
  const copy = t.nirLibrary;
  if (!usage) return null;
  const percentage = Math.min(
    100,
    (usage.accounted_total_bytes /
      Math.max(1, usage.library_write_admission_bytes)) *
      100,
  );
  const rows = [
    [copy.datasetsStorage, usage.dataset_bytes],
    [copy.modelsStorage, usage.model_bytes],
    [copy.threadStorage, usage.thread_bytes],
    [copy.accountedTotal, usage.accounted_total_bytes],
    [copy.admissionLimit, usage.library_write_admission_bytes],
    [copy.diskFree, usage.disk_free_bytes],
  ] as const;
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <HardDriveIcon className="size-4" />
          {copy.storageTitle}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Progress value={percentage} />
        <div className="grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-3">
          {rows.map(([label, value]) => (
            <div
              key={label}
              className="flex justify-between gap-3 rounded-md border px-3 py-2"
            >
              <span className="text-muted-foreground">{label}</span>
              <span className="font-medium tabular-nums">
                {formatBytes(value)}
              </span>
            </div>
          ))}
        </div>
        <p className="text-muted-foreground text-xs">{copy.notStrictQuota}</p>
      </CardContent>
    </Card>
  );
}
