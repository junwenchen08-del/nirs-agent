"use client";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/core/i18n/hooks";
import {
  confirmDatasetProfile,
  createDatasetProfile,
  getDataset,
  getDatasetPreview,
  listDatasetUses,
  type NIRDataset,
  type NIRDatasetDetail,
  type NIRDatasetPreview,
  type NIRDatasetUse,
} from "@/core/nir-library";

import { DatasetParsingSettings } from "./dataset-parsing-settings";
import { DatasetPartitions } from "./dataset-partitions";
import { DatasetPreview } from "./dataset-preview";

export function DatasetDetailsPanel({
  dataset: selected,
  section = "all",
  disabled = false,
  onSaved,
  expandParsing = false,
  profileId = null,
}: {
  dataset: NIRDataset;
  section?: "all" | "overview" | "parsing";
  disabled?: boolean;
  onSaved?: () => void;
  expandParsing?: boolean;
  profileId?: string | null;
}) {
  const { t, locale } = useI18n();
  const copy = t.nirLibrary;
  const text = (cn: string, en: string) => (locale.startsWith("zh") ? cn : en);
  const [detail, setDetail] = useState<NIRDatasetDetail | null>(null);
  const [detailError, setDetailError] = useState(false);
  const [previews, setPreviews] = useState<Record<string, NIRDatasetPreview>>(
    {},
  );
  const [previewErrors, setPreviewErrors] = useState<Record<string, string>>(
    {},
  );
  const [uses, setUses] = useState<NIRDatasetUse[]>([]);
  const [usesError, setUsesError] = useState(false);
  const [parsingExpanded, setParsingExpanded] = useState(expandParsing);
  useEffect(() => {
    if (section === "parsing") setParsingExpanded(true);
  }, [section]);
  const [working, setWorking] = useState(false);
  const busy = working || disabled;
  const [revision, setRevision] = useState(0);
  const [taskType, setTaskType] = useState("calibration");
  const [schemaStatus, setSchemaStatus] = useState("confirmed_mapping");
  const [mappingJson, setMappingJson] = useState("{}");
  const groupLabels = {
    classification: text("分类识别", "Classification"),
    regression: text("定量分析", "Quantitative analysis"),
  };
  const refresh = () => setRevision((value) => value + 1);
  useEffect(() => {
    const controller = new AbortController();
    setDetail(null);
    setDetailError(false);
    setPreviews({});
    setPreviewErrors({});
    setUses([]);
    setUsesError(false);
    void getDataset(selected.id, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          if (data.id !== selected.id || data.sha256 !== selected.sha256)
            throw new Error("Dataset identity mismatch");
          setDetail(data);
        }
      })
      .catch(() => {
        if (!controller.signal.aborted) setDetailError(true);
      });
    void getDatasetPreview(selected.id, controller.signal, profileId)
      .then((data) => {
        if (!controller.signal.aborted) {
          if (
            data.dataset_id !== selected.id ||
            data.source_sha256 !== selected.sha256 ||
            (profileId && data.profile_id !== profileId)
          )
            throw new Error("Dataset preview identity mismatch");
          setPreviews({ [selected.id]: data });
        }
      })
      .catch(() => {
        if (!controller.signal.aborted)
          setPreviewErrors({
            [selected.id]: text(
              "数据预览读取失败，请重试。",
              "Preview read failed. Retry.",
            ),
          });
      });
    void listDatasetUses(selected.id, controller.signal)
      .then((data) => {
        if (!Array.isArray(data))
          throw new Error("Usage history response is invalid");
        if (!controller.signal.aborted) setUses(data);
      })
      .catch(() => {
        if (!controller.signal.aborted) setUsesError(true);
      });
    return () => controller.abort();
    // Locale does not change the resource identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected.id, selected.sha256, revision, profileId]);
  async function action(work: () => Promise<unknown>, _close = false) {
    if (disabled) return;
    setWorking(true);
    try {
      await work();
      toast.success(copy.operationSuccess);
      refresh();
      onSaved?.();
    } catch (reason) {
      toast.error(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWorking(false);
    }
  }
  return (
    <div className="space-y-4">
      {disabled && section === "parsing" && (
        <p role="status" className="text-sm">
          {text(
            "请先保存样本用途与波段修改，再更改解析设置。",
            "Save sample roles and wavelength changes before editing parsing settings.",
          )}
        </p>
      )}
      <div hidden={section === "parsing"}>
        {selected && previews[selected.id] ? (
          <DatasetPreview preview={previews[selected.id]!} />
        ) : (
          <div className="space-y-2">
            <p className="text-muted-foreground text-sm">
              {selected && previewErrors[selected.id]
                ? previewErrors[selected.id]
                : copy.loading}
            </p>
            {selected && previewErrors[selected.id] && (
              <Button variant="outline" onClick={refresh}>
                {copy.refresh}
              </Button>
            )}
          </div>
        )}
      </div>
      {section !== "parsing" && detail && (
        <DatasetPartitions
          datasetId={selected.id}
          profiles={detail.profiles}
          zh={locale.startsWith("zh")}
        />
      )}
      <details
        hidden={section === "overview" || !!profileId}
        className="space-y-3 rounded-lg border p-4"
        open={section === "parsing" || parsingExpanded}
        onToggle={(event) => setParsingExpanded(event.currentTarget.open)}
      >
        <summary className="cursor-pointer font-medium">
          {text("解析设置", "Parsing settings")}
        </summary>
        {selected && (section === "parsing" || parsingExpanded) && (
          <DatasetParsingSettings
            datasetId={selected.id}
            sourceSha={selected.sha256}
            disabled={busy || selected.status !== "ready"}
            onSaved={() => {
              refresh();
              onSaved?.();
            }}
          />
        )}
        <details className="space-y-3 rounded-lg border p-3">
          <summary className="cursor-pointer text-sm font-medium">
            {text("高级设置与解析版本", "Advanced settings and versions")}
          </summary>
          {detailError ? (
            <div className="space-y-2 text-sm">
              <p>
                {text(
                  "解析设置读取失败，请重试。",
                  "Parsing settings could not be loaded. Retry.",
                )}
              </p>
              <Button variant="outline" onClick={refresh}>
                {copy.refresh}
              </Button>
            </div>
          ) : !detail ? (
            <p className="text-muted-foreground text-sm">{copy.loading}</p>
          ) : detail.profiles.length === 0 ? (
            <p className="text-muted-foreground text-sm">{copy.noProfiles}</p>
          ) : (
            detail.profiles
              .filter((item) => item.profile_status !== "split")
              .map((item) => (
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
                    {item.task_type === "classification"
                      ? groupLabels.classification
                      : item.task_type === "calibration"
                        ? groupLabels.regression
                        : text("查看光谱", "Inspect spectra")}{" "}
                    ·{" "}
                    {item.schema_status === "needs_user_mapping"
                      ? text("需要确认列对应关系", "Column mapping needed")
                      : text("已有解析设置", "Parsing settings available")}
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
                  <option value="calibration">
                    {text("定量分析", "Quantitative analysis")}
                  </option>
                  <option value="classification">
                    {text("分类识别", "Classification")}
                  </option>
                  <option value="inspection">
                    {text("仅查看光谱", "Inspect spectra only")}
                  </option>
                </select>
              </FormField>
              <FormField label={copy.schemaStatus}>
                <select
                  className="border-input bg-background h-9 rounded-md border px-3 text-sm"
                  value={schemaStatus}
                  onChange={(event) => setSchemaStatus(event.target.value)}
                >
                  <option value="confirmed_mapping">
                    {text("已确认列对应关系", "Confirmed column mapping")}
                  </option>
                  <option value="auto">
                    {text("自动识别", "Automatic detection")}
                  </option>
                  <option value="legacy_auto_layout">
                    {text("旧格式自动识别", "Legacy format detection")}
                  </option>
                  <option value="needs_user_mapping">
                    {text("需要确认列对应关系", "Column mapping needed")}
                  </option>
                  <option value="spectra_only">
                    {text("只包含光谱数据", "Spectra only")}
                  </option>
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
                    createDatasetProfile(selected.id, {
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
        </details>
      </details>
      <details
        hidden={section === "parsing"}
        className="space-y-3 rounded-lg border p-4"
      >
        <summary className="cursor-pointer font-medium">
          {text("使用记录", "Usage history")}
        </summary>
        {usesError ? (
          <p role="alert" className="text-sm">
            {text("使用记录读取失败。", "Usage history read failed.")}
            <button type="button" className="ml-2 underline" onClick={refresh}>
              {copy.refresh}
            </button>
          </p>
        ) : uses.length === 0 ? (
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
                {text("分析记录", "Run")}: {item.run_id ?? "—"} ·{" "}
                {text("解析版本", "Parsing profile")}: {item.profile_id ?? "—"}
              </p>
            </div>
          ))
        )}
      </details>
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
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
    </label>
  );
}
