"use client";

import {
  BarChart3Icon,
  DatabaseIcon,
  LinkIcon,
  PencilIcon,
  UploadIcon,
  RefreshCwIcon,
  Trash2Icon,
} from "lucide-react";
import { useRouter } from "next/navigation";
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
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/core/i18n/hooks";
import {
  confirmDatasetProfile,
  createDatasetProfile,
  deleteDataset,
  getDataset,
  getDatasetPreview,
  listDatasets,
  listDatasetUses,
  renameDataset,
  uploadDataset,
  useNirLibraryEnabled,
  type NIRDataset,
  type NIRDatasetDetail,
  type NIRDatasetUse,
} from "@/core/nir-library";
import type {
  DatasetAnalysisType,
  NIRDatasetPreview,
} from "@/core/nir-library/types";

import { DatasetParsingSettings } from "./dataset-parsing-settings";
import { DatasetPreview } from "./dataset-preview";
import { formatBytes } from "./storage-usage-card";

type DialogKind = "upload" | "rename" | "delete" | "details" | null;

function dateText(value: string | null, locale: string) {
  return value ? new Date(value).toLocaleString(locale) : "—";
}

export function DatasetLibrary() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const text = (cn: string, en: string) => (locale.startsWith("zh") ? cn : en);
  const copy = t.nirLibrary;
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [datasets, setDatasets] = useState<NIRDataset[]>([]);
  const [previews, setPreviews] = useState<Record<string, NIRDatasetPreview>>(
    {},
  );
  const [previewErrors, setPreviewErrors] = useState<Record<string, string>>(
    {},
  );
  const [previewPending, setPreviewPending] = useState(false);
  const [analysisType, setAnalysisType] = useState<DatasetAnalysisType | "all">(
    "all",
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [dialog, setDialog] = useState<DialogKind>(null);
  const [selected, setSelected] = useState<NIRDataset | null>(null);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
  const [detail, setDetail] = useState<NIRDatasetDetail | null>(null);
  const [detailError, setDetailError] = useState(false);
  const [uses, setUses] = useState<NIRDatasetUse[]>([]);
  const [taskType, setTaskType] = useState("calibration");
  const [schemaStatus, setSchemaStatus] = useState("confirmed_mapping");
  const [mappingJson, setMappingJson] = useState("{}");
  const [parsingExpanded, setParsingExpanded] = useState(false);

  const refresh = useCallback(() => setRefreshKey((value) => value + 1), []);

  useEffect(() => {
    if (featureLoading) return;
    if (!enabled) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    listDatasets(controller.signal)
      .then((datasetRows) => {
        if (controller.signal.aborted) return;
        setDatasets(datasetRows);
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
        if (controller.signal.aborted) return;
        setDetail(datasetDetail);
        setUses(history);
        setDetailError(false);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setDetailError(true);
          toast.error(
            reason instanceof Error ? reason.message : String(reason),
          );
        }
      });
    return () => controller.abort();
  }, [dialog, selected, refreshKey]);

  useEffect(() => {
    if (!enabled || loading) return;
    const controller = new AbortController();
    setPreviews({});
    setPreviewErrors({});
    setPreviewPending(true);
    const queue = datasets.filter((dataset) =>
      ["ready", "archived"].includes(dataset.status),
    );
    let next = 0;
    async function worker() {
      while (!controller.signal.aborted && next < queue.length) {
        const dataset = queue[next++]!;
        try {
          const preview = await getDatasetPreview(
            dataset.id,
            controller.signal,
          );
          if (
            preview.dataset_id !== dataset.id ||
            preview.source_sha256 !== dataset.sha256
          )
            throw new Error("Dataset preview does not match its source");
          if (!controller.signal.aborted)
            setPreviews((current) => ({ ...current, [dataset.id]: preview }));
        } catch {
          if (!controller.signal.aborted)
            setPreviewErrors((current) => ({
              ...current,
              [dataset.id]: locale.startsWith("zh")
                ? "暂时无法读取数据预览，请刷新重试。"
                : "Preview unavailable. Refresh to retry.",
            }));
        }
      }
    }
    void Promise.all([worker(), worker(), worker()]).finally(() => {
      if (!controller.signal.aborted) setPreviewPending(false);
    });
    return () => controller.abort();
  }, [datasets, enabled, loading, locale]);

  const groupLabels = {
    all: text("全部数据集", "All datasets"),
    regression: text("定量分析", "Quantitative analysis"),
    classification: text("分类识别", "Classification"),
    unknown: text("待确认", "Unconfirmed"),
  };

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return datasets.filter(
      (dataset) =>
        (analysisType === "all" ||
          (previews[dataset.id]?.analysis_type ?? "unknown") ===
            analysisType) &&
        [dataset.name, dataset.id, dataset.original_filename ?? ""].some(
          (value) => value.toLowerCase().includes(needle),
        ),
    );
  }, [datasets, query, previews, analysisType]);

  function open(
    kind: DialogKind,
    dataset: NIRDataset | null = null,
    expandParsing = false,
  ) {
    setSelected(dataset);
    setDialog(kind);
    setName(dataset?.name ?? "");
    setUploadFile(null);
    setUploadError(null);
    setDeleteConfirmation("");
    setDetail(null);
    setDetailError(false);
    setUses([]);
    setParsingExpanded(
      expandParsing ||
        (!!dataset && previews[dataset.id]?.status === "needs_mapping"),
    );
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
            {text(
              "查看数据特征，按分析用途查找数据集，一键带入新对话。",
              "Explore your data, filter by analysis type and open a new chat with a dataset.",
            )}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={refresh} disabled={loading}>
            <RefreshCwIcon className="size-4" /> {copy.refresh}
          </Button>
          <Button onClick={() => open("upload")}>
            <UploadIcon className="size-4" />{" "}
            {text("上传数据集", "Upload dataset")}
          </Button>
        </div>
      </div>
      <div
        className="grid grid-cols-2 gap-3 sm:grid-cols-4"
        data-testid="dataset-summary"
      >
        {[
          [text("数据集数量", "Datasets"), datasets.length],
          [
            text("可使用数据集", "Available datasets"),
            datasets.filter((dataset) => dataset.status === "ready").length,
          ],
          [
            text("文件总大小", "Total file size"),
            formatBytes(
              datasets.reduce((sum, dataset) => sum + dataset.size_bytes, 0),
            ),
          ],
          [text("支持的文件格式", "Supported formats"), "CSV · TXT · MAT"],
        ].map(([label, value]) => (
          <div key={label} className="bg-card rounded-xl border p-4">
            <p className="text-muted-foreground text-xs">{label}</p>
            <p className="mt-2 text-xl font-semibold">
              {loading ? "—" : value}
            </p>
          </div>
        ))}
      </div>
      <Tabs
        value={analysisType}
        onValueChange={(value) =>
          setAnalysisType(value as DatasetAnalysisType | "all")
        }
      >
        <TabsList className="h-auto flex-wrap">
          {(["all", "regression", "classification", "unknown"] as const).map(
            (group) => (
              <TabsTrigger key={group} value={group}>
                {groupLabels[group]}
                <span className="ml-2 text-xs opacity-60">
                  {group === "all"
                    ? datasets.length
                    : datasets.filter(
                        (dataset) =>
                          (previews[dataset.id]?.analysis_type ?? "unknown") ===
                          group,
                      ).length}
                </span>
              </TabsTrigger>
            ),
          )}
        </TabsList>
      </Tabs>
      <Input
        aria-label={text("搜索数据集", "Search datasets")}
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder={text(
          "搜索数据集名称或文件名",
          "Search dataset or file names",
        )}
        className="max-w-lg"
      />
      <p className="text-muted-foreground text-xs">
        {previewPending
          ? text(
              "正在读取数据特征并识别分析类型…",
              "Reading data features and identifying analysis types…",
            )
          : text(
              "定量分析用于预测含量等数值，分类识别用于区分品种或类别；分析用途不明确时列为待确认。",
              "Quantitative analysis predicts numbers; classification distinguishes categories. Files without a definite target remain unconfirmed.",
            )}
      </p>
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
            <Card
              key={dataset.id}
              data-testid="nir-dataset-card"
              className="cursor-pointer transition-shadow hover:shadow-md"
              onClick={() =>
                router.push(
                  `/workspace/nir/datasets/${encodeURIComponent(dataset.id)}`,
                )
              }
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <CardTitle className="truncate text-base">
                      <button
                        type="button"
                        className="text-left focus-visible:underline"
                        onClick={() =>
                          router.push(
                            `/workspace/nir/datasets/${encodeURIComponent(dataset.id)}`,
                          )
                        }
                      >
                        {dataset.name}
                      </button>
                    </CardTitle>
                    <CardDescription className="mt-1 font-mono text-xs">
                      {
                        groupLabels[
                          previews[dataset.id]?.analysis_type ?? "unknown"
                        ]
                      }
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
                  <span className="text-muted-foreground">{copy.created}</span>
                  <span>{dateText(dataset.created_at, locale)}</span>
                </div>
                {previews[dataset.id]?.status === "ready" && (
                  <p className="text-muted-foreground text-xs">
                    {text("样本", "Samples")}：
                    {previews[dataset.id]?.sample_count?.toLocaleString() ??
                      "—"}{" "}
                    · {text("光谱变量", "Spectral variables")}：
                    {previews[dataset.id]?.feature_count?.toLocaleString() ??
                      "—"}
                  </p>
                )}
                <div
                  className="flex flex-wrap gap-2 pt-1"
                  onClick={(event) => event.stopPropagation()}
                >
                  {dataset.status === "ready" && (
                    <Button
                      size="sm"
                      onClick={() =>
                        router.push(
                          `/workspace/chats/new?dataset=${encodeURIComponent(dataset.id)}`,
                        )
                      }
                    >
                      <LinkIcon className="size-4" />
                      {text("挂载到对话", "Attach to chat")}
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() =>
                      router.push(
                        `/workspace/nir/datasets/${encodeURIComponent(dataset.id)}`,
                      )
                    }
                  >
                    <BarChart3Icon className="size-4" />
                    {text("查看详情", "View details")}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => open("details", dataset)}
                  >
                    {text("数据概览", "Data overview")}
                  </Button>
                  {previews[dataset.id]?.status === "needs_mapping" && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => open("details", dataset, true)}
                    >
                      {text("解析设置", "Parsing settings")}
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => open("rename", dataset)}
                  >
                    <PencilIcon className="size-4" />
                    {copy.rename}
                  </Button>
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
        open={dialog === "upload"}
        onOpenChange={(value) => !value && !busy && setDialog(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{text("上传数据集", "Upload dataset")}</DialogTitle>
            <DialogDescription>
              {text(
                "从电脑选择 CSV、TXT 或 MAT 文件。上传后可查看数据，也可点击“挂载到对话”或在对话中输入 /datasets 调用。",
                "Choose a CSV, TXT or MAT file. After uploading, view the data or use Attach to chat or /datasets in a chat.",
              )}
            </DialogDescription>
          </DialogHeader>
          <FormField label={text("选择本地文件", "Choose a local file")}>
            <Input
              type="file"
              accept=".csv,.txt,.mat"
              disabled={busy}
              onChange={(event) => {
                const file = event.target.files?.[0] ?? null;
                setUploadError(null);
                if (
                  file &&
                  (!/\.(csv|txt|mat)$/i.test(file.name) || file.size === 0)
                ) {
                  setUploadFile(null);
                  setUploadError(
                    text(
                      "请选择包含数据的 CSV、TXT 或 MAT 文件。",
                      "Choose a nonempty CSV, TXT or MAT file.",
                    ),
                  );
                  return;
                }
                setUploadFile(file);
                setName(file?.name ?? "");
              }}
            />
          </FormField>
          {uploadFile && (
            <p className="text-muted-foreground text-sm break-all">
              {uploadFile.name} · {formatBytes(uploadFile.size)}
            </p>
          )}
          {uploadError && (
            <p className="text-destructive text-sm" role="alert">
              {uploadError}
            </p>
          )}
          <FormField label={copy.name}>
            <Input
              value={name}
              maxLength={256}
              disabled={busy}
              onChange={(event) => setName(event.target.value)}
            />
          </FormField>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => setDialog(null)}
            >
              {copy.cancel}
            </Button>
            <Button
              disabled={busy || !name.trim() || !uploadFile}
              onClick={() => {
                if (!uploadFile) return;
                void action(async () => {
                  await uploadDataset(uploadFile, name.trim());
                  setAnalysisType("all");
                  setQuery("");
                });
              }}
            >
              {busy
                ? text("正在上传…", "Uploading…")
                : text("上传到数据集库", "Upload to library")}
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
            <DialogDescription>
              {text(
                "查看样本规模、光谱形状和目标值分布。",
                "Explore sample counts, spectra and target distributions.",
              )}
            </DialogDescription>
          </DialogHeader>
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
          <details
            className="space-y-3 rounded-lg border p-4"
            open={parsingExpanded}
            onToggle={(event) => setParsingExpanded(event.currentTarget.open)}
          >
            <summary className="cursor-pointer font-medium">
              {text("解析设置", "Parsing settings")}
            </summary>
            {selected && parsingExpanded && (
              <DatasetParsingSettings
                datasetId={selected.id}
                sourceSha={selected.sha256}
                disabled={busy || selected.status !== "ready"}
                onSaved={refresh}
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
                <p className="text-muted-foreground text-sm">
                  {copy.noProfiles}
                </p>
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
            </details>
          </details>
          <details className="space-y-3 rounded-lg border p-4">
            <summary className="cursor-pointer font-medium">
              {text("使用记录", "Usage history")}
            </summary>
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
                    {text("分析记录", "Run")}: {item.run_id ?? "—"} ·{" "}
                    {text("解析版本", "Parsing profile")}:{" "}
                    {item.profile_id ?? "—"}
                  </p>
                </div>
              ))
            )}
          </details>
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
