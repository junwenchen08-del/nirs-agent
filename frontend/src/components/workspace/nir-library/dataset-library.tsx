"use client";

import {
  BarChart3Icon,
  DatabaseIcon,
  LinkIcon,
  PencilIcon,
  UploadIcon,
  RefreshCwIcon,
  Trash2Icon,
  MoreHorizontalIcon,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
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
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useI18n } from "@/core/i18n/hooks";
import {
  deleteDataset,
  getDatasetPreview,
  listDatasets,
  renameDataset,
  uploadDataset,
  useNirLibraryEnabled,
  type NIRDataset,
} from "@/core/nir-library";
import { modelLibraryPath } from "@/core/nir-library/model-catalog";
import type {
  DatasetAnalysisType,
  NIRDatasetPreview,
} from "@/core/nir-library/types";
import { useModelCatalog } from "@/core/nir-library/use-model-catalog";

import { DatasetDetailsPanel } from "./dataset-details-panel";
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
  const modelCatalog = useModelCatalog(enabled && !featureLoading);
  const modelCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const model of modelCatalog.data ?? []) {
      if (model.source_dataset_id)
        counts[model.source_dataset_id] =
          (counts[model.source_dataset_id] ?? 0) + 1;
    }
    return counts;
  }, [modelCatalog.data]);
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
  const [parsingExpanded, setParsingExpanded] = useState(false);

  const refresh = () => {
    setRefreshKey((value) => value + 1);
    void modelCatalog.refetch();
  };

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
      <p
        className="text-muted-foreground flex flex-wrap gap-x-5 gap-y-1 text-sm"
        data-testid="dataset-summary"
      >
        <span>
          {text("共", "Total")} {loading ? "—" : datasets.length}{" "}
          {text("个数据集", "datasets")}
        </span>
        <span>
          {text("文件总大小", "Total file size")}{" "}
          {formatBytes(datasets.reduce((sum, row) => sum + row.size_bytes, 0))}
        </span>
      </p>
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
        <div className="dataset-resource-grid">
          {filtered.map((dataset) => (
            <Card
              key={dataset.id}
              data-testid="nir-dataset-card"
              className="hover:border-primary/40 gap-3 transition-colors"
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <CardTitle className="min-w-0 text-base">
                      <button
                        type="button"
                        className="line-clamp-2 text-left break-words hover:underline"
                        title={dataset.name}
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
                  {dataset.status !== "ready" && (
                    <Badge variant="outline">
                      {copy.statuses[dataset.status] ?? dataset.status}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <Link
                  className="text-primary inline-block text-sm underline underline-offset-4"
                  href={modelLibraryPath(dataset.id)}
                >
                  {modelCatalog.isPending || modelCatalog.isError
                    ? text("查看关联模型", "View linked models")
                    : text(
                        `已保存模型 ${modelCounts[dataset.id] ?? 0} 个`,
                        `${modelCounts[dataset.id] ?? 0} saved models`,
                      )}
                </Link>
                <p className="font-medium">
                  {previews[dataset.id]?.status === "ready"
                    ? `${previews[dataset.id]?.sample_count?.toLocaleString() ?? "—"} ${text("个样本", "samples")} · ${previews[dataset.id]?.feature_count?.toLocaleString() ?? "—"} ${text("个光谱变量", "spectral variables")}`
                    : (previewErrors[dataset.id] ??
                      text("数据结构待确认", "Structure pending confirmation"))}
                </p>
                <p className="text-muted-foreground text-sm">
                  {text("目标：", "Target: ")}
                  {previews[dataset.id]?.targets.length
                    ? previews[dataset.id]!.targets.map(
                        (item) => item.name,
                      ).join("、")
                    : text("待确认", "Unconfirmed")}
                </p>
                <p className="text-muted-foreground text-xs break-words">
                  {dataset.original_filename} ·{" "}
                  {formatBytes(dataset.size_bytes)} ·{" "}
                  {dateText(dataset.created_at, locale)}
                </p>
                <div
                  className="mt-auto flex flex-wrap items-center gap-2 border-t pt-3"
                  onClick={(event) => event.stopPropagation()}
                >
                  {dataset.status === "ready" && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() =>
                        router.push(
                          `/workspace/chats/new?dataset=${encodeURIComponent(dataset.id)}`,
                        )
                      }
                    >
                      <LinkIcon className="size-4" />
                      {text("在新对话中使用", "Use in a new chat")}
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
                    {text("打开工作台", "Open workspace")}
                  </Button>
                  {previews[dataset.id]?.status === "needs_mapping" && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => open("details", dataset, true)}
                    >
                      {text("确认数据结构", "Confirm structure")}
                    </Button>
                  )}
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        size="icon"
                        variant="ghost"
                        className="ml-auto size-8"
                        aria-label={`${text("更多操作", "More actions")}：${dataset.name}`}
                      >
                        <MoreHorizontalIcon className="size-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onSelect={() => open("details", dataset)}
                      >
                        <BarChart3Icon className="size-4" />
                        {text(
                          "数据概览与解析版本",
                          "Overview and parsing versions",
                        )}
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onSelect={() => open("rename", dataset)}
                      >
                        <PencilIcon className="size-4" />
                        {copy.rename}
                      </DropdownMenuItem>
                      {dataset.status === "archived" && (
                        <DropdownMenuItem
                          onSelect={() => open("delete", dataset)}
                        >
                          <Trash2Icon className="size-4" />
                          {copy.delete}
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
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
                "从电脑选择 CSV、TXT 或 MAT 文件。上传后可在工作台检查数据，再带入新对话。支持 CSV、TXT 和 MAT 格式。",
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
          {selected && (
            <DatasetDetailsPanel
              key={selected.id}
              dataset={selected}
              expandParsing={parsingExpanded}
              onSaved={refresh}
            />
          )}
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
