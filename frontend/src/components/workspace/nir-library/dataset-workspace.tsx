"use client";

import { ArrowLeftIcon, LinkIcon, RefreshCwIcon, SaveIcon } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SectionTabs } from "@/components/workspace/section-tabs";
import { useI18n } from "@/core/i18n/hooks";
import { chartColors, seriesColor } from "@/core/nir/chart-theme";
import {
  getDataset,
  getDatasetWorkspace,
  getDatasetWorkspaceRows,
  getDatasetWorkspaceSpectra,
  saveDatasetWorkspace,
  useNirLibraryEnabled,
  type NIRDataset,
  type DatasetWorkspace,
  type DatasetWorkspaceRows,
  type DatasetWorkspaceSpectra,
} from "@/core/nir-library";

import { DatasetDetailsPanel } from "./dataset-details-panel";
import { FeatureDisabled } from "./dataset-library";

const selectClass =
  "border-input bg-background h-7 rounded border px-2 text-sm";

function matches(
  response: {
    dataset_id: string;
    source_sha256: string;
    profile_id: string | null;
  },
  workspace: DatasetWorkspace,
) {
  if (
    response.dataset_id !== workspace.dataset_id ||
    response.source_sha256 !== workspace.source_sha256 ||
    response.profile_id !== workspace.profile_id
  )
    throw new Error("数据版本已变化，请重新载入。当前未保存修改仍保留。");
}

export function DatasetWorkspaceView({ datasetId }: { datasetId: string }) {
  const router = useRouter();
  const { locale } = useI18n();
  const text = (cn: string, en: string) => (locale.startsWith("zh") ? cn : en);
  const labels = [
    text("待分配", "Unassigned"),
    text("训练集", "Training"),
    text("验证集", "Validation"),
    text("测试集", "Test"),
    text("不参与", "Excluded"),
  ];
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [workspace, setWorkspace] = useState<DatasetWorkspace | null>(null);
  const [name, setName] = useState("");
  const [dataset, setDataset] = useState<NIRDataset | null>(null);
  const [section, setSection] = useState<"samples" | "overview" | "parsing">(
    "samples",
  );
  const [metadataOpened, setMetadataOpened] = useState(false);
  const [metadataSection, setMetadataSection] = useState<
    "overview" | "parsing"
  >("overview");
  useEffect(() => {
    setSection("samples");
    setMetadataOpened(false);
  }, [datasetId]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [usages, setUsages] = useState<number[]>([]);
  const [band, setBand] = useState<[string, string]>(["", ""]);
  const [selection, setSelection] = useState<number[]>([]);
  const [batchUsage, setBatchUsage] = useState(1);
  const [rows, setRows] = useState<DatasetWorkspaceRows | null>(null);
  const [rowError, setRowError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [overview, setOverview] = useState<DatasetWorkspaceSpectra | null>(
    null,
  );
  const [selectedSpectra, setSelectedSpectra] =
    useState<DatasetWorkspaceSpectra | null>(null);
  const [plotError, setPlotError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const saveRequest = useRef<AbortController | null>(null);
  const low = Number(band[0]),
    high = Number(band[1]);
  const range: [number, number] = [low, high];
  const dirty =
    !!workspace &&
    (JSON.stringify(usages) !== JSON.stringify(workspace.usages) ||
      band[0] !== String(workspace.range[0]) ||
      band[1] !== String(workspace.range[1]));
  const validRange =
    !!workspace &&
    band.every((v) => v.trim()) &&
    Number.isFinite(low) &&
    Number.isFinite(high) &&
    low <= high &&
    low >= Math.min(...workspace.axis) &&
    high <= Math.max(...workspace.axis) &&
    workspace.axis.some((v) => v >= low && v <= high);
  const selectedFeatures =
    workspace?.axis.filter((v) => v >= low && v <= high).length ?? 0;
  const selectionKey = selection.slice(0, 24).join(",");
  const counts = labels.map(
    (_, code) => usages.filter((value) => value === code).length,
  );

  useEffect(() => {
    if (!enabled) return;
    const controller = new AbortController();
    setReady(false);
    setError(null);
    setWorkspace(null);
    setDataset(null);
    setRows(null);
    setOverview(null);
    setSelectedSpectra(null);
    Promise.all([
      getDataset(datasetId, controller.signal).then((data) => {
        if (!controller.signal.aborted && data.id === datasetId) {
          setDataset(data);
          setName(data.name);
        }
        return data;
      }),
      getDatasetWorkspace(datasetId, controller.signal),
    ])
      .then(([dataset, data]) => {
        if (controller.signal.aborted) return;
        if (
          data.dataset_id !== dataset.id ||
          data.source_sha256 !== dataset.sha256
        )
          throw new Error("数据校验失败，请重新载入。");
        setName(dataset.name);
        setDataset(dataset);
        setWorkspace(data);
        setUsages(data.usages.slice());
        setBand([String(data.range[0]), String(data.range[1])]);
        setSelection([]);
        setOffset(0);
        setQuery("");
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setReady(true);
      });
    return () => controller.abort();
  }, [datasetId, enabled, reload]);

  useEffect(() => {
    if (!workspace) return;
    const controller = new AbortController();
    setRows(null);
    setRowError(null);
    getDatasetWorkspaceRows(datasetId, offset, query, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        matches(data, workspace);
        setRows(data);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setRowError(
            reason instanceof Error ? reason.message : String(reason),
          );
      });
    return () => controller.abort();
  }, [datasetId, offset, query, workspace]);

  useEffect(() => {
    if (!workspace) return;
    const controller = new AbortController();
    setOverview(null);
    setPlotError(null);
    getDatasetWorkspaceSpectra(datasetId, null, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        matches(data, workspace);
        setOverview(data);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setPlotError(
            reason instanceof Error ? reason.message : String(reason),
          );
      });
    return () => controller.abort();
  }, [datasetId, workspace]);

  useEffect(() => {
    if (!workspace || !selectionKey) {
      setSelectedSpectra(null);
      return;
    }
    const controller = new AbortController();
    setSelectedSpectra(null);
    getDatasetWorkspaceSpectra(
      datasetId,
      selectionKey.split(",").map(Number),
      controller.signal,
    )
      .then((data) => {
        if (controller.signal.aborted) return;
        matches(data, workspace);
        setSelectedSpectra(data);
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setPlotError(
            reason instanceof Error ? reason.message : String(reason),
          );
      });
    return () => controller.abort();
  }, [datasetId, selectionKey, workspace]);

  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => {
      if (dirty) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    const click = (event: MouseEvent) => {
      const anchor = (event.target as Element).closest?.("a[href]");
      if (
        dirty &&
        anchor &&
        anchor.getAttribute("href") !== location.pathname &&
        !window.confirm("还有未保存的修改，确定离开并放弃这些修改吗？")
      ) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    window.addEventListener("beforeunload", handler);
    document.addEventListener("click", click, true);
    return () => {
      window.removeEventListener("beforeunload", handler);
      document.removeEventListener("click", click, true);
    };
  }, [dirty]);
  useEffect(() => () => saveRequest.current?.abort(), []);

  function discardAllowed() {
    return (
      !dirty ||
      window.confirm(
        text("还有未保存的修改，确定放弃吗？", "Discard unsaved changes?"),
      )
    );
  }
  function changeUsage(indices: number[], code: number) {
    const selected = new Set(indices);
    setUsages((current) =>
      current.map((value, i) => (selected.has(i) ? code : value)),
    );
  }
  async function save() {
    if (!workspace || !validRange || saving) return;
    const controller = new AbortController();
    saveRequest.current = controller;
    setSaving(true);
    try {
      const data = await saveDatasetWorkspace(
        workspace,
        usages,
        range,
        controller.signal,
      );
      if (controller.signal.aborted) return;
      if (
        data.dataset_id !== datasetId ||
        data.source_sha256 !== workspace.source_sha256
      )
        throw new Error("保存返回的数据不匹配，请重新载入核对。");
      setWorkspace(data);
      setUsages(data.usages.slice());
      setBand([String(data.range[0]), String(data.range[1])]);
      toast.success(text("修改已保存", "Changes saved"));
    } catch (reason) {
      if (!controller.signal.aborted)
        toast.error(reason instanceof Error ? reason.message : String(reason));
    } finally {
      if (!controller.signal.aborted) setSaving(false);
    }
  }

  if (!featureLoading && !enabled)
    return (
      <FeatureDisabled
        title={text("数据集工作台暂未启用", "Dataset workspace unavailable")}
        description={text(
          "请先启用数据集库。",
          "Enable the dataset library first.",
        )}
      />
    );
  return (
    <div
      className="w-full min-w-0 space-y-4 p-4 lg:p-6"
      data-testid="dataset-workspace"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <Button
            aria-label={text("返回数据集库", "Back to datasets")}
            variant="outline"
            size="icon"
            onClick={() => {
              if (discardAllowed()) router.push("/workspace/nir/datasets");
            }}
          >
            <ArrowLeftIcon />
          </Button>
          <div className="min-w-0">
            <h1 className="truncate text-xl font-semibold" title={name}>
              {name || text("数据集工作台", "Dataset workspace")}
            </h1>
            <p className="text-muted-foreground text-sm">
              {text(
                "调整样本用途、选择波段，光谱和标签数值只读。",
                "Assign sample roles and select wavelengths. Values are read-only.",
              )}
            </p>
            {workspace && (
              <p className="text-primary mt-1 text-xs" role="status">
                {dirty
                  ? text("有未保存修改", "Unsaved changes")
                  : workspace.saved
                    ? `${text("已保存", "Saved")} · ${text("版本", "Version")} ${workspace.version}`
                    : text("尚未保存设置", "Settings not saved")}
              </p>
            )}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            disabled={saving}
            onClick={() => {
              if (discardAllowed()) setReload((v) => v + 1);
            }}
          >
            <RefreshCwIcon />
            {text("重新载入", "Reload")}
          </Button>
          <Button
            disabled={
              !workspace || !validRange || saving || (!dirty && workspace.saved)
            }
            variant={dirty || !workspace?.saved ? "default" : "outline"}
            onClick={() => void save()}
          >
            <SaveIcon />
            {saving
              ? text("保存中…", "Saving…")
              : text("保存修改", "Save changes")}
          </Button>
          <Button
            variant={!dirty && workspace?.saved ? "default" : "outline"}
            title={
              dirty ? text("请先保存修改", "Save changes first") : undefined
            }
            disabled={!workspace?.saved || dirty || saving}
            onClick={() =>
              router.push(
                `/workspace/chats/new?dataset=${encodeURIComponent(datasetId)}`,
              )
            }
          >
            <LinkIcon />
            {text("在新对话中使用", "Use in a new chat")}
          </Button>
        </div>
      </div>
      <SectionTabs
        id="dataset-sections"
        label={text("数据集工作台内容", "Dataset workspace sections")}
        value={section}
        onChange={(value) => {
          setSection(value);
          if (value !== "samples") {
            setMetadataOpened(true);
            setMetadataSection(value);
          }
        }}
        items={[
          { key: "samples", label: text("样本与光谱", "Samples and spectra") },
          { key: "overview", label: text("数据概览", "Overview") },
          { key: "parsing", label: text("解析设置", "Parsing settings") },
        ]}
      />
      {dataset && metadataOpened && (
        <section
          hidden={section === "samples"}
          id={`dataset-sections-panel-${metadataSection}`}
          role="tabpanel"
          aria-labelledby={`dataset-sections-${metadataSection}`}
          tabIndex={0}
        >
          <DatasetDetailsPanel
            key={dataset.id}
            dataset={dataset}
            section={metadataSection}
            disabled={dirty || saving}
            onSaved={() => setReload((v) => v + 1)}
          />
        </section>
      )}
      <div
        hidden={section !== "samples"}
        id="dataset-sections-panel-samples"
        role="tabpanel"
        aria-labelledby="dataset-sections-samples"
        tabIndex={0}
      >
        {error ? (
          <div role="alert" className="rounded-lg border p-5 text-sm">
            {error}
          </div>
        ) : !ready || !workspace ? (
          <p>{text("正在读取数据…", "Loading data…")}</p>
        ) : (
          <>
            <div className="bg-card flex flex-wrap items-center gap-x-5 gap-y-2 rounded-lg border px-4 py-3 text-sm">
              <strong>
                {text("共", "Total")} {workspace.sample_count}{" "}
                {text("个样本", "samples")}
              </strong>
              {labels.map((label, i) => (
                <span key={label} style={{ color: seriesColor(i) }}>
                  {label} {counts[i]}
                </span>
              ))}
              <span className="ml-auto">
                {text("光谱变量", "Spectral variables")}{" "}
                {workspace.feature_count}
              </span>
            </div>
            <div className="grid min-w-0 gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,1fr)]">
              <section
                className="bg-card min-w-0 overflow-hidden rounded-lg border"
                aria-label={text("样本表格", "Samples table")}
              >
                <div className="flex flex-wrap items-center gap-2 border-b p-3">
                  <Input
                    className="w-40"
                    placeholder={text("搜索样本名称", "Search samples")}
                    aria-label={text("搜索样本名称", "Search samples")}
                    value={query}
                    disabled={saving}
                    onChange={(e) => {
                      setQuery(e.target.value);
                      setOffset(0);
                    }}
                  />
                  <span className="text-sm">
                    {text("选中", "Selected")} {selection.length}
                  </span>
                  <div
                    hidden={!selection.length}
                    className="flex flex-wrap gap-2"
                  >
                    <select
                      className={selectClass}
                      aria-label={text("批量样本用途", "Bulk sample role")}
                      disabled={saving}
                      value={batchUsage}
                      onChange={(e) => setBatchUsage(Number(e.target.value))}
                    >
                      {labels.map((label, i) => (
                        <option key={label} value={i}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={!selection.length || saving}
                      onClick={() => changeUsage(selection, batchUsage)}
                    >
                      {text("应用到选中样本", "Apply to selection")}
                    </Button>
                  </div>
                  {!selection.length && (
                    <span className="text-muted-foreground text-xs">
                      {text(
                        "选择后可批量设置用途",
                        "Select samples to assign roles in bulk",
                      )}
                    </span>
                  )}
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={saving}
                    onClick={() =>
                      setSelection(
                        Array.from(
                          { length: workspace.sample_count },
                          (_, i) => i,
                        ),
                      )
                    }
                  >
                    {text("选择全部样本", "Select all samples")}
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelection([])}
                  >
                    {text("清除选择", "Clear selection")}
                  </Button>
                </div>
                <div
                  className="max-h-[520px] min-h-60 overflow-auto"
                  data-testid="sample-table-scroll"
                >
                  {rowError ? (
                    <p role="alert" className="p-4 text-sm">
                      {rowError}
                    </p>
                  ) : !rows ? (
                    <p className="p-4 text-sm">
                      {text("正在读取样本…", "Loading samples…")}
                    </p>
                  ) : (
                    <table
                      className="w-full border-collapse text-sm whitespace-nowrap"
                      aria-label={text("样本信息", "Sample information")}
                    >
                      <thead className="bg-muted sticky top-0 z-10">
                        <tr>
                          <th className="border-r border-b px-3 py-2">
                            <input
                              type="checkbox"
                              aria-label={text(
                                "选择本页样本",
                                "Select this page",
                              )}
                              checked={
                                rows.rows.length > 0 &&
                                rows.rows.every((r) =>
                                  selection.includes(r.index),
                                )
                              }
                              onChange={(e) =>
                                setSelection((current) =>
                                  e.target.checked
                                    ? [
                                        ...new Set([
                                          ...current,
                                          ...rows.rows.map((r) => r.index),
                                        ]),
                                      ]
                                    : current.filter(
                                        (i) =>
                                          !rows.rows.some((r) => r.index === i),
                                      ),
                                )
                              }
                            />
                          </th>
                          {[
                            text("序号", "No."),
                            text("样本名称", "Sample"),
                            text("样本用途", "Role"),
                            ...workspace.fields,
                          ].map((label, i) => (
                            <th
                              key={`${i}-${label}`}
                              className="border-r border-b px-3 py-2 text-left font-medium"
                            >
                              {label}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {rows.rows.map((row) => (
                          <tr
                            key={row.index}
                            className={
                              selection.includes(row.index)
                                ? "bg-blue-500/10"
                                : "even:bg-muted/30"
                            }
                          >
                            <td className="border-r border-b px-3 py-1.5">
                              <input
                                type="checkbox"
                                aria-label={`${text("选择样本", "Select sample")} ${row.index + 1}`}
                                checked={selection.includes(row.index)}
                                onChange={(e) =>
                                  setSelection((current) =>
                                    e.target.checked
                                      ? [...current, row.index]
                                      : current.filter((i) => i !== row.index),
                                  )
                                }
                              />
                            </td>
                            <td className="text-muted-foreground border-r border-b px-3 py-1.5">
                              {row.index + 1}
                            </td>
                            <td className="border-r border-b px-3 py-1.5">
                              <button
                                className="max-w-60 truncate text-left hover:underline"
                                title={row.name}
                                onClick={() => setSelection([row.index])}
                              >
                                {row.name}
                              </button>
                            </td>
                            <td className="border-r border-b px-3 py-1.5">
                              <select
                                className={selectClass}
                                style={{
                                  color: seriesColor(usages[row.index] ?? 0),
                                }}
                                aria-label={`${text("样本用途", "Sample role")} ${row.index + 1}`}
                                value={usages[row.index] ?? 0}
                                disabled={saving}
                                onChange={(e) =>
                                  changeUsage(
                                    [row.index],
                                    Number(e.target.value),
                                  )
                                }
                              >
                                {labels.map((label, i) => (
                                  <option key={label} value={i}>
                                    {label}
                                  </option>
                                ))}
                              </select>
                            </td>
                            {row.values.map((value, i) => (
                              <td
                                key={i}
                                className="border-r border-b px-3 py-1.5"
                                title={String(value ?? "")}
                              >
                                {value ?? "—"}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                <div className="flex items-center justify-between gap-2 border-t px-3 py-2 text-xs">
                  <span>
                    {rows
                      ? `${Math.min(offset + 1, rows.total)}–${Math.min(offset + 50, rows.total)} / ${rows.total}`
                      : "…"}
                  </span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={offset === 0 || saving}
                      onClick={() => setOffset((v) => Math.max(0, v - 50))}
                    >
                      {text("上一页", "Previous")}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!rows || offset + 50 >= rows.total || saving}
                      onClick={() => setOffset((v) => v + 50)}
                    >
                      {text("下一页", "Next")}
                    </Button>
                  </div>
                </div>
              </section>
              <aside className="min-w-0 space-y-3">
                <div className="bg-card flex flex-wrap items-center gap-3 rounded-lg border p-3">
                  <span className="text-sm font-medium">
                    {text("保留光谱范围", "Retained spectral range")}
                  </span>
                  <label className="flex items-center gap-2 text-sm">
                    {text("起始", "Start")}
                    <Input
                      className="w-28"
                      type="number"
                      step="any"
                      aria-label={text("波段起始值", "Range start")}
                      disabled={saving}
                      value={band[0]}
                      onChange={(e) => setBand([e.target.value, band[1]])}
                    />
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    {text("结束", "End")}
                    <Input
                      className="w-28"
                      type="number"
                      step="any"
                      aria-label={text("波段结束值", "Range end")}
                      disabled={saving}
                      value={band[1]}
                      onChange={(e) => setBand([band[0], e.target.value])}
                    />
                  </label>
                  <Button
                    variant="outline"
                    disabled={saving}
                    onClick={() =>
                      setBand([
                        String(Math.min(...workspace.axis)),
                        String(Math.max(...workspace.axis)),
                      ])
                    }
                  >
                    {text("全部波段", "Full range")}
                  </Button>
                  <span className="text-muted-foreground text-sm">
                    {text("保留", "Keeping")} {selectedFeatures} /{" "}
                    {workspace.feature_count}{" "}
                    {text(
                      "个光谱变量；也可在图上拖动选择",
                      "variables; drag on either plot to select",
                    )}
                  </span>
                  {!validRange && (
                    <p role="alert" className="text-destructive text-sm">
                      {text(
                        "请输入文件坐标内的有效范围，至少保留一个变量。",
                        "Select a valid range containing at least one variable.",
                      )}
                    </p>
                  )}
                </div>

                <SpectrumPlot
                  title={text("选中样本的光谱", "Selected spectra")}
                  spectra={selectedSpectra}
                  usages={usages}
                  range={range}
                  axisKind={workspace.axis_kind}
                  disabled={saving}
                  onRange={(v) => setBand(v.map(String) as [string, string])}
                  empty={
                    selection.length
                      ? text("正在读取选中光谱…", "Loading selected spectra…")
                      : text(
                          "点击样本名称或勾选样本查看光谱",
                          "Select samples to view spectra",
                        )
                  }
                />
                {selection.length > 24 && (
                  <p className="text-muted-foreground text-xs">
                    {text(
                      "用途调整作用于全部选中样本；图中最多展示前 24 条光谱。",
                      "Role changes apply to the whole selection; the plot displays up to 24 spectra.",
                    )}
                  </p>
                )}
                <SpectrumPlot
                  title={text("数据集光谱概览", "Dataset spectra overview")}
                  spectra={overview}
                  usages={usages}
                  range={range}
                  axisKind={workspace.axis_kind}
                  disabled={saving}
                  onRange={(v) => setBand(v.map(String) as [string, string])}
                  empty={text("正在读取光谱…", "Loading spectra…")}
                />
                {overview && (
                  <p className="text-muted-foreground text-xs">
                    {text("共", "Total")} {overview.total}{" "}
                    {text("个样本，均匀抽取展示", "samples; displaying")}{" "}
                    {overview.curves.length}{" "}
                    {text(
                      "条光谱。每条曲线最多展示 512 个点，保存时保留所选范围内的全部原始数值。",
                      "evenly sampled spectra, up to 512 points per curve. Saving retains all original values in the selected range.",
                    )}
                  </p>
                )}
                {plotError && (
                  <p role="alert" className="text-destructive text-sm">
                    {plotError}
                  </p>
                )}
              </aside>
            </div>
            <p className="text-muted-foreground text-xs">
              {text(
                "训练集用于学习模型；验证集用于选择方法和参数；测试集用于最终检验；不参与的样本保留在文件中，但不用于建模。用途与样本类别分别记录。保存后的设置只作用于后续挂载，已有对话中的副本保留。",
                "Training fits the model; validation selects methods and parameters; test evaluates the final model. Excluded samples remain in the file. Saved settings apply to future attachments; existing chat copies are retained.",
              )}
            </p>
          </>
        )}
      </div>
    </div>
  );
}

function SpectrumPlot({
  title,
  spectra,
  usages,
  range,
  axisKind,
  onRange,
  empty,
  disabled,
}: {
  title: string;
  spectra: DatasetWorkspaceSpectra | null;
  usages: number[];
  range: [number, number];
  axisKind: "axis" | "index";
  onRange: (range: [number, number]) => void;
  empty: string;
  disabled: boolean;
}) {
  const { locale } = useI18n();
  const zh = locale.startsWith("zh");
  const drag = useRef<number | null>(null);
  const [dragRange, setDragRange] = useState<[number, number] | null>(null);
  const geometry = useMemo(() => {
    if (!spectra?.axis.length) return null;
    const values = spectra.curves.flatMap((curve) =>
      curve.values.filter((v): v is number => v !== null && Number.isFinite(v)),
    );
    const xmin = Math.min(...spectra.axis),
      xmax = Math.max(...spectra.axis);
    let ymin = Math.min(...values),
      ymax = Math.max(...values);
    if (!values.length) {
      ymin = 0;
      ymax = 1;
    }
    if (ymin === ymax) {
      ymin -= 0.5;
      ymax += 0.5;
    }
    return { xmin, xmax, ymin, ymax };
  }, [spectra]);
  const G = geometry;
  const x = (value: number) =>
    48 + ((value - G!.xmin) / (G!.xmax - G!.xmin || 1)) * 330;
  const y = (value: number) =>
    175 - ((value - G!.ymin) / (G!.ymax - G!.ymin)) * 150;
  const activeRange = dragRange ?? range;
  function coordinate(event: React.PointerEvent<SVGSVGElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return (
      G!.xmin +
      Math.max(
        0,
        Math.min(
          1,
          (((event.clientX - rect.left) / rect.width) * 400 - 48) / 330,
        ),
      ) *
        (G!.xmax - G!.xmin)
    );
  }
  return (
    <div className="bg-card rounded-lg border p-3">
      <h2 className="mb-1 text-sm font-medium">
        {title}
        {spectra ? ` · ${spectra.curves.length}` : ""}
      </h2>
      {!G || !spectra?.curves.length ? (
        <div className="text-muted-foreground flex h-52 items-center justify-center p-4 text-center text-sm">
          {empty}
        </div>
      ) : (
        <svg
          viewBox="0 0 400 225"
          role="img"
          aria-label={title}
          className="w-full touch-none select-none"
          onPointerDown={(event) => {
            if (disabled) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = coordinate(event);
          }}
          onPointerMove={(event) => {
            if (drag.current !== null)
              setDragRange([
                Math.min(drag.current, coordinate(event)),
                Math.max(drag.current, coordinate(event)),
              ]);
          }}
          onPointerUp={(event) => {
            if (drag.current === null) return;
            const value = coordinate(event),
              start = drag.current;
            drag.current = null;
            setDragRange(null);
            if (Math.abs(value - start) > (G.xmax - G.xmin) / 330)
              onRange([Math.min(start, value), Math.max(start, value)]);
          }}
          onPointerCancel={() => {
            drag.current = null;
            setDragRange(null);
          }}
        >
          {Array.from({ length: 5 }, (_, i) => {
            const xv = G.xmin + ((G.xmax - G.xmin) * i) / 4,
              yv = G.ymin + ((G.ymax - G.ymin) * i) / 4;
            return (
              <g key={i} className="text-muted-foreground" fontSize="12">
                <line
                  x1={x(xv)}
                  x2={x(xv)}
                  y1="25"
                  y2="175"
                  stroke="currentColor"
                  opacity=".15"
                />
                <line
                  x1="48"
                  x2="378"
                  y1={y(yv)}
                  y2={y(yv)}
                  stroke="currentColor"
                  opacity=".15"
                />
                <text x={x(xv)} y="193" textAnchor="middle" fill="currentColor">
                  {Number(xv.toPrecision(5))}
                </text>
                <text x="42" y={y(yv) + 3} textAnchor="end" fill="currentColor">
                  {Number(yv.toPrecision(4))}
                </text>
              </g>
            );
          })}
          {activeRange.every(Number.isFinite) && (
            <rect
              data-testid="selected-band"
              x={x(Math.max(G.xmin, Math.min(G.xmax, activeRange[0])))}
              y="25"
              width={Math.max(
                0,
                x(Math.max(G.xmin, Math.min(G.xmax, activeRange[1]))) -
                  x(Math.max(G.xmin, Math.min(G.xmax, activeRange[0]))),
              )}
              height="150"
              fill={chartColors.processed}
              opacity=".1"
            />
          )}
          {spectra.curves.map((curve) => {
            let pen = false;
            const path = curve.values
              .map((value, i) => {
                if (value === null || !Number.isFinite(value)) {
                  pen = false;
                  return "";
                }
                const command = pen ? "L" : "M";
                pen = true;
                return `${command}${x(spectra.axis[i]!).toFixed(2)},${y(value).toFixed(2)}`;
              })
              .join(" ");
            return (
              <path
                key={curve.index}
                d={path}
                fill="none"
                stroke={seriesColor(usages[curve.index] ?? 0)}
                strokeWidth="1.4"
                opacity=".75"
              >
                <title>{curve.name}</title>
              </path>
            );
          })}
          <text
            x="213"
            y="219"
            textAnchor="middle"
            fill="currentColor"
            fontSize="12"
          >
            {axisKind === "index"
              ? zh
                ? "光谱变量序号"
                : "Spectral variable index"
              : zh
                ? "光谱坐标（单位以原文件为准）"
                : "Spectral coordinate (source units)"}
          </text>
          <text
            transform="translate(12,100) rotate(-90)"
            textAnchor="middle"
            fill="currentColor"
            fontSize="12"
          >
            {zh ? "光谱值" : "Spectral value"}
          </text>
        </svg>
      )}
    </div>
  );
}
