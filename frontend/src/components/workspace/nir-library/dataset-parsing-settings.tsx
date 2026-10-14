"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/core/i18n/hooks";
import {
  getDatasetParsing,
  previewDatasetParsing,
  saveDatasetParsing,
} from "@/core/nir-library";
import type {
  DatasetParsing,
  DatasetParsingPreview,
  DatasetParsingSelection,
} from "@/core/nir-library/types";

import { DatasetPreview } from "./dataset-preview";

const selectClass =
  "border-input bg-background h-9 w-full min-w-0 rounded-md border px-2 text-sm";

export function DatasetParsingSettings({
  datasetId,
  sourceSha,
  disabled,
  onSaved,
}: {
  datasetId: string;
  sourceSha: string | null;
  disabled: boolean;
  onSaved: () => void;
}) {
  const { locale } = useI18n();
  const text = (cn: string, en: string) => (locale.startsWith("zh") ? cn : en);
  const [schema, setSchema] = useState<DatasetParsing | null>(null);
  const [selection, setSelection] = useState<DatasetParsingSelection>({});
  const [taskType, setTaskType] = useState("inspection");
  const [preview, setPreview] = useState<{
    key: string;
    data: DatasetParsingPreview;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [previewing, setPreviewing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setSchema(null);
    setPreview(null);
    setError(null);
    getDatasetParsing(datasetId, controller.signal)
      .then((data) => {
        if (controller.signal.aborted) return;
        if (data.dataset_id !== datasetId || data.source_sha256 !== sourceSha)
          throw new Error("文件校验失败，请重新载入。");
        setSchema(data);
        setSelection(data.current ?? data.suggested ?? {});
        setTaskType(data.task_type ?? "inspection");
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted)
          setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [datasetId, sourceSha, revision]);
  const key = JSON.stringify({
    selection,
    taskType,
    profileId: schema?.profile_id,
  });
  const layout = schema?.layouts.find(
    (item) => item.id === selection.layout_id,
  );
  const groups = [...new Set(schema?.layouts.map((item) => item.group))];
  const canPreview =
    !!schema &&
    (schema.format === "mat"
      ? !!layout
      : selection.spectral_start != null && selection.spectral_end != null);
  useEffect(() => {
    if (!schema || !canPreview) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    setPreview(null);
    setError(null);
    setPreviewing(true);
    const timer = setTimeout(() => {
      previewDatasetParsing(schema, selection, taskType, controller.signal)
        .then((data) => {
          if (controller.signal.aborted) return;
          if (data.dataset_id !== datasetId || data.source_sha256 !== sourceSha)
            throw new Error("预览与原文件不匹配。");
          setPreview({ key, data });
        })
        .catch((reason: unknown) => {
          if (!controller.signal.aborted)
            setError(reason instanceof Error ? reason.message : String(reason));
        })
        .finally(() => {
          if (!controller.signal.aborted) setPreviewing(false);
        });
    }, 300);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [schema, canPreview, selection, taskType, datasetId, sourceSha, key]);
  function chooseLayout(id: string) {
    const next = schema?.layouts.find((item) => item.id === id);
    setSelection(
      next
        ? {
            layout_id: next.id,
            target_id: next.default_target,
            axis_id: next.default_axis,
          }
        : {},
    );
  }
  function update(patch: Partial<DatasetParsingSelection>) {
    setSelection((current) => ({ ...current, ...patch }));
  }
  async function save() {
    if (!schema || preview?.key !== key) return;
    setSaving(true);
    try {
      const result = await saveDatasetParsing(schema, selection, taskType);
      setSchema((current) =>
        current
          ? { ...current, profile_id: result.id, current: selection }
          : current,
      );
      toast.success(
        text(
          "解析设置已保存。查看详情和挂载到对话会使用当前选择。",
          "Parsing saved. Details and chat attachments will use this selection.",
        ),
      );
      onSaved();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  }
  const locked = disabled || saving;
  const columnOptions = schema?.columns.map((column) => (
    <option key={column.index} value={column.index}>
      {column.index + 1}. {column.name}
    </option>
  ));
  return (
    <section className="space-y-3" data-testid="dataset-parsing-settings">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="font-medium">
          {text("选择解析内容", "Choose data to parse")}
        </h4>
        <Button
          size="sm"
          variant="outline"
          disabled={loading || saving}
          onClick={() => setRevision((v) => v + 1)}
        >
          {text("重新识别", "Detect again")}
        </Button>
      </div>
      <p className="text-muted-foreground text-sm">
        {text(
          "先选择数据结构，再检查下方预览。保存只记录解析方式，原文件不会更改。更换数据组或列选择后，需重新设置该组的样本用途和波段。",
          "Choose the structure and check the preview. Saving keeps the source unchanged. Changing groups or columns resets their sample roles and wavelength range.",
        )}
      </p>
      {loading && (
        <p className="text-sm" role="status">
          {text("正在识别文件结构…", "Detecting file structure…")}
        </p>
      )}
      {schema && (
        <fieldset disabled={locked} className="space-y-3">
          {schema.format === "mat" ? (
            <>
              <label className="grid gap-1 text-sm">
                {text("数据组", "Data group")}
                <select
                  className={selectClass}
                  value={layout?.group ?? ""}
                  aria-label={text("数据组", "Data group")}
                  onChange={(event) =>
                    chooseLayout(
                      schema.layouts.find(
                        (item) => item.group === event.target.value,
                      )?.id ?? "",
                    )
                  }
                >
                  <option value="">
                    {text("请选择一个数据组", "Choose a data group")}
                  </option>
                  {groups.map((group) => {
                    const first = schema.layouts.find(
                      (item) => item.group === group,
                    )!;
                    return (
                      <option key={group} value={group}>
                        {group.split(".").at(-1)} · {first.sample_count}{" "}
                        {text("个样本", "samples")}
                      </option>
                    );
                  })}
                </select>
              </label>
              {layout && (
                <>
                  <label className="grid gap-1 text-sm">
                    {text("光谱范围", "Spectral block")}
                    <select
                      className={selectClass}
                      value={layout.id}
                      aria-label={text("光谱范围", "Spectral block")}
                      onChange={(event) => chooseLayout(event.target.value)}
                    >
                      {schema.layouts
                        .filter((item) => item.group === layout.group)
                        .map((item) => (
                          <option key={item.id} value={item.id}>
                            {item.label} · {item.feature_count}{" "}
                            {text("个光谱变量", "variables")}
                          </option>
                        ))}
                    </select>
                  </label>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <label className="grid gap-1 text-sm">
                      {text("目标值", "Target")}
                      <select
                        className={selectClass}
                        value={selection.target_id ?? ""}
                        aria-label={text("目标值", "Target")}
                        onChange={(event) =>
                          update({ target_id: event.target.value || null })
                        }
                      >
                        <option value="">
                          {text("不选择目标值", "No target")}
                        </option>
                        {layout.targets.map((item) => (
                          <option key={item.id} value={item.id}>
                            {item.label.split(".").at(-1)}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="grid gap-1 text-sm">
                      {text("光谱坐标", "Spectral coordinates")}
                      <select
                        className={selectClass}
                        value={selection.axis_id ?? ""}
                        aria-label={text("光谱坐标", "Spectral coordinates")}
                        onChange={(event) =>
                          update({ axis_id: event.target.value || null })
                        }
                      >
                        <option value="">
                          {text("使用光谱变量序号", "Use variable index")}
                        </option>
                        {layout.axes.map((item) => (
                          <option key={item.id} value={item.id}>
                            {item.label
                              .split(" + ")
                              .map((name) => name.split(".").at(-1))
                              .join(" + ")}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </>
              )}
            </>
          ) : (
            <>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={selection.has_header ?? false}
                  onChange={(event) =>
                    update({ has_header: event.target.checked })
                  }
                />
                {text("首行是列名", "First row contains column names")}
              </label>
              <label className="grid gap-1 text-sm">
                {text("表格排列方式", "Table orientation")}
                <select
                  className={selectClass}
                  value={selection.transpose ? "columns" : "rows"}
                  aria-label={text("表格排列方式", "Table orientation")}
                  onChange={(event) => {
                    const transpose = event.target.value === "columns";
                    update({
                      transpose,
                      target_columns: [],
                      sample_column: null,
                    });
                    if (transpose) setTaskType("inspection");
                  }}
                >
                  <option value="rows">
                    {text("每行一个样本", "Samples in rows")}
                  </option>
                  <option value="columns">
                    {text("每列一个样本", "Samples in columns")}
                  </option>
                </select>
              </label>
              <div className="grid grid-cols-2 gap-3">
                {(["spectral_start", "spectral_end"] as const).map(
                  (field, i) => (
                    <label key={field} className="grid gap-1 text-sm">
                      {i
                        ? text("光谱结束列", "Last spectral column")
                        : text("光谱起始列", "First spectral column")}
                      <Input
                        type="number"
                        min={1}
                        max={schema.columns.length}
                        value={
                          selection[field] == null ? "" : selection[field] + 1
                        }
                        onChange={(event) =>
                          update({
                            [field]: event.target.value
                              ? Number(event.target.value) - 1
                              : null,
                          })
                        }
                      />
                    </label>
                  ),
                )}
              </div>
              <p className="text-muted-foreground text-xs">
                {text(
                  "列号从 1 开始，可参考下方列名选择。",
                  "Column numbers start at 1. Use the names below as a guide.",
                )}
              </p>
              {selection.transpose ? (
                <label className="grid gap-1 text-sm">
                  {text("光谱坐标列", "Coordinate column")}
                  <select
                    className={selectClass}
                    value={selection.coordinate_column ?? ""}
                    aria-label={text("光谱坐标列", "Coordinate column")}
                    onChange={(event) =>
                      update({
                        coordinate_column: event.target.value
                          ? Number(event.target.value)
                          : null,
                      })
                    }
                  >
                    <option value="">{text("请选择", "Choose")}</option>
                    {columnOptions}
                  </select>
                </label>
              ) : (
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="grid gap-1 text-sm">
                    {text("目标值所在列", "Target column")}
                    <select
                      className={selectClass}
                      value={selection.target_columns?.[0] ?? ""}
                      aria-label={text("目标值所在列", "Target column")}
                      onChange={(event) =>
                        update({
                          target_columns: event.target.value
                            ? [Number(event.target.value)]
                            : [],
                        })
                      }
                    >
                      <option value="">
                        {text("不选择目标值", "No target")}
                      </option>
                      {columnOptions}
                    </select>
                  </label>
                  <label className="grid gap-1 text-sm">
                    {text("样本名称所在列", "Sample name column")}
                    <select
                      className={selectClass}
                      value={selection.sample_column ?? ""}
                      aria-label={text("样本名称所在列", "Sample name column")}
                      onChange={(event) =>
                        update({
                          sample_column: event.target.value
                            ? Number(event.target.value)
                            : null,
                        })
                      }
                    >
                      <option value="">
                        {text("自动编号", "Automatic numbering")}
                      </option>
                      {columnOptions}
                    </select>
                  </label>
                </div>
              )}
            </>
          )}
          <label className="grid gap-1 text-sm">
            {text("分析用途", "Analysis type")}
            <select
              className={selectClass}
              value={taskType}
              aria-label={text("分析用途", "Analysis type")}
              disabled={selection.transpose}
              onChange={(event) => setTaskType(event.target.value)}
            >
              <option value="inspection">
                {text("查看光谱（暂不指定分析用途）", "Inspect spectra")}
              </option>
              <option value="calibration">
                {text("定量分析（预测含量等数值）", "Quantitative analysis")}
              </option>
              <option value="classification">
                {text("分类识别（区分品种或类别）", "Classification")}
              </option>
            </select>
          </label>
        </fieldset>
      )}
      {error && (
        <p className="text-destructive text-sm" role="alert">
          {error}
        </p>
      )}
      {previewing && canPreview && (
        <p className="text-muted-foreground text-sm" role="status">
          {text("正在生成所选数据的预览…", "Preparing selected-data preview…")}
        </p>
      )}
      {preview?.key === key && (
        <div
          className="space-y-3 rounded-md border p-3"
          data-testid="parsing-preview"
        >
          <DatasetPreview preview={preview.data} />
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <caption className="mb-2 text-left font-medium">
                {text("前 5 个样本的目标值", "Targets for the first 5 samples")}
              </caption>
              <thead>
                <tr>
                  <th>{text("样本", "Sample")}</th>
                  <th>{text("目标值", "Target")}</th>
                </tr>
              </thead>
              <tbody>
                {preview.data.samples.map((sample, i) => (
                  <tr key={i} className="border-t">
                    <td className="py-1">{sample.name}</td>
                    <td>
                      {sample.values.map((v) => v ?? "—").join(" / ") || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <Button
        size="sm"
        disabled={locked || preview?.key !== key || previewing}
        onClick={() => void save()}
      >
        {saving
          ? text("正在保存…", "Saving…")
          : text("保存解析设置", "Save parsing settings")}
      </Button>
    </section>
  );
}
