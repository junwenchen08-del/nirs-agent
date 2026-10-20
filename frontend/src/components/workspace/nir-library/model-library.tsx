"use client";

import {
  ChevronDownIcon,
  DatabaseIcon,
  RefreshCwIcon,
  SearchIcon,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/core/i18n/hooks";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber } from "@/core/nir/process-display";
import { listDatasets, useNirLibraryEnabled } from "@/core/nir-library";
import { datasetVersionPath } from "@/core/nir-library/dataset-partitions";
import {
  groupModels,
  modelTask,
  type CatalogModel,
} from "@/core/nir-library/model-catalog";
import {
  cardMetrics,
  modelChatPath,
  modelDisplayName,
  modelReportPath,
  scopeLabel,
} from "@/core/nir-library/model-report";
import { useModelCatalog } from "@/core/nir-library/use-model-catalog";

import { FeatureDisabled } from "./dataset-library";
import { ModelQualityNotice } from "./model-quality-notice";

const EMPTY: CatalogModel[] = [];

export function ModelLibrary() {
  const { t, locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const params = useSearchParams();
  const datasetFilter = params.get("dataset");
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const catalog = useModelCatalog(enabled && !featureLoading);
  const models = catalog.data ?? EMPTY;
  const [datasetNames, setDatasetNames] = useState<Record<string, string>>({});
  const [query, setQuery] = useState("");
  const [target, setTarget] = useState("");
  const [task, setTask] = useState("");
  const [method, setMethod] = useState("");
  const [quality, setQuality] = useState("");
  const [view, setView] = useState<"dataset" | "all">("dataset");
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [groupTargets, setGroupTargets] = useState<Record<string, string>>({});
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (!enabled || featureLoading) return;
    const controller = new AbortController();
    void listDatasets(controller.signal)
      .then((rows) => {
        if (!controller.signal.aborted)
          setDatasetNames(
            Object.fromEntries(rows.map((row) => [row.id, row.name])),
          );
      })
      .catch(() => {
        /* Known IDs remain grouped when names are unavailable. */
      });
    return () => controller.abort();
  }, [enabled, featureLoading, refresh]);

  const nameOf = (model: CatalogModel) =>
    datasetNames[model.source_dataset_id ?? ""] ?? model.recoveredDatasetName;
  const scoped = useMemo(
    () =>
      models.filter(
        (model) => !datasetFilter || model.source_dataset_id === datasetFilter,
      ),
    [models, datasetFilter],
  );
  const targets = [
    ...new Set(
      scoped.map((model) => model.report_summary?.target?.trim() ?? ""),
    ),
  ].sort();
  const methods = [
    ...new Set(scoped.map((model) => model.method ?? "")),
  ].sort();
  const filtered = scoped.filter((model) => {
    const targetName = model.report_summary?.target?.trim() ?? "";
    return (
      (!target ||
        (target === "__unknown" ? !targetName : targetName === target)) &&
      (!task || modelTask(model) === task) &&
      (!method || (model.method ?? "__unknown") === method) &&
      (!quality ||
        (quality === "accepted"
          ? model.quality_status === "below_threshold_user_approved"
          : !model.quality_status)) &&
      [
        model.model_id,
        model.version,
        model.method ?? "",
        targetName,
        nameOf(model) ?? "",
        methodLabel(model.method, zh),
      ].some((value) =>
        value.toLowerCase().includes(query.trim().toLowerCase()),
      )
    );
  });
  const groups = groupModels(filtered);
  const date = (value: string | null) =>
    value && Number.isFinite(Date.parse(value))
      ? new Date(value).toLocaleDateString(locale)
      : text("时间未记录", "Date not recorded");
  const groupName = (id: string | null, rows: CatalogModel[]) =>
    id
      ? (nameOf(rows[0]!) ??
        text("数据集名称暂不可用", "Dataset name unavailable"))
      : text("训练数据来源待确认", "Training source unconfirmed");

  function row(model: CatalogModel, showDataset: boolean) {
    const metric =
      cardMetrics(model).find(
        (item) =>
          item.scope === "final_test" &&
          (item.R2 !== null || item.RMSE !== null),
      ) ?? cardMetrics(model).find((item) => item.scope === "validation");
    const steps = model.preprocessing?.steps;
    const pipeline = Array.isArray(steps)
      ? steps
          .map((step) =>
            typeof step.method === "string"
              ? methodLabel(step.method, zh)
              : text("未记录方法", "Unrecorded method"),
          )
          .join(" → ")
      : "";
    return (
      <article
        key={model.id}
        data-testid="nir-model-card"
        className="grid min-w-0 gap-4 border-t p-4 xl:grid-cols-[minmax(0,1fr)_220px_180px] xl:items-center"
      >
        <div className="min-w-0 space-y-2">
          <Link
            href={modelReportPath(model)}
            className="font-semibold break-words hover:underline"
          >
            {modelDisplayName(model, zh)}
          </Link>
          <p className="text-muted-foreground text-xs break-all">
            {model.model_id} · {model.version} · {date(model.created_at)}
          </p>
          <p className="text-sm break-words">
            {pipeline ? `${pipeline} → ` : ""}
            {methodLabel(model.method, zh)}
          </p>
          {showDataset && (
            <p className="text-muted-foreground text-sm">
              {nameOf(model) ??
                (model.source_dataset_id
                  ? text("数据集名称暂不可用", "Dataset name unavailable")
                  : text("训练数据来源待确认", "Training source unconfirmed"))}
            </p>
          )}
          <p className="text-muted-foreground text-xs break-all">
            {text("训练数据版本：", "Training data version: ")}
            {model.source_dataset_id && model.source_profile_id ? (
              <Link
                className="underline underline-offset-4"
                href={datasetVersionPath(
                  model.source_dataset_id,
                  model.source_profile_id,
                )}
              >
                {model.source_profile_id}
              </Link>
            ) : (
              text("未记录", "Not recorded")
            )}
          </p>
          <ModelQualityNotice status={model.quality_status} zh={zh} />
        </div>
        <div className="bg-muted/40 rounded-lg p-3 text-sm">
          <p className="text-muted-foreground mb-2 text-xs">
            {metric
              ? `${scopeLabel(metric.scope, zh)}${metric.n !== null ? ` · ${metric.n} ${text("个样本", "samples")}` : ""}`
              : text("评价指标未记录", "Metrics not recorded")}
          </p>
          {modelTask(model) === "classification" ? (
            <p>
              {text(
                "分类评价见模型报告",
                "See report for classification results",
              )}
            </p>
          ) : (
            <div className="flex gap-6 tabular-nums">
              <span>
                R² <strong>{formatProcessNumber(metric?.R2)}</strong>
              </span>
              <span>
                RMSE <strong>{formatProcessNumber(metric?.RMSE)}</strong>
              </span>
            </div>
          )}
        </div>
        <div className="flex flex-wrap gap-2 xl:justify-end">
          <Button size="sm" variant="outline" asChild>
            <Link href={modelReportPath(model)}>
              {text("查看详情", "View details")}
            </Link>
          </Button>
          <Button size="sm" asChild>
            <Link href={modelChatPath(model)}>
              {text("在新对话中使用", "Use in a new chat")}
            </Link>
          </Button>
        </div>
      </article>
    );
  }

  if (!featureLoading && !enabled)
    return (
      <FeatureDisabled
        title={t.nirLibrary.featureDisabledTitle}
        description={t.nirLibrary.featureDisabledDescription}
      />
    );
  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 p-4 sm:p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{t.nirLibrary.modelsTitle}</h1>
          <p className="text-muted-foreground mt-2 text-sm">
            {text(
              "按训练数据查找模型，查看预测目标、训练版本和评价结果。",
              "Find models by training dataset, target, training version and evaluation results.",
            )}
          </p>
        </div>
        <Button
          variant="outline"
          disabled={catalog.isPending || catalog.isFetching}
          onClick={() => {
            void catalog.refetch();
            setRefresh((value) => value + 1);
          }}
        >
          <RefreshCwIcon className="size-4" />
          {t.nirLibrary.refresh}
        </Button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative w-full max-w-lg">
          <SearchIcon
            className="text-muted-foreground absolute top-3 left-3 size-4"
            aria-hidden
          />
          <Input
            aria-label={text("搜索模型", "Search models")}
            placeholder={text(
              "搜索模型、数据集或预测目标",
              "Search models, datasets or targets",
            )}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="pl-9"
          />
        </div>
        <div
          className="flex gap-2"
          role="group"
          aria-label={text("模型展示方式", "Model view")}
        >
          <Button
            size="sm"
            variant={view === "dataset" ? "secondary" : "ghost"}
            aria-pressed={view === "dataset"}
            onClick={() => setView("dataset")}
          >
            {text("按数据集", "By dataset")}
          </Button>
          <Button
            size="sm"
            variant={view === "all" ? "secondary" : "ghost"}
            aria-pressed={view === "all"}
            onClick={() => setView("all")}
          >
            {text("全部模型", "All models")}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap gap-3">
        <Filter
          label={text("预测目标", "Target")}
          value={target}
          onChange={setTarget}
          all={text("全部目标", "All targets")}
          options={targets.map((value) => [
            value || "__unknown",
            value || text("目标未记录", "Target not recorded"),
          ])}
        />
        <Filter
          label={text("分析类型", "Analysis type")}
          value={task}
          onChange={setTask}
          all={text("全部类型", "All types")}
          options={[
            ["regression", text("定量分析", "Regression")],
            ["classification", text("分类识别", "Classification")],
            ["unknown", text("类型待确认", "Type unconfirmed")],
          ]}
        />
        <Filter
          label={text("算法", "Algorithm")}
          value={method}
          onChange={setMethod}
          all={text("全部算法", "All algorithms")}
          options={methods.map((value) => [
            value || "__unknown",
            methodLabel(value || null, zh),
          ])}
        />
        <Filter
          label={text("质量标记", "Quality flag")}
          value={quality}
          onChange={setQuality}
          all={text("全部质量标记", "All quality flags")}
          options={[
            [
              "accepted",
              text("未达门槛，经确认保存", "Below threshold, user accepted"),
            ],
            ["unmarked", text("无未达标标记", "No below-threshold flag")],
          ]}
        />
      </div>
      {datasetFilter && (
        <div className="flex flex-wrap items-center gap-3 text-sm">
          <span>
            {text("已限定训练数据集", "Filtered by training dataset")}：
            {datasetNames[datasetFilter] ?? datasetFilter}
          </span>
          <Link className="underline" href="/workspace/nir/models">
            {text("清除数据集筛选", "Clear dataset filter")}
          </Link>
        </div>
      )}
      <p className="text-muted-foreground text-sm">
        {text(
          `${filtered.length} 个已保存模型 · 按最近保存排序`,
          `${filtered.length} saved models · Most recently saved first`,
        )}
      </p>
      {catalog.isError ? (
        <div role="alert" className="rounded-xl border p-6">
          {text(
            "模型暂时未能加载，请刷新重试。",
            "Models could not be loaded. Refresh to retry.",
          )}
        </div>
      ) : featureLoading || catalog.isPending ? (
        <p className="py-12 text-center">{t.nirLibrary.loading}</p>
      ) : !filtered.length ? (
        <p className="text-muted-foreground rounded-xl border border-dashed p-12 text-center">
          {models.length
            ? text(
                "没有匹配的模型，请调整筛选。",
                "No matching models. Adjust the filters.",
              )
            : text(
                "模型保存后会出现在这里。",
                "Saved models will appear here.",
              )}
        </p>
      ) : view === "all" ? (
        <div className="overflow-hidden rounded-xl border">
          {filtered.map((model) => row(model, true))}
        </div>
      ) : (
        groups.map((group) => {
          const key = group.datasetId ?? "__unconfirmed";
          const title = groupName(group.datasetId, group.models);
          const options = [
            ...new Set(
              group.models.map(
                (model) => model.report_summary?.target?.trim() ?? "",
              ),
            ),
          ];
          const selected = groupTargets[key] ?? "";
          const effectiveTarget = options.some(
            (value) => (value || "__unknown") === selected,
          )
            ? selected
            : "";
          const visible = group.models.filter(
            (model) =>
              !effectiveTarget ||
              ((model.report_summary?.target?.trim() ?? "") || "__unknown") ===
                effectiveTarget,
          );
          return (
            <section
              key={key}
              data-testid="model-dataset-group"
              data-dataset-id={group.datasetId ?? ""}
              aria-label={title}
              className="overflow-hidden rounded-xl border"
            >
              <div className="bg-muted/30 flex flex-wrap items-center justify-between gap-3 p-4">
                <button
                  type="button"
                  className="flex min-w-0 flex-1 items-start gap-3 text-left"
                  aria-expanded={!collapsed[key]}
                  onClick={() =>
                    setCollapsed((current) => ({
                      ...current,
                      [key]: !current[key],
                    }))
                  }
                >
                  <DatabaseIcon className="text-primary mt-1 size-5 shrink-0" />
                  <span className="min-w-0">
                    <span className="block font-semibold break-words">
                      {title}
                    </span>
                    <span className="text-muted-foreground mt-1 block text-xs">
                      {text(
                        `${group.models.length} 个模型 · 最近保存 ${date(group.models[0]!.created_at)}`,
                        `${group.models.length} models · Latest ${date(group.models[0]!.created_at)}`,
                      )}
                    </span>
                    {group.datasetId && (
                      <span className="text-muted-foreground mt-1 block text-xs break-all">
                        {text("数据标识：", "Dataset ID: ")}
                        {group.datasetId.slice(-12)}
                      </span>
                    )}
                  </span>
                  <ChevronDownIcon
                    className={`ml-auto size-4 shrink-0 ${collapsed[key] ? "-rotate-90" : ""}`}
                  />
                </button>
                {group.datasetId && (
                  <Button size="sm" variant="outline" asChild>
                    <Link href={datasetVersionPath(group.datasetId)}>
                      {text("查看数据集", "View dataset")}
                    </Link>
                  </Button>
                )}
              </div>
              {!collapsed[key] && (
                <>
                  {options.length > 1 && (
                    <div className="border-t px-4 py-3">
                      <Filter
                        label={text("组内预测目标", "Target in this dataset")}
                        value={effectiveTarget}
                        onChange={(value) =>
                          setGroupTargets((current) => ({
                            ...current,
                            [key]: value,
                          }))
                        }
                        all={text("全部目标", "All targets")}
                        options={options.map((value) => [
                          value || "__unknown",
                          value || text("目标未记录", "Target not recorded"),
                        ])}
                      />
                    </div>
                  )}
                  {visible.map((model) => row(model, false))}
                </>
              )}
            </section>
          );
        })
      )}
    </div>
  );
}

function Filter({
  label,
  value,
  onChange,
  all,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  all: string;
  options: string[][];
}) {
  return (
    <label className="flex items-center gap-2 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="bg-background max-w-64 rounded-md border px-2 py-2"
      >
        <option value="">{all}</option>
        {options.map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </select>
    </label>
  );
}
