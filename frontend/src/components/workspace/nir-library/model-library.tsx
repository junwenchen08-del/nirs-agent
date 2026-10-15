"use client";

import {
  ArrowUpRightIcon,
  BoxesIcon,
  DatabaseIcon,
  LinkIcon,
  RefreshCwIcon,
  SearchIcon,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useI18n } from "@/core/i18n/hooks";
import { methodLabel } from "@/core/nir/presentation";
import { formatProcessNumber } from "@/core/nir/process-display";
import {
  listDatasets,
  listModels,
  useNirLibraryEnabled,
  type NIRModelVersion,
} from "@/core/nir-library";
import {
  cardMetrics,
  modelChatPath,
  modelReportPath,
  scopeLabel,
} from "@/core/nir-library/model-report";

import { FeatureDisabled } from "./dataset-library";

export function ModelLibrary() {
  const { t, locale } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [models, setModels] = useState<NIRModelVersion[]>([]);
  const [datasetNames, setDatasetNames] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [query, setQuery] = useState("");
  const [refresh, setRefresh] = useState(0);

  useEffect(() => {
    if (featureLoading || !enabled) return;
    const controller = new AbortController();
    setLoading(true);
    setError(false);
    void listModels(controller.signal)
      .then((rows) => {
        if (!controller.signal.aborted)
          setModels(rows.filter((row) => row.status === "ready"));
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    void listDatasets(controller.signal)
      .then((rows) => {
        if (!controller.signal.aborted)
          setDatasetNames(
            Object.fromEntries(rows.map((row) => [row.id, row.name])),
          );
      })
      .catch(() => {
        /* Models remain usable when dataset names are unavailable. */
      });
    return () => controller.abort();
  }, [enabled, featureLoading, refresh]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return models.filter((model) =>
      [
        model.model_id,
        model.version,
        model.method ?? "",
        model.report_summary?.target ?? "",
        datasetNames[model.source_dataset_id ?? ""] ?? "",
        methodLabel(model.method, zh),
      ].some((value) => value.toLowerCase().includes(needle)),
    );
  }, [models, query, datasetNames, zh]);

  if (!featureLoading && !enabled)
    return (
      <FeatureDisabled
        title={t.nirLibrary.featureDisabledTitle}
        description={t.nirLibrary.featureDisabledDescription}
      />
    );

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-4 sm:p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">{t.nirLibrary.modelsTitle}</h1>
          <p className="text-muted-foreground mt-2 text-sm">
            {text(
              "查看模型用过的数据、处理方法和预测效果，也可以带到新对话中使用。",
              "Explore training data, processing steps and prediction results, or use a model in a new chat.",
            )}
          </p>
        </div>
        <Button
          variant="outline"
          disabled={loading}
          onClick={() => setRefresh((value) => value + 1)}
        >
          <RefreshCwIcon className="size-4" />
          {t.nirLibrary.refresh}
        </Button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="relative w-full max-w-lg">
          <SearchIcon
            aria-hidden
            className="text-muted-foreground absolute top-3 left-3 size-4"
          />
          <Input
            aria-label={text("搜索模型", "Search models")}
            placeholder={text(
              "搜索模型、数据集或预测目标",
              "Search models, datasets or targets",
            )}
            className="pl-9"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
        <p className="text-muted-foreground text-sm">
          {text(
            `${filtered.length} 个可用模型`,
            `${filtered.length} available models`,
          )}
        </p>
      </div>
      {error ? (
        <div role="alert" className="rounded-xl border p-6 text-sm">
          <p>
            {text(
              "模型暂时未能加载，请重试。",
              "Models could not be loaded. Please retry.",
            )}
          </p>
          <Button
            variant="outline"
            className="mt-3"
            onClick={() => setRefresh((value) => value + 1)}
          >
            {text("重新加载", "Retry")}
          </Button>
        </div>
      ) : loading || featureLoading ? (
        <p className="text-muted-foreground py-12 text-center">
          {t.nirLibrary.loading}
        </p>
      ) : !filtered.length ? (
        <div className="text-muted-foreground rounded-xl border border-dashed p-12 text-center">
          <BoxesIcon className="mx-auto mb-3 size-8" />
          <p>
            {query
              ? text(
                  "没有找到匹配的模型，试试其他关键词。",
                  "No matching models. Try another search.",
                )
              : text(
                  "模型保存后会出现在这里。",
                  "Saved models will appear here.",
                )}
          </p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {filtered.map((model) => {
            const metrics = cardMetrics(model);
            const metric =
              metrics.find(
                (row) =>
                  row.scope === "final_test" &&
                  (row.R2 !== null || row.RMSE !== null),
              ) ?? metrics.find((row) => row.scope === "validation");
            const datasetName = datasetNames[model.source_dataset_id ?? ""];
            return (
              <Card
                key={model.id}
                data-testid="nir-model-card"
                className="group relative gap-4 transition-shadow hover:shadow-md"
              >
                <CardHeader className="gap-3 pb-0">
                  <div className="flex items-center justify-between gap-3">
                    <div className="bg-primary/5 rounded-lg p-2.5">
                      <BoxesIcon className="text-primary size-5" />
                    </div>
                    <Badge variant="secondary">{text("可使用", "Ready")}</Badge>
                  </div>
                  <CardTitle className="min-w-0 text-base">
                    <Link
                      href={modelReportPath(model)}
                      className="break-words after:absolute after:inset-0 focus-visible:outline-2 focus-visible:outline-offset-4"
                    >
                      {model.model_id}
                    </Link>
                  </CardTitle>
                  <p className="text-muted-foreground text-sm">
                    {methodLabel(model.method, zh)}
                    {model.report_summary?.target
                      ? ` · ${model.report_summary.target}`
                      : ""}
                  </p>
                </CardHeader>
                <CardContent className="flex flex-1 flex-col gap-4 text-sm">
                  <p className="text-muted-foreground flex items-start gap-2">
                    <DatabaseIcon className="mt-0.5 size-4 shrink-0" />
                    <span className="line-clamp-2">
                      {datasetName ??
                        text(
                          model.source_dataset_id
                            ? "已关联训练数据集"
                            : "早期记录未关联数据集",
                          model.source_dataset_id
                            ? "Linked training dataset"
                            : "No dataset linked in this older record",
                        )}
                    </span>
                  </p>
                  <div className="bg-muted/40 rounded-xl p-3">
                    <p className="text-muted-foreground mb-2 text-xs">
                      {metric
                        ? scopeLabel(metric.scope, zh)
                        : text("预测效果", "Prediction results")}
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-muted-foreground text-xs">
                          {text("贴合程度 R²", "Fit R²")}
                        </p>
                        <p className="mt-1 text-xl font-semibold tabular-nums">
                          {formatProcessNumber(metric?.R2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-muted-foreground text-xs">
                          {text("预测误差 RMSE", "Prediction error RMSE")}
                        </p>
                        <p className="mt-1 text-xl font-semibold tabular-nums">
                          {formatProcessNumber(metric?.RMSE)}
                        </p>
                      </div>
                    </div>
                  </div>
                  <p className="text-muted-foreground text-xs">
                    {text(
                      "点击卡片查看完整数据、方法和图表",
                      "Open the card for data, methods and charts",
                    )}
                  </p>
                  <div className="relative z-10 mt-auto flex flex-wrap gap-2 border-t pt-4">
                    <Button size="sm" asChild>
                      <Link href={modelChatPath(model)}>
                        <LinkIcon className="size-4" />
                        {text("挂载到对话", "Attach to chat")}
                      </Link>
                    </Button>
                    <Button size="sm" variant="ghost" asChild>
                      <Link href={modelReportPath(model)}>
                        {text("查看详情", "View details")}
                        <ArrowUpRightIcon className="size-4" />
                      </Link>
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
