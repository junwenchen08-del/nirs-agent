"use client";

import {
  ArrowLeftIcon,
  ArrowUpRightIcon,
  DatabaseIcon,
  LinkIcon,
  RefreshCwIcon,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SectionTabs } from "@/components/workspace/section-tabs";
import { useI18n } from "@/core/i18n/hooks";
import { methodLabel } from "@/core/nir/presentation";
import { factLabel, formatProcessNumber } from "@/core/nir/process-display";
import {
  getDatasetPreview,
  getModelReport,
  useNirLibraryEnabled,
  type NIRDatasetPreview,
  type NIRModelDetail,
} from "@/core/nir-library";
import { datasetVersionPath } from "@/core/nir-library/dataset-partitions";
import {
  modelChatPath,
  modelDisplayName,
  scopeLabel,
  simpleStep,
} from "@/core/nir-library/model-report";

import { FeatureDisabled } from "./dataset-library";
import { PartitionCounts } from "./dataset-partitions";
import { DatasetPreview } from "./dataset-preview";
import { ModelQualityNotice } from "./model-quality-notice";
import { PredictionFigure, SpectraFigure } from "./model-report-charts";

export function ModelDetail({
  modelId,
  version,
}: {
  modelId: string;
  version: string;
}) {
  const { locale, t } = useI18n();
  const zh = locale.startsWith("zh");
  const text = (cn: string, en: string) => (zh ? cn : en);
  const { enabled, isLoading: featureLoading } = useNirLibraryEnabled();
  const [detail, setDetail] = useState<NIRModelDetail | null>(null);
  const [preview, setPreview] = useState<NIRDatasetPreview | null>(null);
  const [previewError, setPreviewError] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [datasetRefresh, setDatasetRefresh] = useState(0);
  const [section, setSection] = useState<"results" | "data" | "methods">(
    "results",
  );
  const [scope, setScope] = useState("");
  useEffect(() => {
    setSection("results");
    setScope("");
  }, [modelId, version]);

  useEffect(() => {
    if (featureLoading || !enabled) return;
    const controller = new AbortController();
    setLoading(true);
    setError(false);
    setDetail(null);
    void getModelReport(modelId, version, controller.signal)
      .then((result) => {
        if (!controller.signal.aborted) setDetail(result);
      })
      .catch(() => {
        if (!controller.signal.aborted) setError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [modelId, version, refresh, enabled, featureLoading]);

  useEffect(() => {
    setPreview(null);
    setPreviewError(false);
    if (!detail?.dataset) return;
    const dataset = detail.dataset;
    const controller = new AbortController();
    setPreviewLoading(true);
    void getDatasetPreview(dataset.id, controller.signal, dataset.profile_id)
      .then((result) => {
        if (
          result.dataset_id !== dataset.id ||
          result.source_sha256 !== dataset.source_sha256 ||
          (dataset.profile_id && result.profile_id !== dataset.profile_id)
        )
          throw new Error("Dataset preview identity mismatch");
        if (!controller.signal.aborted) setPreview(result);
      })
      .catch(() => {
        if (!controller.signal.aborted) setPreviewError(true);
      })
      .finally(() => {
        if (!controller.signal.aborted) setPreviewLoading(false);
      });
    return () => controller.abort();
  }, [detail, datasetRefresh]);

  if (!featureLoading && !enabled)
    return (
      <FeatureDisabled
        title={t.nirLibrary.featureDisabledTitle}
        description={t.nirLibrary.featureDisabledDescription}
      />
    );
  const parameterName = (key: string) =>
    ({
      n_components: text("使用的成分数量", "Number of components"),
      window_length: text("平滑窗口大小", "Smoothing window"),
      polyorder: text("平滑曲线阶数", "Polynomial order"),
      C: text("误差惩罚强度 C", "Error penalty C"),
      gamma: text("曲线影响范围 γ", "Kernel gamma"),
      epsilon: text("允许的小误差 ε", "Error tolerance ε"),
      alpha: text("约束强度 α", "Regularization α"),
    })[key] ??
    factLabel(key, zh) ??
    key;
  function parameters(
    values: Record<string, string | number | boolean | null>,
    isStep = false,
  ) {
    const entries = Object.entries(values);
    return entries.length ? (
      <dl className="mt-3 grid gap-2 sm:grid-cols-2">
        {entries.map(([key, value]) => (
          <div key={key} className="bg-muted/40 rounded-lg px-3 py-2 text-sm">
            <dt className="text-muted-foreground text-xs">
              {parameterName(key)}
            </dt>
            <dd className="mt-1 font-medium break-words">
              {typeof value === "number"
                ? formatProcessNumber(value)
                : typeof value === "boolean"
                  ? value
                    ? text("是", "Yes")
                    : text("否", "No")
                  : String(value ?? "—")}
            </dd>
          </div>
        ))}
      </dl>
    ) : (
      <p className="text-muted-foreground mt-2 text-sm">
        {text(
          isStep ? "未记录额外参数。" : "当时未保存具体参数。",
          isStep
            ? "No additional parameters were recorded."
            : "Detailed parameters were not saved.",
        )}
      </p>
    );
  }
  const report = detail?.report;
  const scopes = [
    ...new Set([
      ...(report?.metrics.map((row) => row.scope) ?? []),
      ...(report?.charts
        .filter((chart) => chart.type === "prediction")
        .map((row) =>
          row.scope === "holdout"
            ? "final_test"
            : row.scope === "tuning"
              ? "validation"
              : row.scope,
        ) ?? []),
    ]),
  ];
  const currentScope = scopes.includes(scope)
    ? scope
    : scopes.includes("final_test")
      ? "final_test"
      : scopes.includes("validation")
        ? "validation"
        : (scopes[0] ?? "");
  const predictions =
    report?.charts.filter(
      (chart) =>
        chart.type === "prediction" &&
        (chart.scope === "holdout"
          ? "final_test"
          : chart.scope === "tuning"
            ? "validation"
            : chart.scope) === currentScope,
    ) ?? [];
  const spectra =
    report?.charts.filter((chart) => chart.type === "spectra") ?? [];
  return (
    <div className="mx-auto w-full max-w-6xl space-y-6 p-4 sm:p-6">
      <Button asChild variant="ghost" size="sm">
        <Link href="/workspace/nir/models">
          <ArrowLeftIcon className="size-4" />
          {text("返回模型库", "Back to models")}
        </Link>
      </Button>
      {error ? (
        <div role="alert" className="rounded-xl border p-8">
          <p>
            {text(
              "模型详情暂时无法读取，请重试。",
              "Model details could not be loaded. Please retry.",
            )}
          </p>
          <Button
            className="mt-4"
            variant="outline"
            onClick={() => setRefresh((v) => v + 1)}
          >
            <RefreshCwIcon className="size-4" />
            {text("重新加载", "Retry")}
          </Button>
        </div>
      ) : loading || featureLoading || !detail || !report ? (
        <p className="text-muted-foreground p-12 text-center">
          {t.nirLibrary.loading}
        </p>
      ) : (
        <>
          <header className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0 space-y-2">
              <div className="flex flex-wrap items-center gap-3">
                <h1 className="text-2xl font-semibold break-words">
                  {modelDisplayName(detail.model, zh)}
                </h1>
                <Badge variant="secondary">
                  {methodLabel(report.method ?? detail.model.method, zh)}
                </Badge>
              </div>
              <p className="text-muted-foreground text-sm">
                {report.target
                  ? text(
                      `预测目标：${report.target}${report.unit ? `（${report.unit}）` : ""}`,
                      `Prediction target: ${report.target}${report.unit ? ` (${report.unit})` : ""}`,
                    )
                  : text(
                      "查看这个模型使用的数据、处理步骤和已保存的预测结果。",
                      "Explore the data, processing steps and saved prediction results.",
                    )}
              </p>
            </div>
            {detail.model.status === "ready" && (
              <Button asChild>
                <Link href={modelChatPath(detail.model)}>
                  <LinkIcon className="size-4" />
                  {text("在新对话中使用", "Use in a new chat")}
                </Link>
              </Button>
            )}
          </header>
          <ModelQualityNotice status={detail.model.quality_status} zh={zh} />
          <SectionTabs
            id="model-sections"
            label={text("模型详情目录", "Model report sections")}
            value={section}
            onChange={setSection}
            items={[
              { key: "results", label: text("预测效果", "Prediction results") },
              { key: "data", label: text("训练数据", "Training data") },
              { key: "methods", label: text("处理方法", "Processing steps") },
            ]}
          />
          <section
            id="model-sections-panel-results"
            role="tabpanel"
            aria-labelledby="model-sections-results"
            tabIndex={0}
            hidden={section !== "results"}
            className="space-y-4"
          >
            <div>
              <h2 className="text-lg font-semibold">
                {text("预测效果", "Prediction results")}
              </h2>
              <p className="text-muted-foreground mt-1 text-sm">
                {text(
                  "R² 越接近 1，整体越贴近真实值；RMSE 越小，预测误差越小。",
                  "R² closer to 1 indicates a closer overall fit; lower RMSE indicates smaller prediction errors.",
                )}
              </p>
            </div>
            {scopes.length > 0 && (
              <label className="flex flex-wrap items-center gap-3 text-sm">
                {text("评价范围", "Evaluation scope")}
                <select
                  className="bg-card rounded-md border p-2"
                  aria-label={text("评价范围", "Evaluation scope")}
                  value={currentScope}
                  onChange={(event) => setScope(event.target.value)}
                >
                  {scopes.map((key) => (
                    <option key={key} value={key}>
                      {scopeLabel(key, zh)}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {report.metrics.length ? (
              <div className="grid gap-3">
                {report.metrics
                  .filter((metric) => metric.scope === currentScope)
                  .map((metric) => (
                    <div className="rounded-xl border p-4" key={metric.scope}>
                      <p className="text-sm font-medium">
                        {scopeLabel(metric.scope, zh)}
                        {metric.n !== null
                          ? ` · ${metric.n} ${text("个样本", "samples")}`
                          : ""}
                      </p>
                      <div className="mt-4 grid grid-cols-2 gap-3">
                        <div>
                          <p className="text-muted-foreground text-xs">
                            {text("贴合程度 R²", "Fit R²")}
                          </p>
                          <p className="mt-1 text-2xl font-semibold tabular-nums">
                            {formatProcessNumber(metric.R2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-muted-foreground text-xs">
                            {text("预测误差 RMSE", "Prediction error RMSE")}
                          </p>
                          <p className="mt-1 text-2xl font-semibold tabular-nums">
                            {formatProcessNumber(metric.RMSE)}
                          </p>
                        </div>
                      </div>
                      {metric.MAE !== null && (
                        <details className="mt-3 text-sm">
                          <summary className="cursor-pointer">
                            {text("其他已记录指标", "Other recorded metrics")}
                          </summary>
                          <p className="mt-2">
                            {text(
                              "平均绝对误差 MAE",
                              "Mean absolute error MAE",
                            )}
                            ：{formatProcessNumber(metric.MAE)}
                          </p>
                        </details>
                      )}
                      <p className="text-muted-foreground mt-3 text-xs">
                        {metric.scope === "calibration"
                          ? text(
                              "模型学习过的数据，不能单凭这项判断预测效果。",
                              "Used to fit the model; this alone cannot establish prediction performance.",
                            )
                          : metric.scope === "validation"
                            ? text(
                                "用于选择处理方法和参数的数据。",
                                "Used to choose processing steps and parameters.",
                              )
                            : text(
                                "用于检查模型预测效果的数据。",
                                "Used to check prediction performance.",
                              )}
                      </p>
                    </div>
                  ))}
              </div>
            ) : (
              <Missing
                text={text(
                  "这个模型当时未保存评价指标。",
                  "Evaluation metrics were not saved for this model.",
                )}
              />
            )}
            <p className="text-muted-foreground text-xs">
              {detail.model.validation_scope ===
              "independent_external_validation"
                ? text(
                    "评价记录来自独立的外部验证数据。",
                    "Evaluation used independent external validation data.",
                  )
                : text(
                    "评价来自内部划分的数据，未作为外部数据验证。",
                    "Evaluation used an internal data split, not external validation.",
                  )}{" "}
              {text(
                "实际使用效果还取决于新数据是否适合这个模型。",
                "Performance on new data depends on its compatibility with this model.",
              )}
            </p>
            {predictions.length ? (
              <div className="grid gap-4">
                {predictions.map((chart, i) => (
                  <PredictionFigure
                    key={i}
                    chart={chart}
                    metrics={report.metrics}
                    method={(
                      report.method ??
                      detail.model.method ??
                      ""
                    ).toUpperCase()}
                    zh={zh}
                  />
                ))}
              </div>
            ) : (
              <Missing
                text={text(
                  "当时未保存逐样本的预测结果，因此无法显示真实值与预测值对比图。已有指标仍可在上方查看。",
                  "Per-sample predictions were not saved, so a comparison chart is unavailable. Recorded metrics are shown above.",
                )}
              />
            )}
          </section>
          <section
            id="model-sections-panel-data"
            role="tabpanel"
            aria-labelledby="model-sections-data"
            tabIndex={0}
            hidden={section !== "data"}
            className="space-y-4"
          >
            <h2 className="flex items-center gap-2 text-lg font-semibold">
              <DatabaseIcon className="size-5" />
              {text("使用的数据", "Training data")}
            </h2>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              {[
                [
                  text("训练记录中的样本数", "Recorded samples"),
                  report.sample_count,
                ],
                [
                  text("原始光谱变量数", "Original spectral variables"),
                  report.feature_count,
                ],
                [
                  text("模型使用的变量数", "Model spectral variables"),
                  report.model_feature_count,
                ],
              ].map(([label, value]) => (
                <div key={String(label)} className="bg-muted/40 rounded-lg p-3">
                  <p className="text-muted-foreground text-xs">{label}</p>
                  <p className="mt-1 text-xl font-semibold">
                    {formatProcessNumber(value)}
                  </p>
                </div>
              ))}
            </div>
            {detail.dataset ? (
              <div className="space-y-4 rounded-xl border p-4 sm:p-5">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h3 className="font-medium">{detail.dataset.name}</h3>
                    <p className="text-muted-foreground mt-1 text-xs">
                      {detail.dataset.partition
                        ? text(
                            `关联本次建模的固定划分版本 v${detail.dataset.partition.version}，样本用途与顺序保持不变。`,
                            `Linked to the exact modeling partition v${detail.dataset.partition.version}.`,
                          )
                        : detail.dataset.profile_id
                          ? text(
                              "下方复用数据集库的概览，使用模型训练时保存的解析版本。",
                              "This dataset overview uses the parsing version recorded at training.",
                            )
                          : text(
                              "下方展示库内数据概览。训练时未单独记录解析设置，当前设置可能与训练时不同。",
                              "No training parsing version was saved. This current library overview may differ from training settings.",
                            )}
                    </p>
                  </div>
                  <Button asChild variant="outline" size="sm">
                    <Link
                      href={datasetVersionPath(
                        detail.dataset.id,
                        detail.dataset.profile_id,
                      )}
                    >
                      {detail.dataset.partition
                        ? text("查看训练时的划分", "Open modeling partition")
                        : text("查看完整数据集", "Open dataset")}
                      <ArrowUpRightIcon className="size-4" />
                    </Link>
                  </Button>
                </div>
                {detail.dataset.partition && (
                  <PartitionCounts
                    partition={detail.dataset.partition}
                    zh={zh}
                  />
                )}
                {previewLoading ? (
                  <p className="text-muted-foreground text-sm">
                    {text(
                      "正在读取数据特征…",
                      "Loading dataset characteristics…",
                    )}
                  </p>
                ) : preview ? (
                  <DatasetPreview preview={preview} />
                ) : previewError ? (
                  <div role="alert" className="space-y-3 text-sm">
                    <p>
                      {text(
                        "数据特征暂时无法读取，模型和预测结果仍可查看。",
                        "Dataset characteristics could not be loaded. Model results remain available.",
                      )}
                    </p>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setDatasetRefresh((v) => v + 1)}
                    >
                      {text("重试数据预览", "Retry dataset preview")}
                    </Button>
                  </div>
                ) : null}
              </div>
            ) : (
              <Missing
                text={text(
                  "这个模型未记录可核实的数据集关联。已保存的训练信息显示在上方。",
                  "No verified library dataset link was recorded for this model. Saved training information is shown above.",
                )}
              />
            )}
          </section>
          <section
            id="model-sections-panel-methods"
            role="tabpanel"
            aria-labelledby="model-sections-methods"
            tabIndex={0}
            hidden={section !== "methods"}
            className="space-y-4"
          >
            <div>
              <h2 className="text-lg font-semibold">
                {text("处理方法", "Processing steps")}
              </h2>
              <p className="text-muted-foreground mt-1 text-sm">
                {text(
                  "以下为训练时实际保存的处理顺序和参数，使用模型时自动沿用，无需重新设置。",
                  "These recorded processing steps and parameters are reused automatically when applying the model.",
                )}
              </p>
            </div>
            {report.preprocessing === null ? (
              <Missing
                text={text(
                  "当时未保存光谱处理步骤。",
                  "Spectral processing steps were not recorded.",
                )}
              />
            ) : report.preprocessing.length === 0 ? (
              <div className="rounded-xl border p-4 text-sm">
                {text(
                  "直接使用原始光谱，未添加光谱预处理。",
                  "Original spectra were used without additional preprocessing.",
                )}
              </div>
            ) : (
              <ol className="space-y-3">
                {report.preprocessing.map((step, i) => (
                  <li className="rounded-xl border p-4" key={i}>
                    <div className="flex items-center gap-3">
                      <span className="bg-primary/10 text-primary flex size-7 shrink-0 items-center justify-center rounded-full text-sm font-semibold">
                        {i + 1}
                      </span>
                      <h3 className="font-medium">
                        {simpleStep(step.method, zh) ??
                          methodLabel(step.method, zh)}
                      </h3>
                    </div>
                    {parameters(step.params, true)}
                  </li>
                ))}
              </ol>
            )}
            {spectra.length ? (
              <>
                <div className="grid gap-4 lg:grid-cols-2">
                  {spectra.map((chart, i) => (
                    <SpectraFigure key={i} chart={chart} zh={zh} />
                  ))}
                </div>
                <p className="text-muted-foreground text-xs">
                  {text(
                    "同色曲线表示同一个训练样本。处理可能改变数值尺度，两幅图的纵轴范围请分别查看。",
                    "Matching colors represent the same training sample. Processing may change the scale, so check each vertical axis.",
                  )}
                </p>
              </>
            ) : (
              <Missing
                text={text(
                  "当时未保存处理前后的光谱曲线，暂时无法展示对比图。",
                  "Before-and-after spectral curves were not saved for this model.",
                )}
              />
            )}
            <div className="rounded-xl border p-4">
              <h3 className="font-medium">
                {text("预测方法", "Prediction method")} ·{" "}
                {methodLabel(report.method ?? detail.model.method, zh)}
              </h3>
              {parameters(report.parameters)}
            </div>
          </section>
          <details className="rounded-xl border p-4 text-sm">
            <summary className="cursor-pointer font-medium">
              {text("模型版本与来源", "Version and source")}
            </summary>
            <dl className="mt-4 grid gap-3 sm:grid-cols-2">
              {[
                [text("模型标识", "Model ID"), detail.model.model_id],
                [text("版本", "Version"), detail.model.version],
                [
                  text("保存时间", "Saved"),
                  detail.model.created_at
                    ? new Date(detail.model.created_at).toLocaleString(
                        zh ? "zh-CN" : "en-US",
                      )
                    : "—",
                ],
                [
                  text("来源对话编号", "Source chat ID"),
                  detail.model.source_thread_id,
                ],
                [
                  text("数据解析版本", "Dataset parsing version"),
                  detail.model.source_profile_id ??
                    text("未记录", "Not recorded"),
                ],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt className="text-muted-foreground text-xs">{label}</dt>
                  <dd className="mt-1 break-all">{value}</dd>
                </div>
              ))}
            </dl>
          </details>
        </>
      )}
    </div>
  );
}

function Missing({ text }: { text: string }) {
  return (
    <p className="text-muted-foreground rounded-xl border border-dashed p-4 text-sm leading-relaxed">
      {text}
    </p>
  );
}
