import { record, text, type AutonomousSummary } from "@/core/nir/autonomous";
import { formatProcessNumber as fmt } from "@/core/nir/process-display";
import { type LifecycleNode } from "@/core/nir/process-lifecycle";

export function ProcessDataOverview({
  summary,
  node,
  zh,
}: {
  summary: AutonomousSummary;
  node: LifecycleNode;
  zh: boolean;
}) {
  const context = summary.data_context;
  const partitions = record(context.partition_counts);
  const intake = node.id === "input" ? node.facts : {};
  const sourceFile = record(context.source_file);
  const hasSource = sourceFile.evidence_status === "recorded";
  const missing = zh ? "未记录" : "Not recorded";
  const source =
    (hasSource ? text(sourceFile.name, 255) : "") ||
    text(intake.filename, 200) ||
    text(intake.source_name, 200);
  const target =
    text(summary.task.target_name, 120) ||
    text(context.target_name, 120) ||
    text(summary.task.analyte, 120);
  const fields = [
    [
      zh ? "数据来源" : "Data source",
      source || (zh ? "来源记录未提供" : "Source not recorded"),
    ],
    [zh ? "分析目标" : "Target", target || missing],
    [zh ? "Cal 样本数" : "Cal samples", fmt(context.calibration_samples)],
    [zh ? "光谱变量数" : "Spectral variables", fmt(context.n_features)],
    [zh ? "目标单位" : "Target unit", text(context.unit, 60) || missing],
    [
      zh ? "文件格式" : "File format",
      hasSource ? text(sourceFile.format, 30) || missing : missing,
    ],
    [
      zh ? "文件大小（字节）" : "File size (bytes)",
      hasSource && typeof sourceFile.size_bytes === "number"
        ? sourceFile.size_bytes.toLocaleString(zh ? "zh-CN" : "en-US")
        : missing,
    ],
    [
      zh ? "源文件校验标识" : "Source fingerprint",
      text(hasSource ? sourceFile.sha256 : intake.source_sha256, 64).slice(
        0,
        12,
      ) || missing,
    ],
  ];
  return (
    <section className="space-y-4" data-testid="process-data-overview">
      <div>
        <h4 className="font-medium">
          {zh ? "本次建模使用的数据" : "Data used for this run"}
        </h4>
        <p className="text-muted-foreground mt-1 text-xs">
          {zh
            ? "规模与来源来自本次搜索准备时的固定记录；来源缺失不表示没有上传文件。"
            : "Dimensions and source use the fixed search-preparation receipt. Missing intake records do not mean no file was uploaded."}
        </p>
      </div>
      <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {fields.map(([label, value]) => (
          <div
            key={label}
            className="bg-muted/30 min-w-0 rounded-lg border p-3"
          >
            <dt className="text-muted-foreground text-xs">{label}</dt>
            <dd className="mt-2 text-sm font-medium break-words">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="grid gap-3 sm:grid-cols-3">
        {[
          {
            key: "calibration",
            alias: "cal",
            title: "Cal",
            purpose: zh
              ? "模型拟合；交叉验证在 Cal 内进行"
              : "Model fitting; cross-validation stays within Cal",
          },
          {
            key: "validation",
            alias: "val",
            title: "Val",
            purpose: zh
              ? "用于搜索和比较方案"
              : "Used to search and compare plans",
          },
          {
            key: "test",
            alias: "test",
            title: "Test",
            purpose: zh
              ? "搜索阶段封存；最终评价单独展示"
              : "Sealed during search; final evaluation is separate",
          },
        ].map(({ key, alias, title, purpose }) => (
          <div key={key} className="rounded-lg border p-3">
            <div className="flex items-baseline justify-between gap-2">
              <strong className="text-sm">{title}</strong>
              <span className="text-lg tabular-nums">
                {fmt(partitions[key] ?? partitions[alias])}
              </span>
            </div>
            <p className="text-muted-foreground mt-2 text-xs">{purpose}</p>
          </div>
        ))}
      </div>
      <p className="text-muted-foreground text-xs">
        {zh
          ? "Cal / Val 已参与建模与选择；内部留出不代表外部验证。光谱预览只展示已保存的 Cal 曲线。"
          : "Cal / Val participate in fitting and selection. Internal holdout is not external validation. Spectral previews use saved Cal curves only."}
      </p>
    </section>
  );
}
