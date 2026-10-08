import { selectionExplanation } from "@/core/nir/presentation";
import {
  factLabel,
  factValue,
  formatProcessNumber,
  isTechnicalFact,
} from "@/core/nir/process-display";

export function ProcessFacts({ value, zh }: { value: unknown; zh: boolean }) {
  if (
    value == null ||
    (typeof value === "object" && Object.keys(value).length === 0)
  )
    return (
      <p className="text-muted-foreground text-sm">
        {zh ? "没有对应记录" : "No recorded facts"}
      </p>
    );
  if (typeof value !== "object")
    return (
      <span className="break-words">
        {typeof value === "number"
          ? formatProcessNumber(value)
          : typeof value === "boolean"
            ? value
              ? zh
                ? "是"
                : "Yes"
              : zh
                ? "否"
                : "No"
            : typeof value === "string"
              ? value
              : "—"}
      </span>
    );
  if (Array.isArray(value))
    return (
      <ol className="space-y-2">
        {value.map((item, index) => (
          <li key={index} className="rounded-md border p-2">
            <span className="text-muted-foreground mr-2">{index + 1}.</span>
            <ProcessFacts value={item} zh={zh} />
          </li>
        ))}
      </ol>
    );
  const record = value as Record<string, unknown>;
  const entries = Object.entries(record).filter(([, item]) => item != null);
  const technical = entries.filter(
    ([key]) =>
      isTechnicalFact(key) ||
      (key === "score" && record.cv_rmse === record.score),
  );
  const visible = entries.filter(
    ([key]) => !technical.some(([name]) => name === key),
  );
  const reasonCode =
    typeof record.reason_code === "string" ? record.reason_code : null;
  return (
    <div className="min-w-0 space-y-3">
      {typeof record.error === "string" && (
        <p className="text-sm text-amber-700">
          {zh
            ? "此方案执行失败，无法参与性能比较；原始错误见技术记录。"
            : "This option failed and cannot be compared; see technical records for the original error."}
        </p>
      )}
      {reasonCode && (
        <p className="text-sm leading-6">
          {selectionExplanation(
            "model",
            {
              reasonCode,
              comparisonReasonCode: null,
              reason: typeof record.reason === "string" ? record.reason : null,
            },
            zh,
          )}
        </p>
      )}
      <dl className="space-y-2">
        {visible.map(([key, item]) => (
          <div key={key} className="border-b pb-2 last:border-0">
            <dt className="text-muted-foreground mb-1 text-xs">
              {factLabel(key, zh)}
            </dt>
            <dd className="text-sm">
              <ProcessFacts value={factValue(key, item, zh)} zh={zh} />
            </dd>
          </div>
        ))}
      </dl>
      {technical.length > 0 && (
        <details
          className="text-muted-foreground min-w-0 text-xs"
          data-testid="process-technical-record"
        >
          <summary className="cursor-pointer">
            {zh
              ? "查看技术记录（含内部编号）"
              : "Technical records (including internal IDs)"}
          </summary>
          {record.candidate_id != null && (
            <p className="mt-2">
              {zh
                ? "内部编号用于关联方案与评估记录，不代表排名或质量。"
                : "Internal IDs link configurations and evaluation records; they do not represent rank or quality."}
            </p>
          )}
          <pre className="mt-2 max-h-80 overflow-auto break-all whitespace-pre-wrap">
            {JSON.stringify(record, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}
