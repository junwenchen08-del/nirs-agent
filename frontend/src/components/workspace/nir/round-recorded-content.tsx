import { record } from "@/core/nir/autonomous";
import { factLabel, formatProcessNumber } from "@/core/nir/process-display";
import { recordedStatements } from "@/core/nir/round-content";

export function RecordedNarrative({
  value,
  zh,
}: {
  value: string;
  zh: boolean;
}) {
  if (!value) return <span>{zh ? "未记录" : "Not recorded"}</span>;
  if (!zh || /[\u3400-\u9fff]/u.test(value))
    return <span className="whitespace-pre-wrap">{value}</span>;
  return (
    <details className="text-sm">
      <summary className="text-muted-foreground cursor-pointer">
        历史理由为英文，展开查看原始记录
      </summary>
      <p className="mt-2 whitespace-pre-wrap" lang="en">
        {value}
      </p>
    </details>
  );
}

export function RecordedStatements({
  value,
  zh,
}: {
  value: unknown;
  zh: boolean;
}) {
  const statements = recordedStatements(value);
  return statements.length ? (
    <ul className="space-y-2 text-sm leading-6">
      {statements.map((statement) => (
        <li key={statement}>
          <RecordedNarrative value={statement} zh={zh} />
        </li>
      ))}
    </ul>
  ) : null;
}

function parameterValue(value: unknown, zh: boolean): string {
  if (typeof value === "number") return formatProcessNumber(value);
  if (typeof value === "boolean")
    return value ? (zh ? "是" : "Yes") : zh ? "否" : "No";
  if (typeof value === "string") return value;
  if (Array.isArray(value))
    return value.map((part) => parameterValue(part, zh)).join("、");
  if (value === null) return zh ? "未设置" : "Unset";
  return Object.entries(record(value))
    .map(([key, item]) => `${key}: ${parameterValue(item, zh)}`)
    .join("; ");
}

export function RoundParameters({
  value,
  zh,
}: {
  value: unknown;
  zh: boolean;
}) {
  const parameters = Object.entries(record(value));
  return parameters.length ? (
    <dl
      className="grid grid-cols-[minmax(5rem,auto)_minmax(0,1fr)] gap-x-6 gap-y-2 text-sm"
      data-testid="round-parameters"
    >
      {parameters.map(([key, item]) => (
        <div key={key} className="contents">
          <dt className="text-muted-foreground">
            {factLabel(key, zh) ? `${factLabel(key, zh)} (${key})` : key}
          </dt>
          <dd className="break-words tabular-nums">
            {parameterValue(item, zh)}
          </dd>
        </div>
      ))}
    </dl>
  ) : (
    <p className="text-muted-foreground text-sm">
      {zh ? "未记录具体参数。" : "No parameters recorded."}
    </p>
  );
}
