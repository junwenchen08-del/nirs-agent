import { record, text } from "./autonomous";

// Read the recorded human-facing statement, never serialize evidence references
// as prose or derive a new scientific conclusion from a technical object.
export function recordedStatements(value: unknown, depth = 0): string[] {
  if (depth > 4) return [];
  if (typeof value === "string")
    return text(value, 4000) ? [text(value, 4000)] : [];
  if (Array.isArray(value))
    return [
      ...new Set(
        value
          .slice(0, 20)
          .flatMap((item) => recordedStatements(item, depth + 1)),
      ),
    ];
  const item = record(value);
  for (const key of [
    "claim",
    "text",
    "description",
    "reason",
    "rationale",
    "statement",
    "hypothesis",
    "uncertainty",
  ]) {
    const statement = text(item[key], 4000);
    if (statement) return [statement];
  }
  return [];
}

export function hasRecordedContent(value: unknown): boolean {
  return (
    value != null &&
    value !== "" &&
    (typeof value !== "object" || Object.keys(value).length > 0)
  );
}
