export interface MethodKnowledgeView {
  mode: "retrieval_guided" | "rule_fallback" | "disabled" | "unknown";
  reason: string | null;
  findings: string[];
  scope: string | null;
  sources: {
    method: string;
    title: string;
    url: string | null;
    evidenceId: string | null;
    eligible: boolean | null;
  }[];
}

const record = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
const text = (value: unknown) =>
  typeof value === "string" && value.trim() ? value.trim().slice(0, 160) : null;

function sourceURL(value: unknown): string | null {
  if (typeof value !== "string" || value.length > 500) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" &&
      url.hostname === "chemotools.org" &&
      !url.username &&
      !url.password &&
      !url.port &&
      url.pathname.startsWith("/methods/")
      ? url.href
      : null;
  } catch {
    return null;
  }
}

export function selectMethodKnowledge(
  value: unknown,
): MethodKnowledgeView | null {
  const data = record(value);
  if (!data) return null;
  const mode = data.mode;
  return {
    mode:
      mode === "retrieval_guided" ||
      mode === "rule_fallback" ||
      mode === "disabled"
        ? mode
        : "unknown",
    reason: text(data.reason),
    scope: text(data.profile_scope),
    findings: Array.isArray(data.problem_labels)
      ? data.problem_labels.slice(0, 8).flatMap((item) => text(item) ?? [])
      : [],
    sources: Array.isArray(data.sources)
      ? data.sources.slice(0, 6).flatMap((item) => {
          const source = record(item);
          const method = text(source?.method_id);
          return source && method
            ? [
                {
                  method,
                  title: text(source.title) ?? method,
                  url: sourceURL(source.source_url),
                  evidenceId: text(source.evidence_id),
                  eligible:
                    typeof source.auto_eligible === "boolean"
                      ? source.auto_eligible
                      : null,
                },
              ]
            : [];
        })
      : [],
  };
}
