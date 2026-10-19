"use client";

export type NodeDetailTab = "overview" | "config" | "evaluation" | "reason";
const keys: NodeDetailTab[] = ["overview", "config", "evaluation", "reason"];

export function NodeDetailTabs({
  value,
  onChange,
  id,
  zh,
}: {
  value: NodeDetailTab;
  onChange: (tab: NodeDetailTab) => void;
  id: string;
  zh: boolean;
}) {
  return (
    <div
      className="flex shrink-0 flex-wrap gap-1 border-b px-4"
      role="tablist"
      aria-label={zh ? "节点信息" : "Node information"}
    >
      {keys.map((key, index) => (
        <button
          key={key}
          type="button"
          role="tab"
          id={`${id}-${key}`}
          aria-selected={value === key}
          aria-controls={`${id}-panel`}
          tabIndex={value === key ? 0 : -1}
          onClick={() => onChange(key)}
          onKeyDown={(event) => {
            const next =
              event.key === "ArrowRight"
                ? (index + 1) % keys.length
                : event.key === "ArrowLeft"
                  ? (index + keys.length - 1) % keys.length
                  : event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? keys.length - 1
                      : null;
            if (next === null) return;
            event.preventDefault();
            onChange(keys[next]!);
            document.getElementById(`${id}-${keys[next]}`)?.focus();
          }}
          className={`border-b-2 px-3 py-3 text-sm ${value === key ? "border-primary text-primary font-medium" : "text-muted-foreground border-transparent"}`}
        >
          {
            {
              overview: ["概览", "Overview"],
              config: ["参数", "Parameters"],
              evaluation: ["诊断图", "Diagnostics"],
              reason: ["LLM 建议", "LLM advice"],
            }[key][zh ? 0 : 1]
          }
        </button>
      ))}
    </div>
  );
}
