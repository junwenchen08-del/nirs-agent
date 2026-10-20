"use client";

export type NodeDetailTab = "overview" | "config" | "evaluation" | "reason";
const keys: NodeDetailTab[] = ["overview", "config", "evaluation", "reason"];
export type DetailTabOption = { key: NodeDetailTab; label: string };

export function NodeDetailTabs({
  value,
  onChange,
  id,
  zh,
  options,
}: {
  value: NodeDetailTab;
  onChange: (tab: NodeDetailTab) => void;
  id: string;
  zh: boolean;
  options?: DetailTabOption[];
}) {
  const tabs =
    options ??
    keys.map((key) => ({
      key,
      label: {
        overview: ["本轮变化", "Round changes"],
        config: ["实际方案", "Executed plan"],
        evaluation: ["结果与诊断", "Results and diagnostics"],
        reason: ["决策与建议", "Decisions and advice"],
      }[key][zh ? 0 : 1]!,
    }));
  return (
    <div
      className="flex shrink-0 flex-wrap gap-1 border-b px-4"
      role="tablist"
      aria-label={zh ? "节点信息" : "Node information"}
    >
      {tabs.map(({ key, label }, index) => (
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
                ? (index + 1) % tabs.length
                : event.key === "ArrowLeft"
                  ? (index + tabs.length - 1) % tabs.length
                  : event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? tabs.length - 1
                      : null;
            if (next === null) return;
            event.preventDefault();
            onChange(tabs[next]!.key);
            document.getElementById(`${id}-${tabs[next]!.key}`)?.focus();
          }}
          className={`border-b-2 px-3 py-3 text-sm ${value === key ? "border-primary text-primary font-medium" : "text-muted-foreground border-transparent"}`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
