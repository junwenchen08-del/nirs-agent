"use client";

/** Product section navigation with one keyboard focus stop. */
export function SectionTabs<K extends string>({
  id,
  label,
  items,
  value,
  onChange,
}: {
  id: string;
  label: string;
  items: { key: K; label: string }[];
  value: K;
  onChange: (key: K) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label={label}
      className="flex shrink-0 flex-wrap gap-1 border-b"
    >
      {items.map((item, index) => (
        <button
          key={item.key}
          type="button"
          role="tab"
          id={`${id}-${item.key}`}
          aria-selected={value === item.key}
          aria-controls={`${id}-panel-${item.key}`}
          tabIndex={value === item.key ? 0 : -1}
          onClick={() => onChange(item.key)}
          onKeyDown={(event) => {
            const next =
              event.key === "ArrowRight"
                ? (index + 1) % items.length
                : event.key === "ArrowLeft"
                  ? (index + items.length - 1) % items.length
                  : event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? items.length - 1
                      : null;
            if (next === null) return;
            event.preventDefault();
            onChange(items[next]!.key);
            document.getElementById(`${id}-${items[next]!.key}`)?.focus();
          }}
          className={`border-b-2 px-3 py-3 text-sm ${value === item.key ? "border-primary text-primary font-medium" : "text-muted-foreground border-transparent"}`}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
