import { cn } from "@/lib/utils";

/** A molecular outline and spectrum identify ChemAgent at small sizes. */
export function ChemAgentMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      aria-hidden="true"
      className={cn("size-8 shrink-0", className)}
    >
      <path
        d="M24 3 42 13.5v21L24 45 6 34.5v-21Z"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <path
        d="M11 29h6l4-13 6 20 4-13 3 6h3"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
