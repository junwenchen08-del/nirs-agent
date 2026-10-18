import { useMemo } from "react";

import { BRAND } from "@/core/branding";
import { cn } from "@/lib/utils";

export type FooterProps = {
  className?: string;
};

export function Footer({ className }: FooterProps) {
  const year = useMemo(() => new Date().getFullYear(), []);
  return (
    <footer
      className={cn(
        "mx-auto mt-12 flex max-w-6xl flex-col items-center justify-center px-6",
        className,
      )}
    >
      <hr className="from-border/0 to-border/0 via-border m-0 h-px w-full border-none bg-linear-to-r" />
      <div className="text-muted-foreground container flex h-20 flex-col items-center justify-center text-sm">
        <p className="text-center text-lg md:text-xl">
          &quot;让每一次近红外分析都有证据、可复现、可追踪。&quot;
        </p>
      </div>
      <div className="text-muted-foreground container mb-8 flex flex-col items-center justify-center text-xs">
        <p>ChemAgent · 近红外光谱智能分析平台</p>
        <p>
          &copy; {year} ChemAgent ·{" "}
          <a
            className="hover:underline"
            href={BRAND.repositoryUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            ChemAgent 项目
          </a>
        </p>
      </div>
    </footer>
  );
}
