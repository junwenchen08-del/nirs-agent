"use client";

import { FlaskConicalIcon } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useArtifacts } from "@/components/workspace/artifacts";
import { useI18n } from "@/core/i18n/hooks";
import { selectNIRProcessView } from "@/core/nir";
import { cn } from "@/lib/utils";

import { NIRProcessContent } from "./nir-process-content";

export function NIRWorkflowPanel({
  workflow,
  threadId,
}: {
  workflow: unknown;
  threadId?: string;
}) {
  const { t, locale } = useI18n();
  const { artifacts, setOpen: setArtifactsOpen } = useArtifacts();
  const [open, setOpen] = useState(false);
  const view = useMemo(() => selectNIRProcessView(workflow), [workflow]);
  if (!view) return null;
  const blocked = view.base.stage === "blocked";
  const completed = ["approved", "registered", "completed"].includes(
    view.base.stage,
  );
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <button
          type="button"
          aria-label={t.nirWorkflow.open}
          title={t.nirWorkflow.open}
          className="hover:bg-accent focus-visible:ring-ring relative inline-flex h-8 items-center gap-1.5 rounded-md px-2 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:outline-none"
        >
          <FlaskConicalIcon className="size-4" />
          <span className="hidden lg:inline">NIR</span>
          <span
            className={cn(
              "size-2 rounded-full",
              blocked
                ? "bg-destructive"
                : completed
                  ? "bg-emerald-500"
                  : "bg-amber-500",
            )}
          />
        </button>
      </SheetTrigger>
      <SheetContent className="w-[min(1040px,92vw)] max-w-none gap-0 sm:max-w-[1040px]">
        <SheetHeader className="border-b pr-12">
          <SheetTitle className="flex items-center gap-2">
            <FlaskConicalIcon className="size-5" />
            {locale === "zh-CN"
              ? "ChemAgent 建模过程解释面板"
              : "ChemAgent modeling process"}
          </SheetTitle>
          <SheetDescription>{t.nirWorkflow.description}</SheetDescription>
          {threadId &&
            process.env.NEXT_PUBLIC_NIR_PROCESS_ENABLED !== "false" && (
              <Link
                href={`/workspace/nir/process/${encodeURIComponent(threadId)}`}
                onClick={() => setOpen(false)}
                className="bg-primary text-primary-foreground mt-2 inline-flex w-fit rounded-lg px-4 py-2 text-sm"
              >
                {locale === "zh-CN"
                  ? "查看完整建模过程"
                  : "Explore full modeling process"}
              </Link>
            )}
        </SheetHeader>
        <NIRProcessContent
          key={view.base.projectId}
          view={view}
          hasArtifacts={artifacts.length > 0}
          onArtifacts={() => {
            setOpen(false);
            setArtifactsOpen(true);
          }}
        />
      </SheetContent>
    </Sheet>
  );
}
