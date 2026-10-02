"use client";

import {
  ActivityIcon,
  CheckCircle2Icon,
  DatabaseIcon,
  FlaskConicalIcon,
  ShieldCheckIcon,
} from "lucide-react";
import { useMemo } from "react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { useI18n } from "@/core/i18n/hooks";
import { selectNIRWorkflowView } from "@/core/nir";
import { cn } from "@/lib/utils";

interface NIRWorkflowPanelProps {
  workflow: unknown;
}

function translated(
  values: Record<string, string>,
  key: string | null,
  fallback: string,
) {
  return key ? (values[key] ?? key) : fallback;
}

function Section({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof ActivityIcon;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border-border/70 bg-muted/20 rounded-xl border p-3">
      <h3 className="text-muted-foreground mb-2 flex items-center gap-2 text-xs font-medium tracking-wide uppercase">
        <Icon className="size-3.5" />
        {title}
      </h3>
      {children}
    </section>
  );
}

function Definition({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] gap-3 py-1 text-sm">
      <dt className="text-muted-foreground truncate">{label}</dt>
      <dd className="truncate text-right font-medium" title={value}>
        {value}
      </dd>
    </div>
  );
}

export function NIRWorkflowPanel({ workflow }: NIRWorkflowPanelProps) {
  const { t } = useI18n();
  const view = useMemo(() => selectNIRWorkflowView(workflow), [workflow]);
  if (!view) {
    return null;
  }

  const stage = translated(
    t.nirWorkflow.stages,
    view.stage,
    t.nirWorkflow.notAvailable,
  );
  const task = translated(
    t.nirWorkflow.taskTypes,
    view.taskType,
    t.nirWorkflow.notAvailable,
  );
  const validation = view.validationScope
    ? translated(
        t.nirWorkflow.validationScopes,
        view.validationScope,
        t.nirWorkflow.notAvailable,
      )
    : translated(
        t.nirWorkflow.validationGoals,
        view.validationGoal,
        t.nirWorkflow.notAvailable,
      );
  const approval = translated(
    t.nirWorkflow.approvals,
    view.approvalStatus,
    t.nirWorkflow.notAvailable,
  );
  const blocked = view.stage === "blocked";
  const completed = ["approved", "registered", "completed"].includes(
    view.stage,
  );

  return (
    <Sheet>
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
      <SheetContent className="w-[92vw] gap-0 sm:max-w-md">
        <SheetHeader className="border-b pr-12">
          <SheetTitle className="flex items-center gap-2">
            <FlaskConicalIcon className="size-5" />
            {t.nirWorkflow.title}
          </SheetTitle>
          <SheetDescription>{t.nirWorkflow.description}</SheetDescription>
        </SheetHeader>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
          <Section icon={ActivityIcon} title={t.nirWorkflow.currentStage}>
            <div className="flex items-center justify-between gap-3">
              <span
                className={cn(
                  "rounded-full px-2.5 py-1 text-sm font-semibold",
                  blocked
                    ? "bg-destructive/10 text-destructive"
                    : completed
                      ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                      : "bg-amber-500/10 text-amber-700 dark:text-amber-300",
                )}
              >
                {stage}
              </span>
              <span className="text-muted-foreground text-xs">
                {t.nirWorkflow.attempt(view.attempt, view.maxAttempts)}
              </span>
            </div>
            <dl className="mt-2 divide-y">
              <Definition label={t.nirWorkflow.task} value={task} />
              <Definition
                label={t.nirWorkflow.revision}
                value={String(view.revision)}
              />
              {view.nextAction && (
                <Definition
                  label={t.nirWorkflow.nextAction}
                  value={view.nextAction}
                />
              )}
            </dl>
          </Section>

          <Section icon={FlaskConicalIcon} title={t.nirWorkflow.method}>
            <dl className="divide-y">
              <Definition
                label={t.nirWorkflow.method}
                value={view.method ?? t.nirWorkflow.notAvailable}
              />
              <Definition
                label={t.nirWorkflow.preprocessing}
                value={
                  view.preprocessing.length > 0
                    ? view.preprocessing.join(" → ")
                    : t.nirWorkflow.notAvailable
                }
              />
            </dl>
          </Section>

          <Section icon={CheckCircle2Icon} title={t.nirWorkflow.metrics}>
            {view.metrics.length > 0 ? (
              <dl className="divide-y">
                {view.metrics.map((metric) => (
                  <Definition
                    key={metric.key}
                    label={t.nirWorkflow.metricLabels[metric.key] ?? metric.key}
                    value={metric.value}
                  />
                ))}
              </dl>
            ) : (
              <p className="text-muted-foreground text-sm">
                {t.nirWorkflow.notAvailable}
              </p>
            )}
          </Section>

          <Section icon={ShieldCheckIcon} title={t.nirWorkflow.validation}>
            <dl className="divide-y">
              <Definition label={t.nirWorkflow.validation} value={validation} />
              <Definition label={t.nirWorkflow.approval} value={approval} />
            </dl>
          </Section>

          {view.datasetId && (
            <Section icon={DatabaseIcon} title={t.nirWorkflow.source}>
              <dl className="divide-y">
                <Definition
                  label={t.nirWorkflow.dataset}
                  value={view.datasetId}
                />
                {view.datasetProfileId && (
                  <Definition
                    label={t.nirWorkflow.profile}
                    value={view.datasetProfileId}
                  />
                )}
                {view.datasetHashShort && (
                  <Definition
                    label={t.nirWorkflow.hash}
                    value={`${view.datasetHashShort}…`}
                  />
                )}
              </dl>
            </Section>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}
