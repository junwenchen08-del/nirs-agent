"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useMemo } from "react";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { BRAND } from "@/core/branding";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

import { GithubIcon } from "./github-icon";
import { Tooltip } from "./tooltip";

export function WorkspaceContainer({
  className,
  children,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div className={cn("flex h-screen w-full flex-col", className)} {...props}>
      {children}
    </div>
  );
}

export function WorkspaceHeader({
  className,
  children,
  returnToThreadId,
  ...props
}: React.ComponentProps<"header"> & { returnToThreadId?: string }) {
  const { t } = useI18n();
  const pathname = usePathname();
  const segments = useMemo(() => {
    const parts = pathname?.split("/") || [];
    if (parts.length > 0) {
      return parts.slice(1, 3);
    }
  }, [pathname]);
  const section = useMemo(() => pathname?.split("/")[3], [pathname]);
  // `/workspace/nir` has no page of its own: the library label would misname the
  // process explorer, and linking there would 404.
  const nirRoot = segments?.[0] === "workspace" && segments?.[1] === "nir";
  return (
    <header
      className={cn(
        "top-0 right-0 left-0 z-20 flex h-16 shrink-0 items-center justify-between gap-2 border-b backdrop-blur-sm transition-[width,height] ease-out group-has-data-[collapsible=icon]/sidebar-wrapper:h-12",
        className,
      )}
      {...props}
    >
      <div className="flex min-w-0 items-center gap-2 px-2 sm:px-4">
        <SidebarTrigger className="md:hidden" />
        <Breadcrumb>
          <BreadcrumbList>
            {segments?.[0] && (
              <BreadcrumbItem className="hidden md:block">
                <BreadcrumbLink asChild>
                  <Link
                    href={
                      returnToThreadId
                        ? `/workspace/chats/${encodeURIComponent(returnToThreadId)}`
                        : `/${segments[0]}`
                    }
                    title={
                      returnToThreadId
                        ? t.breadcrumb.backToCurrentChat
                        : undefined
                    }
                  >
                    {nameOfSegment(segments[0], t)}
                  </Link>
                </BreadcrumbLink>
              </BreadcrumbItem>
            )}
            {segments?.[1] && (
              <>
                <BreadcrumbSeparator className="hidden md:block" />
                <BreadcrumbItem>
                  {nirRoot ? (
                    <BreadcrumbPage>
                      {nameOfSegment(segments[1], t, section)}
                    </BreadcrumbPage>
                  ) : (
                    <BreadcrumbLink asChild>
                      <Link href={`/${segments[0]}/${segments[1]}`}>
                        {nameOfSegment(segments[1], t, section)}
                      </Link>
                    </BreadcrumbLink>
                  )}
                </BreadcrumbItem>
              </>
            )}
            {children && (
              <>
                <BreadcrumbSeparator />
                {children}
              </>
            )}
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="pr-4">
        <Tooltip content={t.workspace.githubTooltip}>
          <a
            href={BRAND.repositoryUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="opacity-75 transition hover:opacity-100"
          >
            <GithubIcon className="size-6" />
          </a>
        </Tooltip>
      </div>
    </header>
  );
}

export function WorkspaceBody({
  className,
  children,
  ...props
}: React.ComponentProps<"main">) {
  return (
    <main
      className={cn(
        "relative flex min-h-0 w-full flex-1 flex-col items-center",
        className,
      )}
      {...props}
    >
      <div className="flex h-full w-full flex-col items-center">{children}</div>
    </main>
  );
}

function nameOfSegment(
  segment: string | undefined,
  t: ReturnType<typeof useI18n>["t"],
  section?: string,
) {
  if (!segment) return t.common.home;
  if (segment === "workspace") return t.breadcrumb.workspace;
  if (segment === "chats") return t.breadcrumb.chats;
  if (segment === "methods") return t.breadcrumb.methods;
  if (segment === "nir") {
    if (section === "process") return t.breadcrumb.nirProcess;
    if (section === "datasets") return t.sidebar.nirDatasets;
    if (section === "models") return t.sidebar.nirModels;
    return t.breadcrumb.nirLibrary;
  }
  return segment[0]?.toUpperCase() + segment.slice(1);
}
