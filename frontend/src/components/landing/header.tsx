import Link from "next/link";

import { Button } from "@/components/ui/button";
import { BRAND } from "@/core/branding";
import type { Locale } from "@/core/i18n/locale";
import { getI18n } from "@/core/i18n/server";
import { cn } from "@/lib/utils";

export type HeaderProps = {
  className?: string;
  homeURL?: string;
  locale?: Locale;
};

export async function Header({ className, homeURL, locale }: HeaderProps) {
  const isExternalHome = !homeURL;
  const { locale: resolvedLocale, t } = await getI18n(locale);
  const lang = resolvedLocale.substring(0, 2);
  return (
    <header
      className={cn(
        "bg-background/95 fixed top-0 right-0 left-0 z-20 mx-auto flex h-16 max-w-6xl items-center justify-between gap-3 px-6 backdrop-blur-sm",
        className,
      )}
    >
      <div className="flex items-center gap-6">
        <a
          href={homeURL ?? BRAND.repositoryUrl}
          target={isExternalHome ? "_blank" : "_self"}
          rel={isExternalHome ? "noopener noreferrer" : undefined}
        >
          <h1 className="text-xl font-semibold">{BRAND.name}</h1>
        </a>
      </div>
      <nav className="ml-auto flex items-center gap-4 text-sm font-medium">
        <Link
          href={`/${lang}/docs`}
          className="text-secondary-foreground hover:text-foreground transition-colors"
        >
          {t.home.docs}
        </Link>
        <Link
          href="/blog/posts"
          className="text-secondary-foreground hover:text-foreground transition-colors"
        >
          {t.home.blog}
        </Link>
      </nav>
      <Button asChild size="sm">
        <Link href="/workspace">
          {resolvedLocale === "zh-CN" ? "进入工作台" : "Open workspace"}
        </Link>
      </Button>
      <hr className="from-border/0 via-border/70 to-border/0 absolute top-16 right-0 left-0 z-10 m-0 h-px w-full border-none bg-linear-to-r" />
    </header>
  );
}
