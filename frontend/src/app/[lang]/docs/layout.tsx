import type { PageMapItem } from "nextra";
import { getPageMap } from "nextra/page-map";
import { Layout, LastUpdated } from "nextra-theme-docs";

import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { BRAND } from "@/core/branding";
import { documentLabels } from "@/core/i18n/docs";
import { getLocaleByLang } from "@/core/i18n/locale";
import "nextra-theme-docs/style.css";

const i18n = [
  { locale: "zh", name: "中文" },
  { locale: "en", name: "English" },
];

function formatPageRoute(base: string, items: PageMapItem[]): PageMapItem[] {
  return items.map((item) => {
    // Page maps are shared by Nextra's docs and blog readers. Keep their source
    // routes intact so visiting docs cannot rewrite a later blog link.
    return {
      ...item,
      ...("route" in item && !item.route.startsWith(base)
        ? { route: `${base}${item.route}` }
        : {}),
      ...("children" in item && item.children
        ? { children: formatPageRoute(base, item.children) }
        : {}),
    };
  });
}

export default async function DocLayout({ children, params }) {
  const { lang } = await params;
  const locale = getLocaleByLang(lang);
  const pages = await getPageMap(`/${lang}`);
  const pageMap = formatPageRoute(`/${lang}/docs`, pages);

  return (
    <Layout
      {...documentLabels(locale)}
      lastUpdated={
        <LastUpdated locale={locale}>
          {locale === "zh-CN" ? "最后更新于" : "Last updated on"}
        </LastUpdated>
      }
      navbar={
        <Header
          className="sticky max-w-full px-10"
          homeURL="/"
          locale={locale}
        />
      }
      pageMap={pageMap}
      docsRepositoryBase={BRAND.docsRepositoryUrl}
      footer={<Footer className="mt-0" />}
      i18n={i18n}
      // ... Your additional layout options
    >
      {children}
    </Layout>
  );
}
