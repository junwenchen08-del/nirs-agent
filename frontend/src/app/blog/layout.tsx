import { Layout, LastUpdated } from "nextra-theme-docs";

import { Footer } from "@/components/landing/footer";
import { Header } from "@/components/landing/header";
import { getBlogIndexData, getPreferredBlogLang } from "@/core/blog";
import { BRAND } from "@/core/branding";
import { documentLabels } from "@/core/i18n/docs";
import { getI18n } from "@/core/i18n/server";
import "nextra-theme-docs/style.css";

export default async function BlogLayout({ children }) {
  const { locale } = await getI18n();
  const { pageMap } = await getBlogIndexData(getPreferredBlogLang(locale));

  return (
    <Layout
      {...documentLabels(locale)}
      lastUpdated={
        <LastUpdated locale={locale}>
          {locale === "zh-CN" ? "最后更新于" : "Last updated on"}
        </LastUpdated>
      }
      navbar={<Header className="relative max-w-full px-10" homeURL="/" />}
      pageMap={pageMap}
      sidebar={{ defaultOpen: true }}
      docsRepositoryBase={BRAND.docsRepositoryUrl}
      footer={<Footer />}
    >
      {children}
    </Layout>
  );
}
