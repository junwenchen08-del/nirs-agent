import { BRAND } from "@/core/branding";

import type { Locale } from "./locale";

export function documentLabels(locale: Locale) {
  const chinese = locale === "zh-CN";
  return {
    editLink: chinese ? "完善本页文档" : "Edit this page",
    feedback: {
      content: chinese ? "反馈问题或建议" : "Feedback and suggestions",
      link: BRAND.issuesUrl,
    },
    themeSwitch: chinese
      ? { dark: "深色", light: "浅色", system: "跟随系统" }
      : { dark: "Dark", light: "Light", system: "System" },
    toc: {
      title: chinese ? "本页内容" : "On this page",
      backToTop: chinese ? "返回顶部" : "Back to top",
    },
    // The optional vendor AI-sharing menu has no localization interface.
    copyPageButton: false,
  };
}
