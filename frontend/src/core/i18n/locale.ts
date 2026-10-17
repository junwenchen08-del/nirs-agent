export const SUPPORTED_LOCALES = ["zh-CN", "en-US"] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];
export const DEFAULT_LOCALE: Locale = "zh-CN";

export function isLocale(value: string): value is Locale {
  return (SUPPORTED_LOCALES as readonly string[]).includes(value);
}

export function getLocaleByLang(lang: string): Locale {
  const normalizedLang = lang.toLowerCase();
  for (const locale of SUPPORTED_LOCALES) {
    if (locale.startsWith(normalizedLang)) {
      return locale;
    }
  }
  return DEFAULT_LOCALE;
}

export function getLangByLocale(locale: Locale): string {
  const parts = locale.split("-");
  if (parts.length > 0 && typeof parts[0] === "string") {
    return parts[0];
  }
  return locale;
}

export function normalizeLocale(locale: string | null | undefined): Locale {
  if (!locale) {
    return DEFAULT_LOCALE;
  }

  if (isLocale(locale)) {
    return locale;
  }

  const language = locale.trim().replaceAll("_", "-").toLowerCase();
  if (language === "zh" || language.startsWith("zh-")) {
    return "zh-CN";
  }
  if (language === "en" || language.startsWith("en-")) return "en-US";

  return DEFAULT_LOCALE;
}

// The product starts in Chinese. Explicit saved choices are restored by
// useI18n; the browser's language must not silently override this default.
export function detectLocale(): Locale {
  return DEFAULT_LOCALE;
}
