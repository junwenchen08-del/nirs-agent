"use client";

import { useI18n } from "@/core/i18n/hooks";
import { SafeStreamdown } from "@/core/streamdown/components";

import { getAboutMarkdown } from "./about-content";

export function AboutSettingsPage() {
  const { locale } = useI18n();
  return <SafeStreamdown>{getAboutMarkdown(locale)}</SafeStreamdown>;
}
