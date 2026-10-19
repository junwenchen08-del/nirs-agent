import { readFileSync, readdirSync, existsSync } from "node:fs";
import path from "node:path";

import { expect, test } from "@rstest/core";

import { getAboutMarkdown } from "@/components/workspace/settings/about-content";
import { BRAND } from "@/core/branding";
import { translations } from "@/core/i18n/translations";

const root = process.cwd();
const upstreamBrand = /deer[\s_-]?flow|🦌|bytedance/i;

test("both locales and about pages expose ChemAgent and its own help links", () => {
  for (const locale of ["en-US", "zh-CN"] as const) {
    expect(JSON.stringify(translations[locale])).not.toMatch(upstreamBrand);
    const about = getAboutMarkdown(locale);
    expect(about).toContain("ChemAgent");
    expect(about).toContain(BRAND.repositoryUrl);
    expect(about).toContain(`/${locale.slice(0, 2)}/docs`);
    expect(about).not.toMatch(upstreamBrand);
  }
});

function publicTextFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return publicTextFiles(file);
    return /\.(?:mdx?|tsx?|json|svg|html|css)$/.test(entry.name) ? [file] : [];
  });
}

test("shipped docs, demo content and images contain no upstream branding", () => {
  const files = ["src/content", "public"].flatMap((directory) =>
    publicTextFiles(path.join(root, directory)),
  );
  const leftovers = files.filter((file) =>
    upstreamBrand.test(readFileSync(file, "utf8")),
  );
  expect(leftovers).toEqual([]);
  expect(existsSync(path.join(root, "public/images/deer.svg"))).toBe(false);
  expect(existsSync(path.join(root, "public/favicon.ico"))).toBe(false);
});
