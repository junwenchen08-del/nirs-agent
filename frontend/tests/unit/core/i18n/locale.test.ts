import { afterEach, expect, rs, test } from "@rstest/core";

import {
  DEFAULT_LOCALE,
  detectLocale,
  normalizeLocale,
} from "@/core/i18n/locale";

afterEach(() => {
  rs.unstubAllGlobals();
});

test("first visits use Chinese even with an English browser", () => {
  rs.stubGlobal("window", {});
  rs.stubGlobal("navigator", { language: "en-US" });
  expect(DEFAULT_LOCALE).toBe("zh-CN");
  expect(detectLocale()).toBe("zh-CN");
  expect(normalizeLocale(undefined)).toBe("zh-CN");
  expect(normalizeLocale("fr-FR")).toBe("zh-CN");
});

test("explicit English choices and Chinese variants remain supported", () => {
  expect(normalizeLocale("en-US")).toBe("en-US");
  expect(normalizeLocale("en-GB")).toBe("en-US");
  expect(normalizeLocale("zh-Hans")).toBe("zh-CN");
  expect(normalizeLocale("zh_CN")).toBe("zh-CN");
});
