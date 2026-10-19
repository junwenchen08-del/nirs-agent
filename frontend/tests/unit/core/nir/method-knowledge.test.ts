import { expect, test } from "@rstest/core";

import { selectMethodKnowledge } from "@/core/nir/method-knowledge";

test("projects recorded method sources and keeps missing evidence explicit", () => {
  expect(selectMethodKnowledge(undefined)).toBeNull();
  const view = selectMethodKnowledge({
    mode: "retrieval_guided",
    problem_labels: ["高频噪声"],
    profile_scope: "calibration_only",
    sources: [
      {
        method_id: "sg_smooth",
        title: "SG 平滑",
        source_url:
          "https://chemotools.org/methods/generated/chemotools.smooth.SavitzkyGolayFilter.html",
        evidence_id: "method:sg_smooth:abc",
        matched_tags: ["high_frequency_noise"],
        auto_eligible: true,
      },
    ],
  });
  expect(view?.mode).toBe("retrieval_guided");
  expect(view?.findings).toEqual(["高频噪声"]);
  expect(view?.sources[0]?.url).toContain("https://chemotools.org/");
  expect(view?.sources[0]?.eligible).toBe(true);
});

test("blocks unsafe source URLs and never copies raw data or arbitrary fields", () => {
  const view = selectMethodKnowledge({
    mode: "rule_fallback",
    reason: "method_retrieval_unavailable",
    raw_spectra: [1, 2, 3],
    sources: [
      {
        method_id: "msc",
        source_url: "javascript:alert(1)",
        secret: "do-not-display",
      },
      {
        method_id: "snv",
        source_url: "https://chemotools.org.evil.example/methods/",
      },
      {
        method_id: "sg_smooth",
        source_url: "https://chemotools.org@evil.example/methods/",
      },
    ],
  });
  expect(view?.sources.every((source) => source.url === null)).toBe(true);
  expect(JSON.stringify(view)).not.toContain("do-not-display");
  expect(JSON.stringify(view)).not.toContain("raw_spectra");
  expect(view?.reason).toBe("method_retrieval_unavailable");
});

test("bounds source cards and handles malformed historical evidence", () => {
  const view = selectMethodKnowledge({
    mode: "unexpected",
    sources: Array.from({ length: 100 }, () => ({
      method_id: "snv",
      title: "x".repeat(2000),
    })),
  });
  expect(view?.mode).toBe("unknown");
  expect(view?.sources.length).toBe(6);
  expect(view?.sources[0]?.title.length).toBeLessThanOrEqual(160);
  expect(selectMethodKnowledge([])).toBeNull();
});
