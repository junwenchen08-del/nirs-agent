import { expect, test } from "@rstest/core";

import {
  groupDeliveryFiles,
  resolveReportImage,
} from "@/core/artifacts/delivery";

test("NIR deliveries show report and bundle, with technical files folded", () => {
  const root = "/mnt/user-data/outputs/nir_analysis/";
  const files = [
    "report.md",
    "metrics.json",
    "model.pkl",
    "raw_spectra.png",
    "report.html",
    "delivery.zip",
  ].map((name) => root + name);
  const grouped = groupDeliveryFiles(files);
  expect(grouped.primary).toEqual([
    root + "report.html",
    root + "delivery.zip",
  ]);
  expect(grouped.supporting).toHaveLength(4);
});

test("generic artifacts and incomplete deliveries remain visible", () => {
  const files = [
    "/mnt/user-data/outputs/report.md",
    "/mnt/user-data/outputs/metrics.json",
    "/mnt/user-data/outputs/photo.png",
  ];
  expect(groupDeliveryFiles(files).primary).toEqual(files);
  expect(groupDeliveryFiles(files).supporting).toEqual([]);
});

test("collection summary folds only known subset outputs", () => {
  const root = "/mnt/user-data/outputs/nir_collection/";
  const files = [
    "collection_summary.md",
    "R562/report.md",
    "R562/model.pkl",
    "R562/metrics.json",
    "notes.md",
  ].map((name) => root + name);
  expect(groupDeliveryFiles(files).primary).toEqual([
    root + "collection_summary.md",
    root + "notes.md",
  ]);
  expect(groupDeliveryFiles(files).supporting).toHaveLength(3);
});

test("relative report figures resolve inside the current artifact directory", () => {
  const url =
    "http://localhost:2027/api/threads/t/artifacts/mnt/user-data/outputs/run/report.md";
  expect(resolveReportImage("raw_spectra.png", url)).toBe(
    url.replace("report.md", "raw_spectra.png"),
  );
  expect(resolveReportImage("R562/raw_spectra.png", url)).toBe(
    url.replace("report.md", "R562/raw_spectra.png"),
  );
  expect(resolveReportImage("../../private.png", url)).toBeUndefined();
  expect(resolveReportImage("https://example.com/photo.png", url)).toBe(
    "https://example.com/photo.png",
  );
});
