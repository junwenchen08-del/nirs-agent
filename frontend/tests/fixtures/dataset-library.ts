import type {
  NIRDataset,
  NIRDatasetPreview,
  DatasetAnalysisType,
} from "@/core/nir-library/types";

export const libraryDatasets: NIRDataset[] = [
  {
    id: "ds_corn",
    name: "玉米水分",
    original_filename: "corn.mat",
    sha256: "a".repeat(64),
    size_bytes: 454553,
    media_type: "application/octet-stream",
    status: "ready",
    created_at: "2026-10-13T12:12:58Z",
  },
  {
    id: "ds_wheat",
    name: "小麦品种",
    original_filename: "wheat.csv",
    sha256: "b".repeat(64),
    size_bytes: 20480,
    media_type: "text/csv",
    status: "ready",
    created_at: "2026-10-13T12:12:58Z",
  },
  {
    id: "ds_unknown",
    name: "待分析光谱",
    original_filename: "spectra.txt",
    sha256: "c".repeat(64),
    size_bytes: 4096,
    media_type: "text/plain",
    status: "ready",
    created_at: "2026-10-13T12:12:58Z",
  },
];

export function previewOf(
  dataset: NIRDataset,
  kind: DatasetAnalysisType = "regression",
): NIRDatasetPreview {
  return {
    dataset_id: dataset.id,
    source_sha256: dataset.sha256!,
    status: "ready",
    message: "数据概览",
    analysis_type: kind,
    sample_count: 60,
    feature_count: 700,
    preview_sample_count: 60,
    missing_ratio: 0,
    axis_kind: "axis",
    spectrum: Array.from({ length: 80 }, (_, i) => ({
      x: 1000 + i * 10,
      mean: 0.8 + Math.sin(i / 10) * 0.2,
      low: 0.5 + Math.sin(i / 10) * 0.15,
      high: 1.1 + Math.sin(i / 10) * 0.2,
    })),
    targets:
      kind === "unknown"
        ? []
        : [
            {
              name: kind === "classification" ? "品种" : "水分",
              kind,
              bars:
                kind === "classification"
                  ? [
                      { label: "品种一", count: 35 },
                      { label: "品种二", count: 25 },
                    ]
                  : [
                      { label: "8–10", count: 15 },
                      { label: "10–12", count: 30 },
                      { label: "12–14", count: 15 },
                    ],
            },
          ],
    fields: ["样本编号", "水分", "1000", "1100", "1200"],
  };
}
