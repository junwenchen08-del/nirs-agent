export function ModelQualityNotice({
  status,
  zh,
}: {
  status: string | null | undefined;
  zh: boolean;
}) {
  if (status !== "below_threshold_user_approved") return null;
  return (
    <p
      data-testid="model-quality-notice"
      className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm"
    >
      {zh
        ? "未达质量门槛，已由用户确认保存。评价指标保持原样。"
        : "Below the quality thresholds; saved with user approval. Recorded metrics are unchanged."}
    </p>
  );
}
