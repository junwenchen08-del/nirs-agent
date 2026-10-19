import type { Message } from "@langchain/langgraph-sdk";

import {
  nirDeliveryFiles,
  nirDeliverySession,
} from "@/core/artifacts/delivery";
import { useI18n } from "@/core/i18n/hooks";
import { useProcessResource } from "@/core/nir/process-hooks";

import { ArtifactFileList } from "../artifacts/artifact-file-list";

/** Refresh restores published files without changing checkpoints or training. */
export function NIRDeliveryFiles({
  message,
  threadId,
}: {
  message: Message;
  threadId: string;
}) {
  const { locale } = useI18n();
  const session = nirDeliverySession(message);
  const publication = useProcessResource<unknown>(
    threadId,
    session ? `autonomous/${session}` : null,
  );
  if (!session) return null;
  const files = nirDeliveryFiles(publication.data, session);
  if (files.length)
    return (
      <ArtifactFileList className="my-3" files={files} threadId={threadId} />
    );
  if (publication.isError)
    return (
      <p className="text-muted-foreground text-sm">
        {locale === "zh-CN"
          ? "下载文件暂不可读，请在建模过程页面检查交付状态。"
          : "Downloads are unavailable; check delivery status in the modeling process."}
      </p>
    );
  if (publication.isLoading)
    return (
      <p className="text-muted-foreground text-sm">
        {locale === "zh-CN"
          ? "正在读取已发布的下载文件…"
          : "Loading published downloads…"}
      </p>
    );
  return null;
}
