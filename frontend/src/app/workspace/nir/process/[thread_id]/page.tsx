import { Suspense } from "react";

import { ProcessWorkspace } from "@/components/workspace/nir/process-workspace";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default async function NIRProcessPage({
  params,
}: {
  params: Promise<{ thread_id: string }>;
}) {
  const { thread_id } = await params;
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-hidden">
        <Suspense>
          <ProcessWorkspace threadId={thread_id} />
        </Suspense>
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
