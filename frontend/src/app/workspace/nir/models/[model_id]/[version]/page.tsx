import { ModelDetail } from "@/components/workspace/nir-library/model-detail";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default async function ModelDetailPage({
  params,
}: {
  params: Promise<{ model_id: string; version: string }>;
}) {
  const { model_id, version } = await params;
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <ModelDetail modelId={model_id} version={version} />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
