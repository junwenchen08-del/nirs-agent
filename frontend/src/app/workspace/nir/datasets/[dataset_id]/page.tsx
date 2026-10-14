import { DatasetWorkspaceView } from "@/components/workspace/nir-library/dataset-workspace";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default async function DatasetWorkspacePage({
  params,
}: {
  params: Promise<{ dataset_id: string }>;
}) {
  const { dataset_id } = await params;
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <DatasetWorkspaceView datasetId={dataset_id} />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
