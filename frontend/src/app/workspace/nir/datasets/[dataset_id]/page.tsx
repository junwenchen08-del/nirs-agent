import { DatasetWorkspaceView } from "@/components/workspace/nir-library/dataset-workspace";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default async function DatasetWorkspacePage({
  params,
  searchParams,
}: {
  params: Promise<{ dataset_id: string }>;
  searchParams: Promise<{ profile?: string }>;
}) {
  const { dataset_id } = await params;
  const { profile } = await searchParams;
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <DatasetWorkspaceView
          datasetId={dataset_id}
          profileId={profile ?? null}
        />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
