import { DatasetLibrary } from "@/components/workspace/nir-library";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default function NIRDatasetsPage() {
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <DatasetLibrary />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
