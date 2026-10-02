import { ModelLibrary } from "@/components/workspace/nir-library";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default function NIRModelsPage() {
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <ModelLibrary />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
