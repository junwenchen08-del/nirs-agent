import { MethodLibrary } from "@/components/workspace/method-library/method-library";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default function MethodsPage() {
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <MethodLibrary />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
