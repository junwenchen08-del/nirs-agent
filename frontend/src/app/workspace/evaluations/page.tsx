import { NIREvaluationDashboard } from "@/components/workspace/evaluations/nir-evaluation-dashboard";
import {
  WorkspaceBody,
  WorkspaceContainer,
  WorkspaceHeader,
} from "@/components/workspace/workspace-container";

export default function EvaluationsPage() {
  return (
    <WorkspaceContainer>
      <WorkspaceHeader />
      <WorkspaceBody className="items-stretch overflow-y-auto">
        <NIREvaluationDashboard />
      </WorkspaceBody>
    </WorkspaceContainer>
  );
}
