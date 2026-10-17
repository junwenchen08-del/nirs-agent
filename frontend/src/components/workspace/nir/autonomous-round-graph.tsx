"use client";

import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  type NodeProps,
  type ReactFlowInstance,
} from "@xyflow/react";
import { createContext, useContext, useEffect, useMemo, useRef } from "react";

import {
  type AutonomousSummary,
  type DecisionView,
} from "@/core/nir/autonomous";
import {
  roundGraph,
  roundGraphFocus,
  type RoundEdge,
  type RoundNode,
} from "@/core/nir/round-process";

import { RoundNodeCard } from "./round-node-card";

import "@xyflow/react/dist/style.css";

const GraphContext = createContext<{
  summary: AutonomousSummary;
  rows: DecisionView[];
  selectedId: string;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
} | null>(null);

function RoundGraphNode({ data }: NodeProps<RoundNode>) {
  const context = useContext(GraphContext)!;
  return (
    <div className="w-72">
      <Handle
        type="target"
        position={Position.Top}
        className="opacity-0"
        isConnectable={false}
      />
      <RoundNodeCard
        decision={data.decision}
        summary={context.summary}
        rows={context.rows}
        selected={context.selectedId === data.decision.decision_id}
        zh={context.zh}
        onSelect={context.onSelect}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        className="opacity-0"
        isConnectable={false}
      />
      <Handle
        id="retention-source"
        type="source"
        position={Position.Right}
        className="opacity-0"
        isConnectable={false}
      />
      <Handle
        id="retention-target"
        type="target"
        position={Position.Right}
        className="opacity-0"
        isConnectable={false}
      />
    </div>
  );
}
const nodeTypes = { round: RoundGraphNode };

export function AutonomousRoundGraph(props: {
  summary: AutonomousSummary;
  rows: DecisionView[];
  selectedId: string;
  zh: boolean;
  onSelect: (decision: DecisionView) => void;
}) {
  const graph = useMemo(() => roundGraph(props.rows), [props.rows]);
  const focus = useMemo(
    () => roundGraphFocus(graph, props.selectedId),
    [graph, props.selectedId],
  );
  const instance = useRef<ReactFlowInstance<RoundNode, RoundEdge> | null>(null);
  const fitted = useRef("");
  // A mount-time `fitView` runs before the round history arrives and would shrink
  // every node to an unreadable zoom, so fit again once the shown rounds change.
  useEffect(() => {
    const flow = instance.current;
    if (!flow || !graph.nodes.length) return;
    const signature = `${props.selectedId}:${graph.nodes.length}`;
    if (fitted.current === signature) return;
    fitted.current = signature;
    void flow.fitView({
      nodes: focus.map((node) => ({ id: node.id })),
      maxZoom: 1,
      padding: 0.2,
    });
  }, [focus, graph, props.selectedId]);
  return (
    <GraphContext.Provider value={props}>
      <div
        className="bg-muted/20 h-[640px] min-w-0 rounded-xl border"
        data-testid="round-graph"
        aria-label={props.zh ? "方案关系图" : "Recorded experiment graph"}
      >
        <ReactFlow
          nodes={graph.nodes}
          edges={graph.edges.map((edge) =>
            edge.data?.relation === "retained"
              ? { ...edge, label: props.zh ? "保留" : "Retain" }
              : edge,
          )}
          nodeTypes={nodeTypes}
          onInit={(ready) => {
            instance.current = ready;
          }}
          minZoom={0.2}
          maxZoom={1.5}
          nodesDraggable={false}
          nodesConnectable={false}
          edgesReconnectable={false}
          deleteKeyCode={null}
          zoomOnDoubleClick={false}
          preventScrolling={false}
          proOptions={{ hideAttribution: true }}
          ariaLabelConfig={{
            "controls.zoomIn.ariaLabel": props.zh ? "放大" : "Zoom in",
            "controls.zoomOut.ariaLabel": props.zh ? "缩小" : "Zoom out",
            "controls.fitView.ariaLabel": props.zh ? "适应画布" : "Fit view",
          }}
        >
          <Background gap={20} color="var(--border)" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    </GraphContext.Provider>
  );
}
