import type { Edge, Node } from "@xyflow/react";

import { parseDecision, record, text, type DecisionView } from "./autonomous";

export const roundHistoryPageSize = 20;

export function roundWorkflowPollingInterval(
  workflow: unknown,
): number | false {
  const stage = text(record(workflow).stage);
  return stage &&
    !["review", "approved", "registered", "completed", "blocked"].includes(
      stage,
    )
    ? 3000
    : false;
}
export type RoundNode = Node<{ decision: DecisionView }, "round">;
export type RoundEdge = Edge<{ relation: "feedback" | "retained" }> & {
  pathOptions?: { offset: number };
};

export function parseRoundPage(raw: unknown, offset: number) {
  const page = record(raw);
  return {
    data: Array.isArray(page.data)
      ? page.data
          .slice(0, roundHistoryPageSize)
          .map((item) => parseDecision(item))
          .filter((item): item is DecisionView => item !== null)
      : [],
    next_offset:
      typeof page.next_offset === "number" &&
      Number.isInteger(page.next_offset) &&
      page.next_offset > offset
        ? page.next_offset
        : null,
    total:
      typeof page.total === "number" &&
      Number.isInteger(page.total) &&
      page.total >= 0
        ? page.total
        : null,
  };
}

export function roundConfiguration(decision: DecisionView) {
  const actual = record(decision.feedback?.actual_configuration);
  const planned = record(decision.selected.pipeline);
  const executed =
    decision.status === "succeeded" && Boolean(text(actual.method));
  return {
    actual: executed,
    method: text(executed ? actual.method : planned.model_method),
    steps: (Array.isArray(executed ? actual.steps : planned.preprocessing)
      ? executed
        ? actual.steps
        : planned.preprocessing
      : []) as unknown[],
    parameters: record(
      executed ? decision.feedback?.actual_parameters : planned.fixed_params,
    ),
    optimization: record(decision.selected.optimization),
  };
}

export function roundGraph(rows: DecisionView[]): {
  nodes: RoundNode[];
  edges: RoundEdge[];
} {
  const experiments = new Map(
    rows
      .filter(
        (row) => row.experiment_id && !["STOP", "RETAIN"].includes(row.action),
      )
      .map((row) => [row.experiment_id, row]),
  );
  const indexes = new Map(rows.map((row, index) => [row.decision_id, index]));
  const children = new Map<string, string[]>();
  const parents = new Map<string, string>();
  const edges: RoundEdge[] = [];
  for (const row of rows) {
    const parent = experiments.get(row.base_experiment_id);
    if (
      parent &&
      indexes.get(parent.decision_id)! < indexes.get(row.decision_id)!
    ) {
      parents.set(row.decision_id, parent.decision_id);
      children.set(parent.decision_id, [
        ...(children.get(parent.decision_id) ?? []),
        row.decision_id,
      ]);
      edges.push({
        id: `feedback-${row.decision_id}`,
        source: parent.decision_id,
        target: row.decision_id,
        data: { relation: "feedback" },
      });
    }
    const retained = experiments.get(row.retained_experiment_id);
    if (
      ["STOP", "RETAIN"].includes(row.action) &&
      retained &&
      indexes.get(retained.decision_id)! < indexes.get(row.decision_id)!
    )
      edges.push({
        id: `retained-${row.decision_id}`,
        source: retained.decision_id,
        target: row.decision_id,
        data: { relation: "retained" },
        type: "smoothstep",
        sourceHandle: "retention-source",
        targetHandle: "retention-target",
        pathOptions: { offset: 10 },
        label: "保留",
        labelBgStyle: { fill: "var(--card)" },
        labelStyle: { fill: "var(--foreground)" },
        style: { strokeDasharray: "5 4", stroke: "var(--color-emerald-600)" },
      });
  }
  const positions = new Map<string, { x: number; y: number }>();
  const width = (id: string): number =>
    Math.max(
      1,
      (children.get(id) ?? []).reduce((sum, child) => sum + width(child), 0),
    );
  const place = (id: string, start: number, depth: number) => {
    positions.set(id, {
      x: (start + (width(id) - 1) / 2) * 320,
      y: depth * 220,
    });
    let cursor = start;
    for (const child of children.get(id) ?? []) {
      place(child, cursor, depth + 1);
      cursor += width(child);
    }
  };
  let start = 0;
  rows
    .filter((row) => !parents.has(row.decision_id))
    .forEach((row) => {
      place(row.decision_id, start, 0);
      start += width(row.decision_id);
    });
  return {
    nodes: rows.map((decision) => ({
      id: decision.decision_id,
      type: "round",
      position: positions.get(decision.decision_id)!,
      data: { decision },
      draggable: false,
      connectable: false,
    })),
    edges,
  };
}

/**
 * The selected round plus its directly recorded neighbours. Fitting the whole
 * history to a narrow panel shrinks a long chain below a readable zoom, so the
 * graph opens focused and the viewport controls still show everything.
 */
export function roundGraphFocus(
  graph: { nodes: RoundNode[]; edges: RoundEdge[] },
  selectedId: string,
): RoundNode[] {
  const focus = graph.nodes.filter(
    (node) =>
      node.id === selectedId ||
      graph.edges.some(
        (edge) =>
          (edge.source === selectedId && edge.target === node.id) ||
          (edge.target === selectedId && edge.source === node.id),
      ),
  );
  return focus.length ? focus : graph.nodes;
}

export interface RoundPoints {
  scope: string;
  observed: number[];
  predicted: number[];
  residual: number[];
  sample_ids: string[];
  n_total: number;
  n_returned: number;
  sampling: string;
  unit: string | null;
}

export function parseRoundDiagnostics(
  raw: unknown,
  sid: string,
  eid: string,
  did: string,
  group: string,
): Record<string, RoundPoints> | null {
  const payload = record(raw);
  if (
    payload.schema_version !== 1 ||
    payload.evidence_status !== "available" ||
    payload.session_id !== sid ||
    payload.experiment_id !== eid ||
    payload.decision_id !== did ||
    !group ||
    payload.comparison_group_id !== group
  )
    return null;
  const result: Record<string, RoundPoints> = {};
  for (const [scope, rawChart] of Object.entries(record(payload.charts))) {
    const chart = record(rawChart);
    if (
      !["calibration", "calibration_cv", "validation"].includes(scope) ||
      chart.scope !== scope ||
      chart.evidence_status !== "available"
    )
      return null;
    const arrays = [chart.observed, chart.predicted, chart.residual];
    if (
      !arrays.every(
        (values) =>
          Array.isArray(values) &&
          values.length > 0 &&
          values.length <= 2000 &&
          values.every(
            (value) => typeof value === "number" && Number.isFinite(value),
          ),
      )
    )
      return null;
    const [observed, predicted, residual] = arrays as number[][];
    const ids = chart.sample_ids;
    if (
      !observed ||
      !predicted ||
      !residual ||
      !Array.isArray(ids) ||
      ids.some((id) => typeof id !== "string" || !/^[a-f0-9]{16}$/.test(id)) ||
      observed.length !== predicted.length ||
      observed.length !== residual.length ||
      observed.length !== ids.length ||
      new Set(ids).size !== ids.length ||
      chart.n_returned !== ids.length ||
      !Number.isInteger(chart.n_total) ||
      Number(chart.n_total) < ids.length
    )
      return null;
    if (
      observed.some(
        (value, index) =>
          Math.abs(predicted[index]! - value - residual[index]!) >
          1e-9 * Math.max(1, Math.abs(predicted[index]!), Math.abs(value)),
      )
    )
      return null;
    result[scope] = {
      scope,
      observed,
      predicted,
      residual,
      sample_ids: ids as string[],
      n_total: Number(chart.n_total),
      n_returned: ids.length,
      sampling: text(chart.sampling),
      unit: text(chart.unit, 50) || null,
    };
  }
  return Object.keys(result).length ? result : null;
}
