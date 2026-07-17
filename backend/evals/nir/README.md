# NIR Agent Evaluation

This directory contains the versioned acceptance scenarios for the NIR agent.
The evaluator is deterministic: it scores captured trajectories without calling
an LLM or rerunning chemometric training.

## Run

```bash
cd backend
make eval-nir TRACES=path/to/nir-traces.json
```

For a completed or active Gateway thread, export a CLI-compatible trace directly
from the latest LangGraph checkpoint:

```bash
curl -sS \
  "http://localhost:2026/api/threads/THREAD_ID/nir-evaluation-trace?scenario_id=calibration-register" \
  -o nir-traces.json

cd backend
make eval-nir TRACES=../nir-traces.json
```

The Gateway correlates checkpointed tool observations with persisted skill
activation events and run metadata. No model call or chemometric retraining is
performed during export.

Reports are written to:

- `.deer-flow/nir-evals/nir-eval-report.json`
- `.deer-flow/nir-evals/nir-eval-report.md`

The command fails when the average score is below 100 or any denied/out-of-stage
NIR tool call is observed. To use a different score threshold, call the script
directly with `--fail-under`.

## Trace Contract

```json
{
  "version": 1,
  "traces": [
    {
      "scenario_id": "calibration-register",
      "routed_skill": "nir-coordinator",
      "workflow": {
        "task_type": "calibration",
        "stage": "completed",
        "approval_status": "approved",
        "history": []
      },
      "tool_calls": [
        {
          "name": "nir_train_model",
          "status": "success",
          "stage_before": "execution",
          "stage_after": "review",
          "code": null,
          "duration_ms": 1200
        }
      ],
      "run_ids": ["gateway-run-id"],
      "trace_id": "optional-provider-trace-id",
      "duration_ms": 2500,
      "input_tokens": 800,
      "output_tokens": 350
    }
  ]
}
```

`workflow` must be the final checkpointed `nir_workflow` value. `tool_calls`
records the agent's decisions, including denied calls; a denied call proves the
runtime guard worked but still counts as an agent-policy violation in quality
evaluation.

Each workflow retains at most 100 tool observations. Run IDs, trace IDs, call
IDs, and observation details are excluded from the durable context injected
into the model; the model sees only `tool_observation_count`, while the complete
evidence remains available to the export API. Checkpoints created before these
fields existed export with empty observation/run lists.

## Score

Each scenario declares its expected route, task type, terminal stage, workflow
actions, tools, approval status, retry outcome, and required trace fields. The
evaluator emits explainable weighted checks instead of using another LLM as a
judge. Registration before the `approved` stage and all workflow policy-denial
codes are always counted as safety violations.

The version 1 catalog currently contains 20 scenarios spanning intake, audit
blocking, analysis, calibration, comparison, bounded RAG retries, prediction,
knowledge no-hit honesty, approval/rejection, registration, and multiple NIR
application domains.
