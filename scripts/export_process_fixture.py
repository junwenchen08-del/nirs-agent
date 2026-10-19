"""Export a previously computed process store for browser QA, without retraining."""

import argparse
import json
from pathlib import Path

from deerflow.community.nir.process import ProcessStore

parser = argparse.ArgumentParser()
parser.add_argument("store", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
store = ProcessStore(args.store)
runs = store.runs()
payload = {"runs": runs, "summaries": {}, "steps": {}, "candidates": {}, "charts": {}}
for run in runs["data"]:
    rid = run["run_id"]
    summary = store.summary(rid)
    payload["summaries"][rid] = summary
    for attempt in summary["attempts"]:
        for step in attempt["steps"]:
            sid = step["step_execution_id"]
            detail = store.step(rid, sid)
            payload["steps"][sid] = detail
            payload["candidates"][sid] = store.candidates(rid, sid)
            for ref in detail["charts"]:
                payload["charts"][ref["chart_id"]] = store.chart(
                    rid, ref["chart_id"], ref["version"]
                )
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Exported {len(payload['steps'])} steps and {len(payload['charts'])} charts")
