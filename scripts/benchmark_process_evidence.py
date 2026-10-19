"""Measure bounded process reads using synthetic scale data, without training."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from deerflow.community.nir.process import ProcessStore, regression_evidence


def benchmark(root, rows, columns):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((rows, columns), dtype=np.float32)
    y = X[:, :5].sum(axis=1).astype(float)
    a, b = int(rows * 0.7), int(rows * 0.85)
    partitions = {
        "calibration": (X[:a], y[:a]),
        "tuning": (X[a:b], y[a:b]),
        "holdout": (X[b:], y[b:]),
    }
    metrics = {
        "training_data_hash": "synthetic-scale-only",
        "protocol": "scale_fixture_v1",
        "validation_scope": "independent_holdout_not_external",
        "method": "linear_fixture",
        "val": {"RMSE": 0.0},
        "test": {"RMSE": 0.0},
        "wavelength_selection": {"method": "none"},
    }
    start = time.perf_counter()
    evidence = regression_evidence(
        X=X,
        y=y,
        wv=np.arange(columns),
        partitions=partitions,
        processed_cal=X[:a],
        predictions={"holdout": y[b:], "tuning": y[a:b]},
        metrics=metrics,
        unit="arb",
        target="synthetic",
        namespace="scale-only",
    )
    generation_ms = (time.perf_counter() - start) * 1000
    model = next(item for item in evidence if item["step_key"] == "model")
    model["candidates"] = [
        {
            "candidate_id": str(i),
            "method": "scale_fixture",
            "score": float(i),
            "evaluation_kind": "tuning",
            "parameters": {"value": i},
        }
        for i in range(200)
    ]
    store = ProcessStore(root / f"scale-{rows}-{columns}")
    for index in range(20):
        attempt = store.begin("scale", f"attempt-{index}", "scale_fixture")
        store.publish(attempt, evidence)
        store.finish(attempt, "succeeded", {"stage": "review"})
    summary = store.summary(attempt["run_id"])
    refs = [
        step for step in summary["attempts"][-1]["steps"] if step["step_key"] == "audit"
    ]
    sid = refs[0]["step_execution_id"]
    detail = store.step(attempt["run_id"], sid)
    chart_ref = detail["charts"][0]
    timings = []
    for _ in range(30):
        start = time.perf_counter()
        store.summary(attempt["run_id"])
        store.step(attempt["run_id"], sid)
        json.dumps(
            store.chart(attempt["run_id"], chart_ref["chart_id"], chart_ref["version"])
        )
        timings.append((time.perf_counter() - start) * 1000)
    return {
        "shape": [rows, columns],
        "attempts": 20,
        "candidates": 200,
        "measurements": 30,
        "generation_ms": generation_ms,
        "read_and_serialize_p95_ms": float(np.percentile(timings, 95)),
        "summary_bytes": len(json.dumps(summary).encode()),
        "first_chart_bytes": chart_ref["size_bytes"],
        "store": str(store.root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "measurement": "warm local disk reads plus JSON serialization; excludes training, HTTP and browser rendering",
        "results": [
            benchmark(args.directory, 1000, 1000),
            benchmark(args.directory, 10000, 2000),
        ],
    }
    output = args.directory / "performance.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
