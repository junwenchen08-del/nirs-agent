"""Process evidence must remain truthful, immutable and isolated from summaries."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from deerflow.community.nir.process import ProcessStore, comparison_key, regression_evidence


def test_attempts_are_allocated_before_results_and_replay_is_idempotent(tmp_path):
    store = ProcessStore(tmp_path)
    first = store.begin("project", "call-1", "nir_analyze")
    assert store.summary(first["run_id"])["attempts"][0]["execution_status"] == "running"
    assert store.begin("project", "call-1", "nir_analyze") == first
    second = store.begin("project", "call-2", "nir_analyze")
    assert second["number"] == 2
    store.finish(first, "succeeded", {"stage": "review"})
    store.finish(first, "failed", {"stage": "blocked"})
    assert store.summary(first["run_id"])["attempts"][0]["execution_status"] == "succeeded"


def test_concurrent_begin_keeps_unique_attempt_numbers(tmp_path):
    store = ProcessStore(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda index: store.begin("project", f"call-{index}", "nir_analyze"), range(24)))
    assert sorted(item["number"] for item in attempts) == list(range(1, 25))
    oldest = next(item for item in attempts if item["number"] == 1)
    assert store.summary(oldest["run_id"])["next_offset"] == 20
    old_page = store.summary(oldest["run_id"], attempt_id=oldest["attempt_id"])
    assert old_page["offset"] == 20
    assert [item["number"] for item in old_page["attempts"]] == [1, 2, 3, 4]


def test_evidence_hash_and_cross_run_membership_are_checked(tmp_path):
    store = ProcessStore(tmp_path)
    attempt = store.begin("project", "call", "nir_analyze")
    store.publish(attempt, [{"step_key": "audit", "execution_status": "succeeded", "charts": [{"type": "distribution", "x": [1], "y": [2], "n_total": 2, "n_returned": 1, "sampling": "full_population_histogram_v1"}]}])
    summary = store.summary(attempt["run_id"])
    step = store.step(attempt["run_id"], summary["attempts"][0]["steps"][0]["step_execution_id"])
    ref = step["charts"][0]
    assert store.chart(attempt["run_id"], ref["chart_id"], ref["version"])["type"] == "distribution"
    with pytest.raises(KeyError):
        store.chart("other-run", ref["chart_id"], ref["version"])
    with pytest.raises(ValueError, match="version"):
        store.chart(attempt["run_id"], ref["chart_id"], "wrong")
    artifact = next((Path(tmp_path) / "charts").glob("*.json"))
    artifact.write_text("{}", encoding="utf-8")
    assert store.chart(attempt["run_id"], ref["chart_id"], ref["version"])["evidence_status"] == "unavailable"


def test_more_than_eight_candidates_survive_in_step_details(tmp_path):
    store = ProcessStore(tmp_path)
    attempt = store.begin("project", "call", "nir_analyze")
    candidates = [{"candidate_id": str(i), "score": i / 10} for i in range(200)]
    store.publish(attempt, [{"step_key": "model", "execution_status": "succeeded", "candidates": candidates}])
    sid = store.summary(attempt["run_id"])["attempts"][0]["steps"][0]["step_execution_id"]
    assert len(store.candidates(attempt["run_id"], sid, offset=0)["data"]) == 50
    assert store.candidates(attempt["run_id"], sid, offset=150)["data"][-1]["candidate_id"] == "199"


def test_unknown_or_different_evaluation_context_is_not_comparable():
    context = {"dataset_hash": "a", "interpretation_hash": "b", "target": "protein", "unit": "%", "protocol": "holdout-v1", "split_hash": "c", "evaluation_kind": "tuning", "evaluation_hash": "d", "metric_definition_version": "rmse-v1"}
    assert comparison_key(context) == comparison_key(dict(context))
    assert comparison_key({**context, "split_hash": "different"}) != comparison_key(context)
    assert comparison_key({**context, "evaluation_kind": "holdout"}) is None
    assert comparison_key({**context, "unit": None}) is None


def test_real_regression_evidence_preserves_metrics_and_linked_sample_identity():
    rng = np.random.RandomState(17)
    X = rng.normal(size=(100, 30))
    y = np.arange(100, dtype=float)
    parts = {"calibration": (X[:70], y[:70]), "tuning": (X[70:80], y[70:80]), "holdout": (X[80:], y[80:])}
    metrics = {
        "protocol": "holdout-v1",
        "validation_scope": "independent_holdout_not_external",
        "training_data_hash": "a" * 64,
        "method": "pls",
        "val": {"RMSE": 0.5},
        "test": {"RMSE": 1.0},
        "quality": {"passed": True},
        "wavelength_selection": {"method": "none"},
    }
    steps = regression_evidence(X=X, y=y, wv=np.arange(30), partitions=parts, processed_cal=X[:70] * 2, predictions={"holdout": y[80:] + 1}, metrics=metrics, pipeline=["snv"], unit="%", target="protein", namespace="thread-a", point_limit=5)
    diagnosis = next(step for step in steps if step["step_key"] == "validation")
    scatter = next(chart for chart in diagnosis["charts"] if chart["type"] == "prediction")
    residual = next(chart for chart in diagnosis["charts"] if chart["type"] == "residual")
    assert scatter["n_total"] == 20 and scatter["n_returned"] == 5
    assert scatter["sample_ids"] == residual["sample_ids"]
    assert all(value == 1 for value in residual["y"])
    assert diagnosis["metrics"]["test"]["RMSE"] == 1.0
    model = next(step for step in steps if step["step_key"] == "model")
    assert model["comparison_group_id"]
    changed = dict(parts)
    changed["tuning"] = (X[80:90], y[80:90])
    other = regression_evidence(X=X, y=y, wv=np.arange(30), partitions=changed, processed_cal=X[:70], predictions={}, metrics=metrics, unit="%", target="protein", namespace="thread-a")
    assert next(step for step in other if step["step_key"] == "model")["comparison_group_id"] != model["comparison_group_id"]


def test_chart_budget_and_path_identifiers(tmp_path):
    store = ProcessStore(tmp_path)
    with pytest.raises(ValueError):
        store.summary("../bad")
    attempt = store.begin("p", "call", "nir_analyze")
    with pytest.raises(ValueError, match="size"):
        store.publish(attempt, [{"step_key": "audit", "charts": [{"type": "spectra", "data": "x" * (1024 * 1024 + 1)}]}])


def test_invalid_vector_evidence_never_becomes_available(tmp_path):
    store = ProcessStore(tmp_path)
    attempt = store.begin("p", "call", "nir_analyze")
    chart = {"type": "prediction", "x": [1, 2], "y": [1], "sample_ids": ["a", "b"], "n_total": 2, "n_returned": 2, "sampling": "all"}
    with pytest.raises(ValueError, match="length"):
        store.publish(attempt, [{"step_key": "validation", "charts": [chart]}])
    assert not store.summary(attempt["run_id"])["attempts"][0]["steps"]


def test_restart_recovery_is_scoped_to_the_orphaned_agent_run(tmp_path):
    store = ProcessStore(tmp_path)
    old = store.begin("p", "old-call", "nir_analyze", agent_run_id="old-run")
    live = store.begin("p", "live-call", "nir_analyze", agent_run_id="live-run")
    store.recover("old-run")
    store.recover("old-run")
    attempts = store.summary(old["run_id"])["attempts"]
    assert [item["execution_status"] for item in attempts] == ["interrupted", "running"]
    store.finish(live, "cancelled", {"stage": "execution"})
    assert store.summary(live["run_id"])["attempts"][1]["execution_status"] == "cancelled"


def test_shared_contract_examples_validate():
    from jsonschema import Draft202012Validator

    contracts = Path(__file__).resolve().parents[2] / "contracts"
    validator = Draft202012Validator(json.loads((contracts / "nir-process-v1.schema.json").read_text(encoding="utf-8")))
    examples = json.loads((contracts / "nir-process-v1.examples.json").read_text(encoding="utf-8"))
    assert len(examples) == 6
    for example in examples:
        validator.validate(example["payload"])


def test_v1_index_upgrade_preserves_finished_attempts(tmp_path):
    store = ProcessStore(tmp_path)
    attempt = store.begin("p", "call", "nir_analyze")
    store.publish(attempt, [{"step_key": "model", "facts": {"method": "pls"}}])
    store.finish(attempt, "succeeded", {"stage": "review"})
    with sqlite3.connect(store.db) as connection:
        connection.execute("DROP TABLE agent_runs")
        connection.execute("PRAGMA user_version=1")
    snapshot = ProcessStore(tmp_path).summary(attempt["run_id"])
    assert snapshot["attempts"][0]["execution_status"] == "succeeded"
    assert len(snapshot["attempts"][0]["steps"]) == 1
    with sqlite3.connect(store.db) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2


def test_descending_axis_selection_retains_correct_intervals():
    from deerflow.community.nir.process import _selected_ranges

    assert _selected_ranges([0, 1, 3], np.asarray([1000, 900, 800, 700])) == [{"start": 900.0, "end": 1000.0, "count": 2}, {"start": 700.0, "end": 700.0, "count": 1}]
