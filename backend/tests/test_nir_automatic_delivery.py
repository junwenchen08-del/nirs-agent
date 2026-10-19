"""Trusted terminal choices use the original numerical and publication services."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from test_nir_delivery_decisions import stop_plan
from test_nir_initial_decisions import proposal, submit
from test_nir_multiround_decisions import execute
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_budget import resolve_policy
from deerflow.community.nir.decision_context import InitialDecisionService
from deerflow.community.nir.process import ProcessStore


def terminal(tmp_path, *, source="llm"):
    search = SearchSessionStore(ProcessStore(tmp_path / "private"))
    from deerflow.community.nir.workflow import start_workflow

    workflow = start_workflow(
        task_type="calibration",
        project_id="auto",
        data_path="/mnt/user-data/uploads/a.npz",
        domain="default",
        analyte="protein",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        decision_strategy="llm_autonomous",
        autonomous_rounds=True,
        autonomous_delivery=True,
        budget_profile="unified",
    )
    policy = resolve_policy("unified", {"max_experiments": 1} if source == "budget" else {})
    workflow["delivery_policy"] = policy
    sid = search.prepare("auto", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy, task_snapshot={"domain": "default", "analyte": "protein", "unit": "%"})["session_id"]
    workflow["search_session_id"] = sid
    service = InitialDecisionService(search)
    ctx = service.context(sid, workflow, "Select a model")
    decision = submit(service, ctx, proposal(ctx))
    feedback = execute(service, decision)
    ctx = service.context(sid, workflow, "Decide next action")
    stopped = submit(service, ctx, stop_plan(ctx, feedback["experiment_id"], "RETAIN" if source == "budget" else "STOP"), "terminal-choice")
    workflow.update(stage="awaiting_final_confirmation", stop_decision_id=stopped["decision_id"], retained_experiment_id=feedback["experiment_id"], stop_source=source)
    runtime = SimpleNamespace(state={"nir_workflow": workflow}, config={}, context={"run_id": "real-test-run"}, tool_call_id="terminal-choice")
    return search, workflow, runtime, stopped, tmp_path / "outputs"


@pytest.mark.parametrize("source", ["llm", "budget"])
def test_terminal_delivery_is_deterministic_single_test_and_idempotent(tmp_path, monkeypatch, source):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, stop, output = terminal(tmp_path, source=source)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    original = np.load
    tests = []

    def load(path, *a, **k):
        if "holdout.npz" in str(path):
            tests.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", load)
    result, updated = complete_terminal_delivery(runtime, search, workflow)
    assert updated["stage"] == "review" and updated["approval_status"] == "pending"
    assert result["freeze"]["experiment_id"] == workflow["retained_experiment_id"]
    assert result["lifecycle"]["registered"] is False
    assert len(tests) == 1
    with search.process._connection() as db:
        session = search._session(db, workflow["search_session_id"])
        record = json.loads(db.execute("SELECT payload FROM initial_decisions WHERE id=?", (stop["decision_id"],)).fetchone()[0])
    assert session["automatic_delivery"]["status"] == "delivered"
    assert session["stop_source"] == source and session["fits"] == 4
    assert record == stop
    with search.process._connection() as db:
        durable_before = search._session(db, workflow["search_session_id"])
    assert complete_terminal_delivery(runtime, search, updated)[0] == result
    with search.process._connection() as db:
        assert search._session(db, workflow["search_session_id"]) == durable_before
    assert len(tests) == 1
    for name in ("model.joblib", "model.joblib.manifest.json", "metrics.json", "decisions.json", "report.md", "report.html", "delivery.zip", "publication.json"):
        assert (output / result["model"].removeprefix("/mnt/user-data/outputs/")).with_name(name).is_file()


def test_completed_test_can_recover_report_only(tmp_path, monkeypatch):
    from deerflow.community.nir import delivery, search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    build = delivery.build_delivery
    monkeypatch.setattr(delivery, "build_delivery", lambda *a, **k: (_ for _ in ()).throw(OSError("publication failed")))
    with pytest.raises(OSError):
        complete_terminal_delivery(runtime, search, workflow)
    with search.process._connection() as db:
        session = search._session(db, workflow["search_session_id"])
    assert session["final"] and session["stage"] == "finalized"
    original = np.load

    def no_test(path, *a, **k):
        assert "holdout.npz" not in str(path)
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", no_test)
    monkeypatch.setattr(delivery, "build_delivery", build)
    assert complete_terminal_delivery(runtime, search, workflow)[1]["stage"] == "review"


def test_concurrent_terminal_delivery_cannot_consume_test_twice(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    original = np.load
    started, release = threading.Event(), threading.Event()

    def load(path, *a, **k):
        if "holdout.npz" in str(path):
            started.set()
            assert release.wait(10)
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(complete_terminal_delivery, runtime, search, workflow)
        assert started.wait(10)
        with pytest.raises(ValueError, match="in-flight|audit"):
            complete_terminal_delivery(runtime, search, workflow)
        release.set()
        assert first.result()[1]["stage"] == "review"


def test_uncertain_test_lease_is_audit_required(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    search.freeze(workflow["search_session_id"], workflow["retained_experiment_id"])
    with search.process._connection(write=True) as db:
        session = search._session(db, workflow["search_session_id"])
        session["stage"] = "testing"
        search._save(db, session)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Uncertain Test cannot reopen arrays"))
    with pytest.raises(ValueError, match="audit"):
        complete_terminal_delivery(runtime, search, workflow)


def test_frozen_orphan_recovers_only_named_run(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools, web_entry
    from deerflow.community.nir.automatic_delivery import recover_orphan_terminal

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    monkeypatch.setattr(web_entry, "persist_unified_workflow", lambda *a, **k: None)
    search.freeze(workflow["search_session_id"], workflow["retained_experiment_id"])
    with search.process._connection(write=True) as db:
        session = search._session(db, workflow["search_session_id"])
        session["task_snapshot"]["_workflow_snapshot"] = workflow
        session["automatic_delivery"].update(lease={"id": "crashed", "run_id": "orphan"}, trigger_run_id="orphan")
        search._save(db, session)
    recover_orphan_terminal(runtime, search.process, "other-run")
    with search.process._connection() as db:
        assert search._session(db, session["id"])["stage"] == "frozen"
    recover_orphan_terminal(runtime, search.process, "orphan")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Completed orphan recovery is read-only"))
    recover_orphan_terminal(runtime, search.process, "orphan")
    with search.process._connection() as db:
        recovered = search._session(db, session["id"])
    assert recovered["automatic_delivery"]["status"] == "delivered" and recovered["fits"] == 4
    assert any(event["stage"] == "recovered" for event in recovered["automatic_delivery"]["events"])


@pytest.mark.parametrize("fault", ["cancelled", "unknown_publication", "model_hash", "retained_mismatch"])
def test_terminal_faults_fail_closed_without_test(tmp_path, monkeypatch, fault):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    if fault == "model_hash":
        search.freeze(workflow["search_session_id"], workflow["retained_experiment_id"])
        (search._dir(workflow["search_session_id"]) / (workflow["retained_experiment_id"] + ".joblib")).write_bytes(b"damaged")
    elif fault == "retained_mismatch":
        workflow["retained_experiment_id"] = "another_experiment"
    else:
        with search.process._connection(write=True) as db:
            session = search._session(db, workflow["search_session_id"])
            if fault == "cancelled":
                session["delivery_interrupted_by_user"] = True
            else:
                session["export_state"] = {"status": "building", "generation": "uncertain"}
            search._save(db, session)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Faults cannot open arrays"))
    with pytest.raises(ValueError):
        complete_terminal_delivery(runtime, search, workflow)
    with search.process._connection() as db:
        assert not search._session(db, workflow["search_session_id"])["final"]


def test_completed_corrupt_publication_requires_audit_without_retest(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    result, updated = complete_terminal_delivery(runtime, search, workflow)
    model = output / result["model"].removeprefix("/mnt/user-data/outputs/")
    model.with_name("report.html").write_bytes(b"damaged report")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Corrupt delivery must not reopen arrays"))
    with pytest.raises(ValueError, match="integrity"):
        complete_terminal_delivery(runtime, search, updated)
    with search.process._connection() as db:
        session = search._session(db, workflow["search_session_id"])
    assert session["stage"] == "finalized" and session["automatic_delivery"]["status"] == "audit_required"


def test_runtime_cancel_after_freeze_keeps_test_sealed(tmp_path, monkeypatch):
    from nir_core.io.resources import ResourceLimitError

    from deerflow.community.nir import search_tools
    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery

    search, workflow, runtime, _, output = terminal(tmp_path)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    calls = []

    def cancel(stage):
        calls.append(stage)
        if len(calls) == 2:
            raise ResourceLimitError("resource_cancelled", "User cancelled", stage=stage)

    monkeypatch.setattr(search_tools, "budget_for_runtime", lambda *a: SimpleNamespace(checkpoint=cancel))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Cancel after freeze must not open arrays"))
    with pytest.raises(ResourceLimitError):
        complete_terminal_delivery(runtime, search, workflow)
    with search.process._connection() as db:
        session = search._session(db, workflow["search_session_id"])
    assert session["freeze"] and not session["final"] and session["stop_source"] == "user"
    assert session["automatic_delivery"]["status"] == "cancelled"
