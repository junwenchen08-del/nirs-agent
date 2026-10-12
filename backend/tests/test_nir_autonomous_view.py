"""Read-only evidence is projected from the same sealed session index."""

import copy
import json
from unittest.mock import patch

import pytest
from test_nir_delivery_export import final_runtime, invoke_final
from test_nir_delivery_freeze import stopped_session

from deerflow.community.nir.autonomous_view import AutonomousView
from deerflow.community.nir.process import _json


def test_invalid_metrics_are_missing_instead_of_fabricated_zero():
    from deerflow.community.nir.autonomous_view import scoped_metrics

    value = scoped_metrics({"val": {"scope": "validation", "RMSE": None, "R2": "bad", "MAE": float("nan"), "RPD": float("inf"), "n": 10}})
    assert value == {"val": {"scope": "validation", "n": 10}}


def test_old_process_index_has_no_autonomous_session(tmp_path):
    import sqlite3

    from deerflow.community.nir.process import ProcessStore

    store = ProcessStore(tmp_path)
    store.root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(store.db) as db:
        db.execute("CREATE TABLE old_evidence (id TEXT)")
    view = AutonomousView(store)
    assert view.sessions()["data"] == []
    with pytest.raises(KeyError):
        view.summary("search_unknown")


def test_sealed_view_has_actual_decisions_feedback_and_no_test(tmp_path):
    service, ctx, result, stop = stopped_session(tmp_path)
    view = AutonomousView(service.search.process)
    with patch("numpy.load", side_effect=AssertionError("read endpoint must not load arrays")):
        page = view.sessions()
        summary = view.summary(ctx["session_id"])
        timeline = view.decisions(ctx["session_id"])
        detail = view.decision(ctx["session_id"], stop["decision_id"])
    assert page["data"][0]["session_id"] == ctx["session_id"]
    assert summary["test_status"] == "sealed"
    assert "test" not in summary["metrics"]
    assert summary["retained_experiment_id"] == result["experiment_id"]
    assert timeline["total"] == 2
    assert timeline["data"][0]["feedback"]["metrics"]["cal_cv"]["scope"] == "calibration_cv"
    assert detail["proposal"]["action"] == "STOP"
    text = json.dumps([summary, timeline, detail])
    for forbidden in ["holdout.npz", "search.npz", "api_key", "args_sha256", str(tmp_path)]:
        assert forbidden not in text


def test_decision_session_binding_and_unknown_session(tmp_path):
    service, ctx, _result, stop = stopped_session(tmp_path)
    view = AutonomousView(service.search.process)
    with pytest.raises(KeyError):
        view.decision("search_unknown", stop["decision_id"])
    with pytest.raises(KeyError):
        view.summary("search_unknown")


def test_pagination_does_not_drop_or_repeat_rounds(tmp_path):
    service, ctx, _result, _stop = stopped_session(tmp_path)
    view = AutonomousView(service.search.process)
    first = view.decisions(ctx["session_id"], 0, 1)
    second = view.decisions(ctx["session_id"], first["next_offset"], 1)
    assert first["total"] == second["total"] == 2
    assert first["data"][0]["decision_id"] != second["data"][0]["decision_id"]
    assert second["next_offset"] is None


def test_final_view_reads_cached_test_points_and_downloads_only(tmp_path, monkeypatch):
    service, result, _stop, _output, runtime = final_runtime(tmp_path, monkeypatch)
    payload = invoke_final(runtime)
    sid = payload["session_id"]
    view = AutonomousView(service.search.process)
    with patch("numpy.load", side_effect=AssertionError("no Test replay")), patch.object(service.search, "finalize", side_effect=AssertionError("read must not finalize")):
        summary = view.summary(sid)
        points = view.chart(sid, "final_test")
        again = view.summary(sid)
    assert summary == again
    assert summary["freeze"]["experiment_id"] == result["experiment_id"]
    assert summary["metrics"]["test"]["scope"] == "final_test"
    assert points["scope"] == "final_test" and len(points["observed"]) == points["n_returned"]
    assert points["n_total"] == payload["metrics"]["test"]["n"]
    assert {item["name"] for item in summary["downloads"]} >= {"model.joblib", "report.md", "delivery.zip"}
    assert all(item["path"].startswith("mnt/user-data/outputs/") for item in summary["downloads"])
    assert summary["lifecycle"]["user_approved"] is False
    assert "charts" not in payload
    with service.search.process._connection() as db:
        session = service.search._session(db, sid)
    assert "charts" in session["final"]
    assert "charts" not in session["export"]
    publication = next(_output.rglob("publication.json"))
    assert "charts" not in json.loads(publication.read_text(encoding="utf-8"))


def test_more_than_twenty_rounds_are_pageable(tmp_path):
    service, ctx, _result, _stop = stopped_session(tmp_path)
    with service.search.process._connection(write=True) as db:
        row = db.execute("SELECT * FROM initial_decisions ORDER BY rowid LIMIT 1").fetchone()
        for index in range(3, 24):
            payload = copy.deepcopy(json.loads(row["payload"]))
            payload.update(decision_id=f"d_{index}", round_index=index)
            db.execute("INSERT INTO initial_decisions VALUES (?,?,?,'accepted',NULL)", (payload["decision_id"], ctx["session_id"], _json(payload)))
    view = AutonomousView(service.search.process)
    pages = [view.decisions(ctx["session_id"], offset, 10) for offset in (0, 10, 20)]
    identities = [item["decision_id"] for page in pages for item in page["data"]]
    assert len(identities) == len(set(identities)) == 23
    assert pages[-1]["next_offset"] is None


def test_rejected_model_issued_proposal_has_no_experiment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_nir_initial_decisions import issue

    from deerflow.community.nir import decision_tools

    service, ctx, _result, _stop = stopped_session(tmp_path)
    runtime = SimpleNamespace(state={"nir_workflow": {"project_id": "delivery", "search_session_id": ctx["session_id"], "stage": "execution"}}, tool_call_id="bad-submission", context={}, config={})
    plan = {"action": "TUNE", "rationale": "C:\\private\\file", "test": {"R2": 1.0}}
    issue(service, "nir_submit_decision", {"proposal": plan}, "bad-submission")
    monkeypatch.setattr(decision_tools, "_service", lambda _: service.search)
    decision_tools._error(runtime, ValueError("pipeline.alpha: must be positive"), rejected_proposal=plan)
    rejected = AutonomousView(service.search.process).rejections(ctx["session_id"])
    assert len(rejected["data"]) == 1
    assert rejected["data"][0]["model_executed"] is False
    assert "private" not in json.dumps(rejected) and "R2" not in json.dumps(rejected)
    assert AutonomousView(service.search.process).decisions(ctx["session_id"])["total"] == 2
