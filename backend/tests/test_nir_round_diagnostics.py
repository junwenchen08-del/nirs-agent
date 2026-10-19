"""Round charts retain actual predictions and never replay training on reads."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from test_nir_delivery_freeze import stopped_session

from deerflow.community.nir.autonomous_view import AutonomousView
from deerflow.community.nir.process import _json


def test_round_points_match_full_metrics_and_keep_test_sealed(tmp_path):
    service, ctx, result, _ = stopped_session(tmp_path)
    view = AutonomousView(service.search.process)
    before = service.search.process.db.read_bytes()
    with patch("numpy.load", side_effect=AssertionError("read cannot load data")), patch("joblib.load", side_effect=AssertionError("read cannot load a model")):
        points = view.diagnostics(ctx["session_id"], result["experiment_id"])
        assert view.diagnostics(ctx["session_id"], result["experiment_id"]) == points
    assert points["evidence_status"] == "available"
    assert set(points["charts"]) == {"calibration", "calibration_cv", "validation"}
    for name, scope in (("train", "calibration"), ("train_cv", "calibration_cv"), ("val", "validation")):
        chart = points["charts"][scope]
        residual = np.array(chart["predicted"]) - chart["observed"]
        np.testing.assert_allclose(residual, chart["residual"])
        assert np.sqrt(np.mean(residual**2)) == pytest.approx(result["metrics"][name]["RMSE"])
        assert chart["n_total"] == chart["n_returned"] == len(residual)
        assert len(set(chart["sample_ids"])) == len(residual)
    assert points["charts"]["calibration"]["sample_ids"] == points["charts"]["calibration_cv"]["sample_ids"]
    assert not set(points["charts"]["validation"]["sample_ids"]) & set(points["charts"]["calibration"]["sample_ids"])
    assert "test" not in points["charts"]
    assert view.summary(ctx["session_id"])["test_status"] == "sealed"
    assert service.search.process.db.read_bytes() == before


def test_missing_and_corrupt_old_evidence_never_recomputes(tmp_path):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid, eid = ctx["session_id"], result["experiment_id"]
    view = AutonomousView(service.search.process)
    path = service.search._dir(sid) / (eid + ".diagnostics.json")
    original = path.read_bytes()
    path.write_bytes(b"{}")
    assert view.diagnostics(sid, eid)["evidence_status"] == "unavailable"
    path.write_bytes(original)
    with service.search.process._connection(write=True) as db:
        stored = json.loads(db.execute("SELECT result FROM search_experiments WHERE id=?", (eid,)).fetchone()[0])
        stored.pop("diagnostics_sha256")
        db.execute("UPDATE search_experiments SET result=? WHERE id=?", (_json(stored), eid))
    with patch("numpy.load", side_effect=AssertionError("no historical reconstruction")):
        assert view.diagnostics(sid, eid)["evidence_status"] == "not_recorded"
    with pytest.raises(KeyError):
        view.diagnostics(sid, "exp_other")
    with pytest.raises(KeyError):
        view.diagnostics("search_other", eid)


@pytest.mark.parametrize("field,value", [("session_id", "search_other"), ("experiment_id", "exp_other"), ("decision_id", "d_other"), ("model_hash", "0" * 64)])
def test_hashed_cache_still_requires_exact_execution_identity(tmp_path, field, value):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid, eid = ctx["session_id"], result["experiment_id"]
    path = service.search._dir(sid) / (eid + ".diagnostics.json")
    payload = json.loads(path.read_bytes())
    payload[field] = value
    raw = _json(payload).encode()
    path.write_bytes(raw)
    with service.search.process._connection(write=True) as db:
        stored = json.loads(db.execute("SELECT result FROM search_experiments WHERE id=?", (eid,)).fetchone()[0])
        stored["diagnostics_sha256"] = hashlib.sha256(raw).hexdigest()
        db.execute("UPDATE search_experiments SET result=? WHERE id=?", (_json(stored), eid))
    assert AutonomousView(service.search.process).diagnostics(sid, eid)["evidence_status"] == "unavailable"


def test_display_cache_failure_preserves_successful_model_and_fit_counts(tmp_path, monkeypatch):
    from test_nir_delivery_decisions import setup_delivery
    from test_nir_initial_decisions import proposal, submit
    from test_nir_multiround_decisions import execute

    from deerflow.community.nir import round_diagnostics

    monkeypatch.setattr(round_diagnostics, "capture_round_diagnostics", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("display only")))
    service, ctx, _ = setup_delivery(tmp_path)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    assert result["status"] == "succeeded" and result["fits"] == 4
    assert result["diagnostics_error_type"] == "ValueError"
    assert AutonomousView(service.search.process).diagnostics(ctx["session_id"], result["experiment_id"])["evidence_status"] == "failed"


def test_bounded_points_preserve_full_population_metrics():
    from deerflow.community.nir.round_diagnostics import capture_round_diagnostics

    y = np.arange(5001, dtype=float)
    payload = capture_round_diagnostics(
        y, y[:100], {"calibration": y + 1, "calibration_cv": y + 2, "validation": y[:100] + 3}, session_id="search_a", experiment_id="exp_a", decision_id="d_a", model_hash="a" * 64, comparison_group_id="group_a"
    )
    chart = payload["charts"]["calibration_cv"]
    assert chart["n_total"] == 5001 and chart["n_returned"] == 2000
    assert chart["sampling"] != "full_population"
    assert chart["observed"][0] == 0 and chart["observed"][-1] == 5000


def test_two_real_rounds_keep_old_points_and_link_actual_next_decision(tmp_path):
    from test_nir_initial_decisions import proposal, submit
    from test_nir_multiround_decisions import execute, next_plan, setup_multi

    service, ctx, wf = setup_multi(tmp_path)
    first = execute(service, submit(service, ctx, proposal(ctx)))
    view = AutonomousView(service.search.process)
    a = view.diagnostics(ctx["session_id"], first["experiment_id"])
    next_context = service.context(ctx["session_id"], wf, "Tune based on actual feedback")
    second = execute(service, submit(service, next_context, next_plan(next_context), "second"), "execute-second")
    b = view.diagnostics(ctx["session_id"], second["experiment_id"])
    assert view.diagnostics(ctx["session_id"], first["experiment_id"]) == a
    assert a["comparison_group_id"] == b["comparison_group_id"]
    assert a["charts"]["validation"]["sample_ids"] == b["charts"]["validation"]["sample_ids"]
    assert a["charts"]["validation"]["predicted"] != b["charts"]["validation"]["predicted"]
    detail = view.decision(ctx["session_id"], first["decision_id"])
    assert detail["next_decision"]["decision_id"] == second["decision_id"]
    assert detail["next_decision"]["base_experiment_id"] == first["experiment_id"]
    assert view.decision(ctx["session_id"], second["decision_id"])["next_decision"] is None
    assert view.summary(ctx["session_id"])["data_context"]["n_features"] == 8


def test_real_round_browser_fixture_and_shared_contract(tmp_path):
    import jsonschema
    from test_nir_delivery_decisions import setup_delivery, stop_plan
    from test_nir_initial_decisions import proposal, submit
    from test_nir_multiround_decisions import execute, next_plan

    service, ctx, workflow = setup_delivery(tmp_path)
    sid = ctx["session_id"]
    first = execute(service, submit(service, ctx, proposal(ctx)))
    view = AutonomousView(service.search.process)
    initial = {"summary": view.summary(sid), "decisions": view.decisions(sid, limit=20), "details": {first["decision_id"]: view.decision(sid, first["decision_id"])}}
    following_context = service.context(sid, workflow, "Tune the existing model")
    second = execute(service, submit(service, following_context, next_plan(following_context), "second"), "execute-second")
    stop_context = service.context(sid, workflow, "Stop and retain the first model")
    submit(service, stop_context, stop_plan(stop_context, first["experiment_id"]), "stop")
    service.search.freeze(sid, first["experiment_id"])
    service.search.finalize(sid)
    rows = view.decisions(sid, limit=20)
    diagnostics = {item["experiment_id"]: view.diagnostics(sid, item["experiment_id"]) for item in (first, second)}
    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts" / "nir-round-diagnostics-v1.schema.json").read_text(encoding="utf-8"))
    for payload in diagnostics.values():
        jsonschema.validate(payload, schema)
    fixture = {
        "source": "Actual deterministic training on test-only synthetic arrays; model-issued receipts use the Fake test provider.",
        "summary": view.summary(sid),
        "decisions": rows,
        "details": {item["decision_id"]: view.decision(sid, item["decision_id"]) for item in rows["data"]},
        "diagnostics": diagnostics,
        "spectra": {item["experiment_id"]: view.spectra(sid, item["experiment_id"]) for item in (first, second)},
        "final_test": view.chart(sid, "final_test"),
        "initial": initial,
    }
    assert fixture["summary"]["retained_experiment_id"] == first["experiment_id"]
    assert fixture["decisions"]["data"][-1]["retained_experiment_id"] == first["experiment_id"]
    assert fixture["decisions"]["data"][-1]["base_experiment_id"] == second["experiment_id"]
    destination = os.environ.get("NIR_ROUND_QA_FIXTURE")
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_text(_json(fixture), encoding="utf-8")
