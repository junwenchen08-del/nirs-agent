"""Cal plots use the executed fitted state; reads cannot reopen data or refit."""

import json
from unittest.mock import patch

import joblib
import numpy as np
import pytest
from test_nir_delivery_decisions import setup_delivery
from test_nir_initial_decisions import proposal, submit
from test_nir_multiround_decisions import execute

from deerflow.community.nir.autonomous_view import AutonomousView


@pytest.mark.parametrize(
    "steps", [[], [{"method": "mean_center", "params": {}}], [{"method": "snv", "params": {}}, {"method": "mean_center", "params": {}}], [{"method": "sg_smooth", "params": {"window": 5, "order": 2}}, {"method": "snv", "params": {}}]]
)
def test_cached_curves_match_actual_cal_preprocessor_and_never_enter_feedback(tmp_path, steps):
    service, ctx, _ = setup_delivery(tmp_path)
    plan = proposal(ctx)
    plan["alternatives"][0]["pipeline"]["preprocessing"] = steps
    result = execute(service, submit(service, ctx, plan))
    sid, eid = ctx["session_id"], result["experiment_id"]
    root = service.search._dir(sid)
    with np.load(root / "search.npz", allow_pickle=False) as data:
        raw = data["X_cal"]
        wv = data.get("wv")
    model = joblib.load(root / (eid + ".joblib"))
    expected = model.preprocessor.transform(raw, wv)
    view = AutonomousView(service.search.process)
    before = service.search.process.db.read_bytes()
    with patch("numpy.load", side_effect=AssertionError("plots cannot read datasets")), patch("joblib.load", side_effect=AssertionError("plots cannot reload state")):
        cache = view.spectra(sid, eid)
        assert view.spectra(sid, eid) == cache
    assert service.search.process.db.read_bytes() == before
    assert cache["scope"] == "calibration" and cache["configuration"]["steps"] == steps
    assert cache["experiment_id"] == eid and cache["decision_id"] == result["decision_id"]
    rows = np.linspace(0, len(raw) - 1, min(12, len(raw)), dtype=int)
    for i, row in enumerate(rows):
        np.testing.assert_allclose(cache["before"]["series"][i]["values"], raw[row])
        np.testing.assert_allclose(cache["after"]["series"][i]["values"], expected[row])
    assert cache["before"]["x"] == cache["after"]["x"]
    assert cache["schema_version"] == 2 and len(cache["steps"]) == len(steps)
    assert (cache["steps"][-1]["chart"] if steps else cache["before"])["series"] == cache["after"]["series"]
    assert "series" not in json.dumps(result)
    assert view.summary(sid)["spectra_experiments"][0]["experiment_id"] == eid
    with pytest.raises(KeyError):
        view.spectra(sid, "exp_wrong")


def test_chart_bounds_keep_true_channel_order_and_reuse_state_without_fit():
    from types import SimpleNamespace

    from deerflow.community.nir.cal_spectra import capture_cal_spectra

    X = np.arange(50 * 1300, dtype=float).reshape(50, 1300)
    state = SimpleNamespace(transform=lambda X, wv: X + 2)
    model = SimpleNamespace(preprocessor=state)
    cache = capture_cal_spectra(X, np.linspace(2498, 1100, 1300), model, session_id="s", experiment_id="e", decision_id="d", configuration={"steps": []}, model_hash="hash")
    assert cache["before"]["n_total"] == 50 and cache["before"]["n_returned"] == 12
    assert cache["before"]["axis_returned"] == 512 and cache["before"]["axis_total"] == 1300
    assert cache["before"]["x"] == cache["after"]["x"]
    assert np.all(np.diff(cache["before"]["x"]) < 0)
    np.testing.assert_allclose(np.array(cache["after"]["series"][0]["values"]) - cache["before"]["series"][0]["values"], 2)


def test_old_or_tampered_cal_cache_never_reconstructs_spectra(tmp_path):
    service, ctx, _ = setup_delivery(tmp_path)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    sid, eid = ctx["session_id"], result["experiment_id"]
    path = service.search._dir(sid) / (eid + ".spectra.json")
    path.write_text("{}", encoding="utf-8")
    with patch("numpy.load", side_effect=AssertionError("no reconstruction")):
        assert AutonomousView(service.search.process).spectra(sid, eid)["evidence_status"] == "unavailable"
    with service.search.process._connection(write=True) as db:
        payload = json.loads(db.execute("SELECT result FROM search_experiments WHERE id=?", (eid,)).fetchone()[0])
        payload.pop("spectra_sha256")
        db.execute("UPDATE search_experiments SET result=? WHERE id=?", (json.dumps(payload), eid))
    assert AutonomousView(service.search.process).spectra(sid, eid)["evidence_status"] == "not_recorded"


def test_cumulative_steps_use_full_cal_fitted_center_not_display_subset():
    from types import SimpleNamespace

    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline

    from deerflow.community.nir.cal_spectra import capture_cal_spectra, validate_step_spectra

    X = np.random.default_rng(42).normal(size=(53, 17)) * np.linspace(1, 4, 17) + np.arange(53)[:, None]
    original = X.copy()
    configuration = {"steps": [{"method": "snv", "params": {}}, {"method": "mean_center", "params": {}}]}
    pipeline = PreprocessingPipeline([PreprocessingStep(**step) for step in configuration["steps"]], provider_policy="native").fit(X)
    with patch.object(pipeline, "fit", side_effect=AssertionError("No display fit")):
        cache = capture_cal_spectra(X, None, SimpleNamespace(preprocessor=pipeline), session_id="s", experiment_id="e", decision_id="d", configuration=configuration, model_hash="hash")
    rows = np.linspace(0, len(X) - 1, 12, dtype=int)
    snv = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    centered = snv - snv.mean(axis=0)
    for step, expected in zip(cache["steps"], [snv, centered]):
        np.testing.assert_allclose([item["values"] for item in step["chart"]["series"]], expected[rows])
        assert [item["id"] for item in step["chart"]["series"]] == [item["id"] for item in cache["before"]["series"]]
    np.testing.assert_array_equal(X, original)
    validate_step_spectra(cache, configuration)


def test_step_observer_never_falls_back_to_fitting_display_rows():
    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline([PreprocessingStep(method="mean_center")])
    with patch.object(pipeline, "apply", side_effect=AssertionError("No implicit preview fit")):
        with pytest.raises(ValueError, match="original fitted"):
            pipeline.transform(np.ones((3, 5)), on_step=lambda *args: None)


@pytest.mark.parametrize("damage", ["order", "method", "params", "samples", "axis", "nonfinite", "final"])
def test_rehashed_step_cache_still_rejects_wrong_step_or_execution(tmp_path, damage):
    import hashlib

    service, ctx, _ = setup_delivery(tmp_path)
    plan = proposal(ctx, steps=[{"method": "snv", "params": {}}, {"method": "mean_center", "params": {}}])
    result = execute(service, submit(service, ctx, plan))
    sid, eid = ctx["session_id"], result["experiment_id"]
    path = service.search._dir(sid) / (eid + ".spectra.json")
    payload = json.loads(path.read_bytes())
    step = payload["steps"][0]
    if damage == "order":
        step["index"] = 1
    elif damage == "method":
        step["method"] = "msc"
    elif damage == "params":
        step["params"] = {"other": 1}
    elif damage == "samples":
        step["chart"]["series"][0]["id"] = "wrong"
    elif damage == "axis":
        step["chart"]["x"][0] += 1
    elif damage == "nonfinite":
        step["chart"]["series"][0]["values"][0] = float("inf")
    else:
        payload["after"]["series"][0]["values"][0] += 1
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    with service.search.process._connection(write=True) as db:
        stored = json.loads(db.execute("SELECT result FROM search_experiments WHERE id=?", (eid,)).fetchone()[0])
        stored["spectra_sha256"] = hashlib.sha256(raw).hexdigest()
        db.execute("UPDATE search_experiments SET result=? WHERE id=?", (json.dumps(stored), eid))
    with patch("numpy.load", side_effect=AssertionError("no reconstruction")), patch("joblib.load", side_effect=AssertionError("no estimator")):
        assert AutonomousView(service.search.process).spectra(sid, eid)["evidence_status"] == "unavailable"


def test_step_chart_failure_does_not_change_success_or_fit_ledger(tmp_path):
    from deerflow.community.nir import cal_spectra

    service, ctx, _ = setup_delivery(tmp_path)
    with patch.object(cal_spectra, "capture_cal_spectra", side_effect=ValueError("display failure")):
        result = execute(service, submit(service, ctx, proposal(ctx)))
    assert result["status"] == "succeeded" and result["fits"] == 4
    assert result["spectra_error_type"] == "ValueError"
    view = AutonomousView(service.search.process)
    assert view.spectra(ctx["session_id"], result["experiment_id"])["evidence_status"] == "failed"
    assert view.summary(ctx["session_id"])["test_status"] == "sealed"


def test_v1_cache_remains_readable_without_intermediate_backfill(tmp_path):
    import hashlib

    service, ctx, _ = setup_delivery(tmp_path)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    sid, eid = ctx["session_id"], result["experiment_id"]
    path = service.search._dir(sid) / (eid + ".spectra.json")
    payload = json.loads(path.read_bytes())
    payload["schema_version"] = 1
    payload.pop("steps")
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    with service.search.process._connection(write=True) as db:
        stored = json.loads(db.execute("SELECT result FROM search_experiments WHERE id=?", (eid,)).fetchone()[0])
        stored["spectra_sha256"] = hashlib.sha256(raw).hexdigest()
        db.execute("UPDATE search_experiments SET result=? WHERE id=?", (json.dumps(stored), eid))
    with patch("numpy.load", side_effect=AssertionError("no historical backfill")):
        cache = AutonomousView(service.search.process).spectra(sid, eid)
    assert cache["schema_version"] == 1 and "steps" not in cache
