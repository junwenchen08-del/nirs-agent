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


@pytest.mark.parametrize("steps", [[], [{"method": "mean_center", "params": {}}], [{"method": "sg_smooth", "params": {"window": 5, "order": 2}}, {"method": "snv", "params": {}}]])
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
