"""Freeze reuses the actual Cal artifact and only one durable Test lease."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_nir_delivery_decisions import setup_delivery, stop_plan
from test_nir_initial_decisions import proposal, submit
from test_nir_multiround_decisions import execute

from deerflow.community.nir._common import _sha256_file


def stopped_session(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    first = submit(service, ctx, proposal(ctx))
    result = execute(service, first)
    ctx = service.context(ctx["session_id"], wf, "Decide whether to stop")
    stop = submit(service, ctx, stop_plan(ctx, result["experiment_id"]), "stop")
    return service, ctx, result, stop


def test_cannot_freeze_autonomous_search_before_stop(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    with pytest.raises(ValueError, match="STOP"):
        service.search.freeze(ctx["session_id"], result["experiment_id"])


def test_freeze_binds_real_configuration_and_preserves_artifact(tmp_path):
    service, ctx, result, stop = stopped_session(tmp_path)
    sid = ctx["session_id"]
    path = service.search._dir(sid) / (result["experiment_id"] + ".joblib")
    before = path.read_bytes()
    frozen = service.search.freeze(sid, result["experiment_id"])
    assert frozen["stop_decision_id"] == stop["decision_id"]
    assert frozen["configuration"] == result["actual_configuration"]
    assert frozen["parameters"] == result["actual_parameters"]
    assert frozen["model_hash"] == _sha256_file(str(path))
    assert path.read_bytes() == before
    assert service.search.freeze(sid, result["experiment_id"]) == frozen
    with pytest.raises(ValueError):
        service.search.freeze(sid, "exp_other_session")
    with pytest.raises(ValueError, match="closed"):
        service.search.search(sid, "bypass", method="ridge", steps=[])


@pytest.mark.parametrize("corruption", ["artifact", "parameters", "cv", "ledger"])
def test_freeze_rejects_inconsistent_evidence(tmp_path, corruption):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid = ctx["session_id"]
    if corruption == "artifact":
        (service.search._dir(sid) / (result["experiment_id"] + ".joblib")).write_bytes(b"tampered")
    else:
        with service.search.process._connection(write=True) as db:
            if corruption == "ledger":
                session = service.search._session(db, sid)
                session["fits"] += 1
                service.search._save(db, session)
            else:
                row = db.execute("SELECT result FROM search_experiments WHERE id=?", (result["experiment_id"],)).fetchone()
                evidence = json.loads(row[0])
                if corruption == "parameters":
                    evidence["parameters"]["alpha"] = 999.0
                else:
                    evidence["cv_manifest"]["n_splits"] += 1
                db.execute("UPDATE search_experiments SET result=? WHERE id=?", (json.dumps(evidence), result["experiment_id"]))
    with pytest.raises(ValueError):
        service.search.freeze(sid, result["experiment_id"])


def test_parallel_freeze_and_finalize_are_single_use(tmp_path, monkeypatch):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid = ctx["session_id"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        freezes = list(pool.map(lambda _: service.search.freeze(sid, result["experiment_id"]), range(2)))
    assert freezes[0] == freezes[1]
    original = np.load
    opened = []

    def spy(path, *a, **k):
        if "holdout" in str(path):
            opened.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    final = service.search.finalize(sid)
    assert service.search.finalize(sid) == final and len(opened) == 1
    assert final["metrics"]["test"]["scope"] == "final_test"
    with pytest.raises(ValueError):
        service.context(sid, {}, "Reopen after Test")


def test_interrupted_test_is_not_reissued_after_restart(tmp_path, monkeypatch):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid = ctx["session_id"]
    service.search.freeze(sid, result["experiment_id"])
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        service.search.finalize(sid)
    with pytest.raises(ValueError, match="consumed"):
        service.search.finalize(sid)
