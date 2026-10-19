"""An opaque phase-2 proposal replays after new optional fields are added."""

import json

from test_nir_initial_decisions import proposal, setup, submit
from test_nir_multiround_decisions import execute

from deerflow.community.nir.process import _digest


def test_phase2_saved_proposal_replays_without_rewriting_original_payload(tmp_path):
    service, context = setup(tmp_path)
    plan = proposal(context)  # The original phase-2 wire contract.
    legacy = submit(service, context, plan)
    legacy["proposal"] = plan
    legacy["proposal_sha256"] = _digest(plan)
    for name in ("round_index", "action", "base_experiment_id", "retained_experiment_id", "execution_signature"):
        legacy.pop(name, None)
    # Simulate an existing v4 payload. Migration preservation has a separate test.
    payload = json.dumps(legacy, ensure_ascii=False)
    with service.search.process._connection(write=True) as db:
        db.execute("UPDATE initial_decisions SET payload=? WHERE id=?", (payload, legacy["decision_id"]))
        session = service.search._session(db, context["session_id"])
        for name in ("autonomous_rounds", "max_decisions", "active_decision_id"):
            session.pop(name, None)
        service.search._save(db, session)
    assert submit(service, context, plan) == legacy
    assert execute(service, legacy)["status"] == "succeeded"
    with service.search.process._connection() as db:
        assert db.execute("SELECT payload FROM initial_decisions WHERE id=?", (legacy["decision_id"],)).fetchone()[0] == payload
