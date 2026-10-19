"""Managed method edits must persist and reach training without image changes."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def store(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore

    return MethodCatalogStore(tmp_path / "methods.sqlite3")


def test_default_is_packaged_snapshot_and_read_does_not_create_store(tmp_path):
    catalog = store(tmp_path)
    snapshot = catalog.snapshot()
    assert snapshot["revision"] == 0
    assert len(snapshot["payload"]["cards"]) == len(catalog.defaults()["cards"])
    assert not catalog.path.exists()


def test_edit_is_durable_searchable_and_keeps_old_evidence_immutable(tmp_path):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    catalog = store(tmp_path)
    original = MethodKnowledgeBase(payload=catalog.snapshot()["payload"]).search("sg smooth")
    sg = next(card for card in original["results"] if card["method_id"] == "sg_smooth")
    old_hash = sg["content_sha256"]
    catalog.update("sg_smooth", {"summary_zh": "Reviewed updated description", "candidate_params": [{"window": 13, "order": 2}]}, expected_revision=0, actor="admin")
    persisted = store(tmp_path).snapshot()
    assert persisted["revision"] == 1
    updated = MethodKnowledgeBase(payload=persisted["payload"]).search("sg smooth")
    card = next(card for card in updated["results"] if card["method_id"] == "sg_smooth")
    assert card["candidate_params"] == [{"window": 13, "order": 2}]
    assert card["content_sha256"] != old_hash
    assert sg["content_sha256"] == old_hash
    assert catalog.history()[0]["actor"] == "admin"


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_params": [{"window": 10, "order": 2}]},
        {"candidate_params": [{"window": 7, "order": 8}]},
        {"candidate_params": [{"invented": 1}]},
        {"provider_class": "arbitrary.module.Class"},
        {"problem_tags": ["invented_tag"]},
        {"source_url": "https://chemotools.org.evil.test/methods/x"},
        {"candidate_params": [{"window": float("inf")}]},
    ],
)
def test_invalid_edit_does_not_change_catalog(tmp_path, changes):
    catalog = store(tmp_path)
    with pytest.raises(ValueError):
        catalog.update("sg_smooth", changes, expected_revision=0, actor="admin")
    assert catalog.snapshot()["revision"] == 0


def test_retire_and_restore_defaults_take_effect_without_restarting(tmp_path):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    catalog = store(tmp_path)
    catalog.update("sg_smooth", {"review_status": "retired"}, expected_revision=0, actor="admin")
    assert "sg_smooth" not in {card["method_id"] for card in MethodKnowledgeBase(payload=catalog.snapshot()["payload"]).search("noise smoothing")["results"]}
    catalog.reset("sg_smooth", expected_revision=1, actor="admin")
    assert "sg_smooth" in {card["method_id"] for card in MethodKnowledgeBase(payload=catalog.snapshot()["payload"]).search("noise smoothing")["results"]}


def test_stale_concurrent_edit_cannot_overwrite_a_saved_card(tmp_path):
    from nir_core.knowledge.method_store import CatalogConflict

    catalog = store(tmp_path)

    def save(title):
        try:
            catalog.update("sg_smooth", {"title": title}, expected_revision=0, actor="admin")
            return True
        except CatalogConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, ["one", "two"])) == [False, True]
    assert catalog.snapshot()["revision"] == 1


def test_new_card_can_only_bind_an_installed_method(tmp_path):
    catalog = store(tmp_path)
    from nir_core.knowledge.method_store import MethodCatalogStore

    defaults = catalog.defaults()
    defaults["cards"] = [card for card in defaults["cards"] if card["method_id"] != "derivative2"]
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(defaults), encoding="utf-8")
    catalog = MethodCatalogStore(catalog.path, defaults_path=seed)
    with pytest.raises(ValueError, match="method"):
        catalog.update("invented", {"title": "Unknown"}, expected_revision=0, actor="admin")
    changes = {
        "title": "Second derivative",
        "review_status": "draft",
        "source_url": "https://chemotools.org/methods/index.html",
        "source_section": "Methods",
        "summary_zh": "Reviewed derivative reference",
        "problem_tags": ["baseline_drift"],
    }
    changes.update(keywords="derivative 二阶导数", candidate_params=[{}], planning_notes_zh="Review before publication")
    catalog.update("derivative2", changes, expected_revision=0, actor="admin")
    assert len(catalog.snapshot()["payload"]["cards"]) == len(defaults["cards"]) + 1


def test_api_and_training_read_the_same_managed_catalog(tmp_path, monkeypatch):
    from app.gateway.routers import method_knowledge
    from deerflow.community.nir import method_catalog
    from deerflow.community.nir.knowledge import nir_search_method_knowledge_tool

    catalog = store(tmp_path)
    monkeypatch.setattr(method_catalog, "get_method_store", lambda: catalog)
    app = FastAPI()

    @app.middleware("http")
    async def admin(request: Request, call_next):
        request.state.user = SimpleNamespace(system_role="admin", id="admin-account-42")
        return await call_next(request)

    app.include_router(method_knowledge.router)
    client = TestClient(app)
    assert client.get("/api/method-knowledge/cards").json()["count"] == len(catalog.defaults()["cards"])
    response = client.put("/api/method-knowledge/cards/sg_smooth", json={"expected_revision": 0, "changes": {"candidate_params": [{"window": 13, "order": 2}]}})
    assert response.status_code == 200
    assert response.json()["history"][0]["actor"] == "admin-account-42"
    assert client.put("/api/method-knowledge/cards/sg_smooth", json={"expected_revision": 0, "changes": {"title": "stale"}}).status_code == 409
    result = json.loads(nir_search_method_knowledge_tool.func(runtime=None, query="sg smooth"))
    assert next(card for card in result["results"] if card["method_id"] == "sg_smooth")["candidate_params"] == [{"window": 13, "order": 2}]
    import nir_core.preprocess.recommendation as recommendation

    from deerflow.community.nir import _preprocessing_plan

    observed = []

    def stop_after_retrieval(*args, **kwargs):
        observed.append(kwargs["knowledge_base"].search("sg smooth"))
        raise RuntimeError("stop before fitting")

    monkeypatch.setattr(recommendation, "recommend_preprocessing", stop_after_retrieval)
    with pytest.raises(RuntimeError, match="stop before fitting"):
        _preprocessing_plan.select_preprocessing(None, None, None, None, None)
    assert next(card for card in observed[0]["results"] if card["method_id"] == "sg_smooth")["candidate_params"] == [{"window": 13, "order": 2}]


def test_non_admin_cannot_modify_shared_method_knowledge(tmp_path, monkeypatch):
    from app.gateway.routers import method_knowledge
    from deerflow.community.nir import method_catalog

    catalog = store(tmp_path)
    monkeypatch.setattr(method_catalog, "get_method_store", lambda: catalog)
    app = FastAPI()

    @app.middleware("http")
    async def member(request: Request, call_next):
        request.state.user = SimpleNamespace(system_role="member", user_id="member")
        return await call_next(request)

    app.include_router(method_knowledge.router)
    client = TestClient(app)
    assert client.get("/api/method-knowledge/cards").json()["can_edit"] is False
    assert client.put("/api/method-knowledge/cards/sg_smooth", json={"expected_revision": 0, "changes": {"title": "change"}}).status_code == 403
    assert catalog.snapshot()["revision"] == 0


def test_managed_catalog_failure_preserves_training_rule_fallback(monkeypatch):
    import numpy as np
    from nir_core.preprocess.recommendation import recommend_preprocessing

    from deerflow.community.nir import method_catalog

    def unavailable():
        raise OSError("store unavailable")

    monkeypatch.setattr(method_catalog, "get_method_store", unavailable)
    rng = np.random.default_rng(22)
    axis = np.linspace(-1, 1, 101)
    spectra = 1 + np.exp(-(axis**2))[None, :] + rng.normal(0, 0.1, (30, 101))
    result = recommend_preprocessing(spectra, knowledge_mode="auto", knowledge_base=method_catalog.get_method_knowledge_base())
    assert result.method_knowledge["mode"] == "rule_fallback"
    assert result.method_knowledge["reason"] == "method_retrieval_unavailable"
    assert any(item.candidate_id == "raw" for item in result.candidates)


def test_descriptive_edit_does_not_claim_a_new_source_fetch_date(tmp_path):
    catalog = store(tmp_path)
    old = next(card for card in catalog.view()["cards"] if card["method_id"] == "snv")
    catalog.update("snv", {"planning_notes_zh": "Updated project note"}, expected_revision=0, actor="admin")
    new = next(card for card in catalog.view()["cards"] if card["method_id"] == "snv")
    assert new["retrieved_on"] == old["retrieved_on"]
    assert catalog.history()[0]["changed_at"]


def test_all_saved_parameter_variants_are_considered_within_the_candidate_budget(tmp_path):
    import numpy as np
    from nir_core.knowledge.methods import MethodKnowledgeBase
    from nir_core.preprocess.recommendation import recommend_preprocessing

    payload = store(tmp_path).defaults()
    payload["cards"] = [card for card in payload["cards"] if card["method_id"] == "sg_smooth"]
    payload["cards"][0]["candidate_params"] = [{"window": window, "order": 2} for window in (5, 7, 9, 11, 13, 15)]
    rng = np.random.default_rng(219)
    axis = np.linspace(-1, 1, 101)
    spectra = 1 + np.exp(-(axis**2))[None, :] + rng.normal(0, 0.1, (30, 101))
    result = recommend_preprocessing(spectra, knowledge_mode="auto", knowledge_base=MethodKnowledgeBase(payload=payload))
    windows = {candidate.steps[0].params["window"] for candidate in result.candidates if candidate.evidence_ids}
    assert windows == {5, 7, 9, 11, 13, 15}
    assert len(result.candidates) <= 8


def test_editing_reference_tags_cannot_remove_the_runtime_low_snr_guard(tmp_path):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    payload = store(tmp_path).defaults()
    card = next(card for card in payload["cards"] if card["method_id"] == "derivative1")
    card["avoid_tags"] = []
    result = MethodKnowledgeBase(payload=payload).search("derivative baseline", profile={"tags": ["baseline_drift", "low_signal_to_noise"]})
    derivative = next(card for card in result["results"] if card["method_id"] == "derivative1")
    assert derivative["auto_eligible"] is False
