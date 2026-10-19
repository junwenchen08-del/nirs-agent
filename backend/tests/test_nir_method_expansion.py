"""Complete method references without weakening runtime execution boundaries."""

import copy
import json

import pytest


def test_packaged_cards_cover_all_preprocessors_and_bounded_model_families(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore, validate_card
    from nir_core.preprocess.registry import METHOD_REGISTRY

    catalog = MethodCatalogStore(tmp_path / "methods.sqlite3")
    cards = catalog.view()["cards"]
    assert len(cards) == len(catalog.defaults()["cards"])
    assert {c["method_id"] for c in cards if c["method_kind"] == "preprocessing"} == set(METHOD_REGISTRY)
    assert {c["method_id"] for c in cards if c["method_kind"] == "modeling"} == {"model_pls", "model_ridge", "model_svr", "model_extra_trees"}
    for card in cards:
        validate_card(card)
        assert card["applicability_zh"] and card["limitations_zh"] and card["parameter_guidance_zh"]
        assert card["review_status"] == "published"


@pytest.mark.parametrize(
    "query,expected",
    [
        ("孤立尖峰", "despike"),
        ("稳健SNV", "robust_snv"),
        ("二阶导数", "derivative2"),
        ("单位方差缩放", "autoscale"),
        ("岭回归", "model_ridge"),
        ("支持向量回归", "model_svr"),
        ("偏最小二乘", "model_pls"),
        ("极端随机树", "model_extra_trees"),
    ],
)
def test_new_references_are_searchable_in_chinese(query, expected):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    result = MethodKnowledgeBase().search(query)
    assert expected in {c["method_id"] for c in result["results"]}
    assert result["independent_study_count"] == 0


def test_model_cards_cannot_enter_preprocessing_diagnostic_results():
    from nir_core.knowledge.methods import MethodKnowledgeBase

    kb = MethodKnowledgeBase()
    result = kb.search("nonlinear regression noise baseline", profile={"tags": ["high_frequency_noise", "baseline_drift"]}, top_k=12)
    assert all(c["method_kind"] == "preprocessing" for c in result["results"])
    svr = next(c for c in kb.search("model_svr")["results"] if c["method_id"] == "model_svr")
    assert svr["auto_eligible"] is False
    assert svr["eligibility_reason"] == "model_reference_only"


@pytest.mark.parametrize("query,expected", [("岭回归", "model_ridge"), ("支持向量回归", "model_svr"), ("极端随机树", "model_extra_trees")])
def test_explicit_model_names_rank_before_generic_family_priority(query, expected):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    assert MethodKnowledgeBase().search(query)["results"][0]["method_id"] == expected


def test_native_cards_remain_explicit_and_model_parameter_edits_are_validated(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore

    store = MethodCatalogStore(tmp_path / "methods.sqlite3")
    native = next(c for c in store.view()["cards"] if c["method_id"] == "robust_snv")
    assert native["provider"] == "native" and native["auto_eligible"] is False
    store.update("model_ridge", {"candidate_params": [{"alpha": 10.0}]}, expected_revision=0, actor="qa")
    for params in ({"alpha": -1}, {"alpha": float("nan")}, {"alpha": True}, {"arbitrary": 1}):
        with pytest.raises(ValueError):
            store.update("model_ridge", {"candidate_params": [params]}, expected_revision=1, actor="qa")
    assert store.snapshot()["revision"] == 1
    with pytest.raises(ValueError):
        store.update("model_svr", {"source_url": "https://scikit-learn.org.evil.test/stable/modules/svm.html"}, expected_revision=1, actor="qa")


def test_expanded_defaults_merge_into_old_managed_snapshot_without_overwriting_edits(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore

    store = MethodCatalogStore(tmp_path / "methods.sqlite3")
    full = store.defaults()
    legacy = copy.deepcopy(full)
    legacy["cards"] = full["cards"][:11]
    for card in legacy["cards"]:
        for field in ("applicability_zh", "limitations_zh", "parameter_guidance_zh", "method_kind", "provider"):
            card.pop(field, None)
    old_path = tmp_path / "legacy.json"
    old_path.write_text(json.dumps(legacy), encoding="utf-8")
    old = MethodCatalogStore(store.path, defaults_path=old_path)
    old.update("sg_smooth", {"summary_zh": "User reviewed content", "review_status": "retired"}, expected_revision=0, actor="existing-admin")
    expanded = store.view()
    assert expanded["count"] == len(full["cards"])
    assert expanded["revision"] == 1
    sg = next(c for c in expanded["cards"] if c["method_id"] == "sg_smooth")
    assert sg["summary_zh"] == "User reviewed content" and sg["review_status"] == "retired"
    assert sg["applicability_zh"]
    store.update("model_svr", {"planning_notes_zh": "New review"}, expected_revision=1, actor="qa")
    assert store.view()["count"] == len(full["cards"])
    assert store.history()[-1]["actor"] == "existing-admin"
