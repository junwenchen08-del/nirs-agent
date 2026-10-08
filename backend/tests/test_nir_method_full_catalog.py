"""Official-site coverage stays separate from automatic pipeline execution."""

import copy
import json

import pytest


def test_full_official_and_installed_mcp_inventory_is_covered(tmp_path):
    from nir_core.chemotools_mcp.catalog import build_capability_catalog
    from nir_core.knowledge.method_store import MethodCatalogStore, validate_card

    store = MethodCatalogStore(tmp_path / "methods.sqlite3")
    view = store.view()
    assert view["count"] == 94
    coverage = view["coverage"]
    assert coverage["official_document_count"] == 70
    assert coverage["official_covered_count"] == 70
    assert coverage["mcp_capability_count"] == 84
    assert coverage["mcp_covered_count"] == 84
    ids = {c.get("mcp_capability_id") for c in view["cards"]}
    assert {c["id"] for c in build_capability_catalog()["capabilities"]} <= ids
    for card in view["cards"]:
        validate_card(card)
        assert card["summary_zh"] and card["applicability_zh"] and card["limitations_zh"]
    json.dumps(view, allow_nan=False)
    assert not store.path.exists()


@pytest.mark.parametrize(
    "query,capability",
    [
        ("VIPSelector", "chemotools.feature_selection.VIPSelector"),
        ("仪器迁移", "chemotools.adaptation.DirectStandardization"),
        ("数据增强 加噪", "chemotools.augmentation.AddNoise"),
        ("HotellingT2", "chemotools.outliers.HotellingT2"),
        ("残差绘图", "chemotools.plotting.YResidualsPlot"),
    ],
)
def test_new_categories_can_be_retrieved_with_chinese_and_algorithm_names(query, capability):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    result = MethodKnowledgeBase().search(query, top_k=12)
    assert capability in {card.get("mcp_capability_id") for card in result["results"]}
    assert result["independent_study_count"] == 0


def test_catalog_parameters_are_exact_read_only_mcp_references(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore

    store = MethodCatalogStore(tmp_path / "methods.sqlite3")
    cards = store.view()["cards"]
    vip = next(c for c in cards if c.get("mcp_capability_id") == "chemotools.feature_selection.VIPSelector")
    assert vip["method_kind"] == "mcp_reference" and not vip["auto_eligible"]
    assert "model" in vip["mcp_parameters"]["required"]
    assert vip["mcp_parameters"]["properties"]["threshold"]["default"] == 1.0
    assert vip["runtime_parameters"] == {}
    with pytest.raises(ValueError, match="MCP"):
        store.update(vip["method_id"], {"candidate_params": [{"threshold": 99}]}, expected_revision=0, actor="qa")
    with pytest.raises(ValueError):
        store.update(vip["method_id"], {"mcp_parameters": {}}, expected_revision=0, actor="qa")
    assert store.snapshot()["revision"] == 0


def test_documented_but_unexposed_function_remains_searchable_without_false_execution():
    from nir_core.knowledge.methods import MethodKnowledgeBase

    cards = MethodKnowledgeBase().search("check_metadata_function")["results"]
    card = next(c for c in cards if c["provider_class"] == "chemotools.adaptation.validation.check_metadata_function")
    assert not card["mcp_execution_supported"] and not card["auto_eligible"]
    assert card["eligibility_reason"] == "mcp_unavailable"


def test_full_catalog_cannot_displace_automatic_preprocessing_candidates():
    from nir_core.knowledge.methods import MethodKnowledgeBase

    results = MethodKnowledgeBase().search("noise scatter drift augmentation baseline", top_k=12, profile={"tags": ["high_frequency_noise", "scatter_variation", "baseline_drift"]})["results"]
    assert results and all(c["method_kind"] == "preprocessing" for c in results)


def test_merge_preserves_existing_cards_and_adds_full_catalog_without_read_writes(tmp_path):
    from nir_core.knowledge.method_store import MethodCatalogStore

    store = MethodCatalogStore(tmp_path / "methods.sqlite3")
    payload = copy.deepcopy(store.defaults())
    payload["cards"] = payload["cards"][:25]
    seed = tmp_path / "old.json"
    seed.write_text(json.dumps(payload), encoding="utf-8")
    old = MethodCatalogStore(store.path, defaults_path=seed)
    old.update("sg_smooth", {"summary_zh": "Custom review", "review_status": "retired"}, expected_revision=0, actor="admin")
    before = store.path.read_bytes()
    view = store.view()
    assert view["coverage"]["official_covered_count"] == 70
    assert view["revision"] == 1 and store.path.read_bytes() == before
    sg = next(c for c in view["cards"] if c["method_id"] == "sg_smooth")
    assert sg["summary_zh"] == "Custom review" and sg["review_status"] == "retired"
    assert store.history()[0]["actor"] == "admin"


def test_mcp_reference_cannot_bind_an_arbitrary_python_object():
    from nir_core.knowledge.method_registry import runtime_reference

    with pytest.raises(ValueError):
        runtime_reference("chemotools.os.system")
