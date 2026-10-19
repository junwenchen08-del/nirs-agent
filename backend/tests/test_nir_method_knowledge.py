"""Official method retrieval must change candidates without changing evidence gates."""

import json
from pathlib import Path

import numpy as np
import pytest


def noisy_calibration():
    rng = np.random.default_rng(219)
    axis = np.linspace(-1, 1, 101)
    peak = np.exp(-(((axis - 0.2) / 0.18) ** 2))
    return 1 + rng.uniform(0.7, 1.3, (40, 1)) * peak + rng.normal(0, 0.1, (40, 101))


@pytest.mark.parametrize("query", ["高频噪声 降噪 平滑", "high frequency noise smoothing"])
def test_bilingual_noise_search_returns_cited_constraints_and_runtime_schema(query):
    from nir_core.knowledge.methods import MethodKnowledgeBase

    result = MethodKnowledgeBase().search(query)
    assert result["retrieval_strategy"] == "structured_fts5_bm25"
    assert result["evidence_kind"] == "official_method_reference"
    sg = next(item for item in result["results"] if item["method_id"] == "sg_smooth")
    assert sg["source_url"].startswith("https://chemotools.org/")
    assert sg["content_sha256"] and sg["evidence_id"]
    assert sg["runtime_parameters"]["window"]["odd"]
    assert sg["parameter_mapping"]["window"] == "window_length"
    assert sg["parameter_policy"] == "project_bounded_search_not_official_optimum"
    assert result["independent_study_count"] == 0


def test_unrelated_query_abstains_instead_of_returning_generic_methods():
    from nir_core.knowledge.methods import MethodKnowledgeBase

    result = MethodKnowledgeBase().search("weather forecast tomorrow")
    assert result["results"] == []
    assert result["abstained"] is True
    assert result["reason"] == "no_relevant_method_reference"


def test_drafts_and_incompatible_provider_versions_cannot_drive_planning(tmp_path):
    from nir_core.knowledge.methods import DEFAULT_CARDS_PATH, MethodKnowledgeBase

    payload = json.loads(DEFAULT_CARDS_PATH.read_text(encoding="utf-8"))
    for card in payload["cards"]:
        card["review_status"] = "draft"
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert MethodKnowledgeBase(path).search("noise")["results"] == []
    for card in payload["cards"]:
        card["review_status"] = "published"
        card["runtime_provider_version"] = "99.0.0"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = MethodKnowledgeBase(path).search("noise")
    assert result["results"] == []
    assert any(item["reason"] == "runtime_version_mismatch" for item in result["exclusions"])


def test_noise_retrieval_prioritizes_valid_smoothing_before_same_batch_raw_control():
    from nir_core.preprocess.pipeline import validate_pipeline
    from nir_core.preprocess.recommendation import recommend_preprocessing

    recommendation = recommend_preprocessing(noisy_calibration(), np.linspace(900, 1700, 101), knowledge_mode="auto", budget="small")
    payload = recommendation.as_dict()
    assert payload["method_knowledge"]["mode"] == "retrieval_guided"
    assert payload["method_knowledge"]["profile_scope"] == "calibration_only"
    assert recommendation.candidates[0].steps[0].method == "sg_smooth"
    assert any(item.candidate_id == "raw" for item in recommendation.candidates)
    assert len(recommendation.candidates) <= 4
    assert payload["candidates"][0]["evidence_ids"]
    assert all(validate_pipeline(list(item.steps))[0] for item in recommendation.candidates)
    assert all(step.params.get("window", 0) <= 101 for item in recommendation.candidates for step in item.steps)


def test_irregular_axis_excludes_index_smoothing_even_from_rule_fallback():
    from nir_core.preprocess.recommendation import recommend_preprocessing

    axis = np.cumsum(np.linspace(1, 3, 101))
    recommendation = recommend_preprocessing(noisy_calibration(), axis, knowledge_mode="auto")
    assert not recommendation.profile.regular_wavelength_axis
    assert all(step.method not in {"sg_smooth", "whittaker_smooth", "derivative1", "detrend"} for item in recommendation.candidates for step in item.steps)
    assert any(item["reason"] == "irregular_wavelength_axis" for item in recommendation.exclusions)


def test_failed_retrieval_is_explicit_and_preserves_rule_based_control(monkeypatch):
    from nir_core.knowledge.methods import MethodKnowledgeBase
    from nir_core.preprocess.recommendation import recommend_preprocessing

    def unavailable(*args, **kwargs):
        raise OSError("offline index failure")

    monkeypatch.setattr(MethodKnowledgeBase, "search", unavailable)
    recommendation = recommend_preprocessing(noisy_calibration(), knowledge_mode="auto")
    evidence = recommendation.as_dict()["method_knowledge"]
    assert evidence["mode"] == "rule_fallback"
    assert evidence["reason"] == "method_retrieval_unavailable"
    assert recommendation.candidates[0].candidate_id == "raw"


def test_invalid_card_parameters_are_rejected_without_disabling_valid_controls(tmp_path):
    from nir_core.knowledge.methods import DEFAULT_CARDS_PATH, MethodKnowledgeBase
    from nir_core.preprocess.recommendation import recommend_preprocessing

    payload = json.loads(DEFAULT_CARDS_PATH.read_text(encoding="utf-8"))
    for card in payload["cards"]:
        if card["method_id"] == "sg_smooth":
            card["candidate_params"] = [{"window": 10, "order": 2}]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    recommendation = recommend_preprocessing(noisy_calibration(), knowledge_mode="auto", knowledge_base=MethodKnowledgeBase(path))
    assert any(item["reason"] == "invalid_reference_parameters" for item in recommendation.exclusions)
    assert all(step.params.get("window") != 10 for item in recommendation.candidates for step in item.steps)


def test_explicit_only_methods_are_explained_but_not_automatically_executed():
    from nir_core.knowledge.methods import MethodKnowledgeBase

    result = MethodKnowledgeBase().search("emsc scatter")
    emsc = next(item for item in result["results"] if item["method_id"] == "emsc")
    assert emsc["auto_eligible"] is False
    assert emsc["eligibility_reason"] == "explicit_only"


def test_method_search_tool_is_read_only_and_does_not_satisfy_literature_gate():
    from types import SimpleNamespace

    from deerflow.agents.middlewares.nir_workflow_middleware import _TOOL_POLICIES
    from deerflow.community.nir.knowledge import nir_search_method_knowledge_tool

    policy = _TOOL_POLICIES["nir_search_method_knowledge"]
    assert "planning" in policy.stages
    runtime = SimpleNamespace(state={"nir_workflow": {"stage": "planning"}})
    original = json.loads(json.dumps(runtime.state))
    payload = json.loads(nir_search_method_knowledge_tool.func(runtime=runtime, query="noise"))
    assert payload["status"] == "ok" and payload["results"]
    assert runtime.state == original
    assert "evidence_assessment" not in payload
    assert "knowledge_evidence" not in runtime.state["nir_workflow"]


def test_method_cards_are_packaged_with_installed_nir_core():
    root = Path(__file__).resolve().parents[2]
    assert '"nir_core.knowledge" = ["method_cards/*.json"]' in (root / "nir_core/pyproject.toml").read_text(encoding="utf-8")


def test_html_report_links_method_sources_and_escapes_unsafe_links():
    from deerflow.community.nir.delivery import _inline

    value = _inline("[官方说明](https://chemotools.org/methods/preprocessing.html)")
    assert '<a href="https://chemotools.org/methods/preprocessing.html"' in value
    assert "noopener" in value
    assert '<a href="javascript:' not in _inline("[bad](javascript:alert(1))")


def test_knowledge_is_retrieved_before_any_candidate_fit_and_uses_only_calibration(monkeypatch):
    from nir_core.knowledge.methods import MethodKnowledgeBase
    from nir_core.model import evaluation

    from deerflow.community.nir._preprocessing_plan import select_preprocessing

    order = []
    original_search = MethodKnowledgeBase.search
    original_cv = evaluation.nested_cv_preprocessing

    def search(self, query, **kwargs):
        order.append(("retrieval", kwargs["profile"]["n_samples"]))
        return original_search(self, query, **kwargs)

    def cv(*args, **kwargs):
        order.append(("fit", len(args[0])))
        return original_cv(*args, **kwargs)

    monkeypatch.setattr(MethodKnowledgeBase, "search", search)
    monkeypatch.setattr(evaluation, "nested_cv_preprocessing", cv)
    X = noisy_calibration()
    y = np.linspace(0, 1, len(X))
    _, evidence = select_preprocessing(X[:32], y[:32], X[32:], y[32:], None, max_components=3)
    assert order == [("retrieval", 32), ("fit", 32)]
    assert evidence["recommendation"]["profile"]["n_samples"] == 32
    json.dumps(evidence, allow_nan=False)


def test_changed_final_holdout_cannot_change_method_retrieval_or_selected_pipeline(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from deerflow.community.nir import modeling

    rng = np.random.default_rng(581)
    axis = np.linspace(900, 1700, 31)
    coefficients = rng.uniform(0.5, 1.5, (72, 1))
    peak = np.exp(-(((axis - 1200) / 90) ** 2))
    X = 1 + coefficients * peak + rng.normal(0, 0.08, (72, 31))
    y = coefficients.ravel() * 10 + rng.normal(0, 0.05, 72)
    frame = pd.DataFrame(X, columns=[str(value) for value in axis])
    frame.insert(0, "reference", y)
    frame["partition"] = ["Cal"] * 48 + ["Tuning"] * 12 + ["Test"] * 12
    source = tmp_path / "data.csv"
    output = tmp_path / "outputs"
    output.mkdir()
    frame.to_csv(source, index=False)

    def resolve(_runtime, virtual, *, read_only=True):
        return str(source if virtual.endswith("data.csv") else output / Path(virtual).name)

    monkeypatch.setattr(modeling, "_resolve", resolve)
    runtime = SimpleNamespace(state={}, context={}, config={})
    kwargs = dict(
        runtime=runtime,
        file_path="/mnt/user-data/uploads/data.csv",
        split_col="partition",
        train_label="Cal",
        tuning_label="Tuning",
        test_label="Test",
        y_col=0,
        x_cols="1:32",
        method="pls",
        max_components=3,
        compare_cars=False,
        validation_scope="independent_holdout_not_external",
    )
    first = json.loads(modeling.nir_train_partitioned_model_tool.func(**kwargs))
    assert first["status"] == "ok", first
    first_metrics = json.loads((output / "partitioned_metrics.json").read_text())
    # Change ONLY final holdout spectra/reference, never Cal or Tuning.
    frame.loc[60:, frame.columns[1:32]] += 10
    frame.loc[60:, "reference"] += 100
    frame.to_csv(source, index=False)
    second = json.loads(modeling.nir_train_partitioned_model_tool.func(**kwargs))
    assert second["status"] == "ok", second
    second_metrics = json.loads((output / "partitioned_metrics.json").read_text())
    assert first_metrics["preprocessing_selection"] == second_metrics["preprocessing_selection"]
    assert first_metrics["preprocessing_steps"] == second_metrics["preprocessing_steps"]
    assert first_metrics["test"]["RMSE"] != second_metrics["test"]["RMSE"]
