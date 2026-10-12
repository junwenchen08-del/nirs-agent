"""Search feedback must be invariant to final-test evidence."""

import json
from copy import deepcopy

import pytest
from nir_core.utils.metrics import evaluate_quality, search_metrics
from nir_core.utils.validation import suggest_lv_adjustment


def evidence():
    return {
        "n_components": 3,
        "RMSECV": 0.25,
        "val": {"R2": 0.86, "RPD": 3.0, "RMSE": 0.3, "bias": 0.0, "n": 20},
        "test": {"R2": -10.0, "RPD": 0.01, "RMSE": 100.0, "bias": 99.0},
        "RPD": 0.01,
        "RMSEP": 100.0,
        "quality": {"passed": False},
    }


def test_validation_quality_ignores_all_test_aliases():
    first = evidence()
    second = deepcopy(first)
    second.update(RPD=1000.0, RMSEP=0.0)
    second["test"] = {"R2": 1.0, "RPD": 1000.0, "RMSE": 0.0, "bias": 0.0}
    actual = evaluate_quality(first, evaluation_scope="validation")
    assert actual == evaluate_quality(second, evaluation_scope="validation")
    assert actual["passed"] is True
    assert actual["evaluation_scope"] == "validation"


def test_validation_quality_never_falls_back_to_test():
    with pytest.raises(ValueError, match="validation"):
        evaluate_quality({"test": evidence()["test"]}, evaluation_scope="validation")


def test_parameter_advice_uses_validation_error_and_bias():
    first = evidence()
    second = deepcopy(first)
    second["test"]["bias"] = 0.0
    second["RMSEP"] = 0.01
    actual = suggest_lv_adjustment(first, evaluation_scope="validation")
    assert actual == suggest_lv_adjustment(second, evaluation_scope="validation")
    assert "overfitting" not in actual["issues"]
    assert "bias_issue" not in actual["issues"]


def test_search_projection_drops_test_and_cached_quality():
    projected = search_metrics(evidence())
    assert projected["evaluation_scope"] == "validation"
    assert projected["RPD"] == 3.0
    assert projected["RMSE_val"] == 0.3
    assert "test" not in projected and "RMSEP" not in projected
    assert "quality" not in projected


def test_reflection_of_old_report_uses_only_validation(tmp_path, monkeypatch):
    from deerflow.community.nir import reflect

    report = tmp_path / "metrics.json"
    monkeypatch.setattr(reflect, "_resolve", lambda *a, **k: str(report))
    first = evidence()
    report.write_text(json.dumps(first), encoding="utf-8")
    a = json.loads(reflect.nir_reflect_tool.func(metrics_path="report", runtime=None))
    first["test"] = {"R2": 1.0, "RPD": 10000.0, "RMSE": 0.0, "bias": 0.0}
    first.update(RPD=10000.0, RMSEP=0.0)
    report.write_text(json.dumps(first), encoding="utf-8")
    b = json.loads(reflect.nir_reflect_tool.func(metrics_path="report", runtime=None))
    assert a == b
    assert a["current_quality"]["evaluation_scope"] == "validation"


def test_reflection_accepts_comparison_array(tmp_path, monkeypatch):
    from deerflow.community.nir import reflect

    report = tmp_path / "all_metrics.json"
    report.write_text(json.dumps([evidence()]), encoding="utf-8")
    monkeypatch.setattr(reflect, "_resolve", lambda *a, **k: str(report))
    result = json.loads(reflect.nir_reflect_tool.func(metrics_path="report", runtime=None))
    assert result.get("status") != "error"
    assert result["best_so_far"]["RPD"] == 3.0


def test_finalized_report_cannot_trigger_new_search(tmp_path, monkeypatch):
    from deerflow.community.nir import reflect

    report = tmp_path / "metrics.json"
    value = evidence()
    value.update(evaluation_scope="final_test", test_status="consumed")
    report.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(reflect, "_resolve", lambda *a, **k: str(report))
    result = json.loads(reflect.nir_reflect_tool.func(metrics_path="report", runtime=None))
    assert result["should_retry"] is False
    assert result["search_closed"] is True


def test_validation_gate_does_not_replace_final_registration_gate():
    from deerflow.community.nir.registration import _registration_gate

    metrics = evidence()
    metrics.update(
        training_data_hash="a" * 64,
        scientific_validation={"passed": True, "dataset": {"passed": True}, "partition_separation": {"passed": True}},
        reproducibility={"schema_version": 1, "random_state": 42, "input_sha256": "a" * 64},
        quality=evaluate_quality(metrics, evaluation_scope="validation"),
    )
    passed, reason = _registration_gate(metrics)
    assert not passed and "final" in reason
