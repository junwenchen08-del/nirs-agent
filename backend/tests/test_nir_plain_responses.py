"""Ordinary users receive clear outcomes without weakening result validation."""

import copy
from types import SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.response_grounding import _attempt_evidence_records, render_grounded_nir_response, validate_nir_response


def completed_result():
    search = {
        "experiment_id": "exp-1",
        "method": "svr",
        "protocol": "sealed_single_target_holdout",
        "validation_scope": "independent_holdout_not_external",
        "metrics_summary": {"RMSECV": 1.6731443149150036, "val": {"R2": 0.9909914329483629, "RMSE": 1.5595786623188743}, "test_status": "sealed", "passed": True},
    }
    final = {
        **search,
        "source_tool": "nir_finalize_search",
        "model_path": "/mnt/user-data/outputs/final/model.joblib",
        "metrics_path": "/mnt/user-data/outputs/final/metrics.json",
        "metrics_summary": {**search["metrics_summary"], "test": {"R2": 0.9914944930533893, "RMSE": 1.5036177724350843}, "RMSEP": 1.5036177724350843, "test_status": "consumed"},
    }
    return {
        "stage": "review",
        "next_action": "request_user_approval",
        "task_type": "calibration",
        "approval_status": "pending",
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "autonomous_rounds": True,
        "autonomous_delivery": True,
        "attempts": [search],
        "attempt_evidence": final,
    }


def test_final_result_replaces_earlier_summary_without_duplicate_or_state_changes():
    workflow = completed_result()
    before = copy.deepcopy(workflow)
    records = _attempt_evidence_records(workflow)
    assert len(records) == 1
    assert records[0]["metrics_summary"]["RMSEP"] == 1.5036177724350843
    answer = "内部留出测试 R²=0.9915，RMSEP=1.5036；不是外部验证。[结果数据](/mnt/user-data/outputs/final/metrics.json)"
    assert validate_nir_response(answer, workflow).passed
    assert workflow == before


@pytest.mark.parametrize("advice", ["建议在应用前用一批已知参考值的样品做一次外部验证。", "建议后续补充外部验证。", "We recommend additional external validation before use."])
def test_future_validation_advice_is_not_a_claim_of_completed_validation(advice):
    assert validate_nir_response("内部留出 R²=0.9915；不是外部验证。" + advice, completed_result()).passed


@pytest.mark.parametrize(
    "claim", ["已经通过外部验证。", "建议后续补充外部验证；模型已经通过外部验证。", "建议使用该模型，因为它已经通过外部验证。", "建议使用通过外部验证的该模型。", "We recommend this model: it passed external validation."]
)
def test_positive_external_claim_remains_rejected_even_with_advice(claim):
    assert "validation_scope_overclaim:external" in validate_nir_response(claim, completed_result()).violations


def test_default_result_has_plain_status_final_metrics_and_one_next_step():
    workflow = completed_result()
    text = render_grounded_nir_response(workflow)
    assert "模型训练和测试已完成" in text
    assert "R² = 0.9915" in text and "RMSE = 1.5036" in text
    assert "越接近 1 越好" in text and "越小越好" in text
    assert "同意保存到模型库" in text
    for hidden in ("证据门禁", "质量门禁", "sealed_", "冻结", "RMSECV", "exp-1", "1.5036177724350843", "原回答", "尝试 1"):
        assert hidden not in text
    assert validate_nir_response(text, workflow).passed


@pytest.mark.parametrize("saved", [True, False])
def test_saved_and_failed_library_outcomes_are_not_confused(saved):
    workflow = completed_result()
    workflow.update(stage="completed", approval_status="approved", registered_model={"model_id": "my-model", "version": "v1"}, model_library={"status": "saved" if saved else "failed"})
    text = render_grounded_nir_response(workflow, violations=("metric_value_mismatch:r2",))
    assert "同意保存" not in text and "原回答" not in text
    if saved:
        assert "模型已保存到模型库" in text and "新对话" in text
    else:
        assert "保存到模型库未成功" in text
        assert "模型已保存到模型库" not in text
    assert validate_nir_response(text, workflow).passed


def test_no_results_are_invented_when_ingestion_is_blocked():
    workflow = completed_result()
    workflow.update(stage="blocked", attempts=[], attempt_evidence=None, ingestion_evidence={"code": "nir_ingestion_workspace_invalid", "error": "还有待分配的样本，请先分配用途或标为不参与。"})
    text = render_grounded_nir_response(workflow)
    assert "待分配" in text and "尚未完成" in text
    for hidden in ("Test", "Cal", "R²", "RMSE", "JSON", "封存", "门禁"):
        assert hidden not in text


def test_wrong_joblib_path_and_invented_test_metrics_are_still_rejected():
    text = "内部留出 R²=0.9999，RMSEP=0.01；不是外部验证。[模型文件](/mnt/user-data/outputs/wrong.joblib)"
    verdict = validate_nir_response(text, completed_result())
    assert {"metric_value_mismatch:r2", "metric_value_mismatch:rmsep", "artifact_path_mismatch:model"}.issubset(verdict.violations)


def test_provider_instruction_requests_plain_results_and_keeps_internal_state():
    request = SimpleNamespace(state={"nir_workflow": completed_result()}, system_message=SystemMessage(content="original"))
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    result = NIRWorkflowMiddleware._with_delivery_state(request)
    assert "ordinary Chinese-speaking users" in result.system_message.content
    assert "current_sealed_workflow" in result.system_message.content
    assert "R²" in result.system_message.content
