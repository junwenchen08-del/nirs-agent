"""A mounted, registered model produces a grounded new-batch outcome."""

import copy
import csv
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from sklearn.linear_model import LinearRegression
from test_nir_workflow_middleware import _request

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir._common import _write_trusted_model_artifact
from deerflow.community.nir._prediction_audit import verify_prediction_audit
from deerflow.community.nir.io_tools import nir_predict_tool
from deerflow.community.nir.prediction_evidence import prediction_result_evidence, prediction_succeeded
from deerflow.community.nir.response_grounding import render_grounded_nir_response, validate_nir_response
from deerflow.community.nir.workflow import start_workflow, transition_workflow, workflow_agent_view


def prediction_workflow():
    wf = start_workflow(task_type="prediction", data_path="/mnt/user-data/uploads/new.mat", model_path="/mnt/user-data/outputs/model.pkl", unit="%")
    wf = transition_workflow(wf, action="record_audit", audit_passed=True)
    wf = transition_workflow(wf, action="plan_ready")
    wf["audit_evidence"] = {"has_reference": False, "n_samples": 60}
    return wf


def outcome():
    return {
        "status": "ok",
        "n_samples": 60,
        "prediction_min": 8.568345,
        "prediction_max": 17.06386,
        "prediction_mean": 12.64626,
        "output_path": "/mnt/user-data/outputs/predictions.csv",
        "preprocessing": {"description": "SNV → 均值中心化", "applied": True},
        "drift": {"available": True, "flagged_indices": list(range(29))},
        "audit": {"event_id": "prediction-event", "event_hash": "a" * 64, "path": "/mnt/user-data/outputs/prediction-audit.jsonl"},
    }


def arguments():
    return {"model_path": "/mnt/user-data/outputs/model.pkl", "data_path": "/mnt/user-data/workspace/new.npz", "output_path": "/mnt/user-data/outputs/predictions.csv"}


def record(wf, payload, *, async_call=False):
    request = _request("nir_predict", {"nir_workflow": wf}, args=arguments(), call_id="predict-1")
    message = ToolMessage(name="nir_predict", tool_call_id="predict-1", content=json.dumps(payload))
    if async_call:

        async def handler(_):
            return message

        return NIRWorkflowMiddleware().awrap_tool_call(request, handler)
    return NIRWorkflowMiddleware().wrap_tool_call(request, lambda _: message)


def test_real_inference_receipt_survives_completion_and_grounding_without_extra_fit_or_predict(tmp_path, monkeypatch):
    X = np.random.default_rng(19).normal(size=(80, 5))
    fitted = LinearRegression().fit(X[:20], X[:20] @ np.arange(1, 6))
    root = tmp_path / "user-data"
    (root / "uploads").mkdir(parents=True)
    np.savez(root / "uploads/new.npz", X=X[20:])
    model_file = root / "outputs/model.pkl"
    _write_trusted_model_artifact({"format": "nir_model_artifact", "version": 3, "model": fitted}, str(model_file))
    before = model_file.read_bytes()
    wf = prediction_workflow()
    args = {**arguments(), "data_path": "/mnt/user-data/uploads/new.npz", "detect_drift": False}
    req = _request("nir_predict", {"nir_workflow": wf}, args=args, call_id="predict-1", context={"user_id": "owner", "thread_id": "thread"})
    monkeypatch.setattr("deerflow.community.nir.io_tools._resolve", lambda runtime, path, **kw: str(root / path.removeprefix("/mnt/user-data/")))
    original_predict = LinearRegression.predict
    calls = []

    def predict(self, values):
        calls.append(values.shape)
        return original_predict(self, values)

    monkeypatch.setattr(LinearRegression, "predict", predict)
    with patch.object(LinearRegression, "fit", side_effect=AssertionError("Inference must never fit")):
        result = NIRWorkflowMiddleware().wrap_tool_call(req, lambda r: ToolMessage(name="nir_predict", tool_call_id="predict-1", content=nir_predict_tool.func(runtime=r.runtime, tool_call_id="predict-1", **r.tool_call["args"])))
        completed = transition_workflow(result.update["nir_workflow"], action="complete")
        assert prediction_succeeded(completed)
        receipt = completed["prediction_evidence"]
        assert workflow_agent_view(completed)["prediction_evidence"] == receipt
        assert receipt["n_samples"] == 60
        output = root / "outputs/predictions.csv"
        assert receipt["audit"]["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        assert verify_prediction_audit(root / "outputs/prediction-audit.jsonl")["event_count"] == 1
        with output.open() as f:
            assert len(list(csv.DictReader(f))) == 60
        state = {"nir_workflow": completed, "messages": [AIMessage(content="R²=0.99，RMSE=0.01，已完成外部验证。", id="final")]}
        guarded = NIRWorkflowMiddleware().after_model(state, SimpleNamespace())
        text = guarded["messages"][0].content
        assert "60 个样本" in text and "下载预测结果 CSV" in text
        assert "没有可展示的建模结果" not in text and "0.99" not in text
        assert validate_nir_response(text, guarded["nir_workflow"]).passed
    assert model_file.read_bytes() == before
    assert calls == [(60, 5)]


def test_prediction_fallback_describes_real_summary_and_drift_without_accuracy_claims():
    wf = record(prediction_workflow(), outcome()).update["nir_workflow"]
    wf = transition_workflow(wf, action="complete")
    before = copy.deepcopy(wf)
    text = render_grounded_nir_response(wf)
    assert "60 个样本" in text and "8.5683～17.0639%" in text and "12.6463%" in text
    assert "29 个样本偏离训练参考分布" in text and "没有参考值" in text
    assert "SNV → 均值中心化" in text and "predictions.csv" in text
    assert validate_nir_response(text, wf).passed
    assert wf == before
    assert "flagged_indices" not in json.dumps(wf["prediction_evidence"])
    assert "CSV" not in render_grounded_nir_response(wf, include_downloads=False)
    assert not validate_nir_response("暂时没有可展示的建模结果。请先上传或选择数据，再开始建模。", wf).passed
    assert not validate_nir_response("[结果](/mnt/user-data/outputs/wrong.csv)", wf).passed


@pytest.mark.parametrize("change", ["audit", "model", "output", "count", "error"])
def test_incomplete_or_failed_receipts_cannot_claim_success(change):
    wf, payload, args = prediction_workflow(), outcome(), arguments()
    if change == "audit":
        payload.pop("audit")
    if change == "model":
        args["model_path"] = "/mnt/user-data/outputs/other.pkl"
    if change == "output":
        payload["output_path"] = "/mnt/user-data/outputs/other.csv"
    if change == "count":
        payload["n_samples"] = 0
    if change == "error":
        payload = {"status": "error", "error": "audit persistence failed"}
    wf["prediction_evidence"] = prediction_result_evidence(wf, args, payload, call_id="predict-1")
    wf = transition_workflow(wf, action="complete")
    assert not prediction_succeeded(wf)
    assert "下载" not in render_grounded_nir_response(wf)
    assert not validate_nir_response("预测已完成。", wf).passed


@pytest.mark.asyncio
async def test_latest_async_failure_replaces_earlier_success():
    wf = record(prediction_workflow(), outcome()).update["nir_workflow"]
    result = await record(wf, {"status": "error", "error": "feature mismatch"}, async_call=True)
    failed = result.update["nir_workflow"]
    assert failed["prediction_evidence"]["status"] == "error"
    assert "本次预测未成功" in render_grounded_nir_response(failed)
    assert "predictions.csv" not in render_grounded_nir_response(failed)


def test_model_mount_alone_and_changed_source_never_prove_prediction():
    wf = prediction_workflow()
    assert not prediction_succeeded(wf)
    assert "尚未确认" in render_grounded_nir_response(wf)
    wf = record(wf, outcome()).update["nir_workflow"]
    wf["data_path"] = "/mnt/user-data/uploads/other.mat"
    assert not prediction_succeeded(wf)


@pytest.mark.parametrize("kind", ["classification", "multi"])
def test_non_scalar_predictions_use_their_own_outcome_shape(kind):
    payload = outcome()
    for key in ("prediction_min", "prediction_max", "prediction_mean"):
        payload.pop(key)
    if kind == "classification":
        payload.update(task_kind="classification", needs_review_count=7, accepted_count=53, mean_confidence=0.8)
    else:
        payload["predictions_summary"] = [{"name": "蛋白", "min": 1, "max": 4, "mean": 2}]
    wf = record(prediction_workflow(), payload).update["nir_workflow"]
    wf = transition_workflow(wf, action="complete")
    text = render_grounded_nir_response(wf)
    assert ("7 个样本需复核" if kind == "classification" else "蛋白：范围 1～4，平均值 2") in text
    assert validate_nir_response(text, wf).passed


def test_old_training_metrics_are_never_new_batch_accuracy():
    wf = record(prediction_workflow(), outcome()).update["nir_workflow"]
    wf["attempt_evidence"] = {"metrics_summary": {"R2": 0.8335, "RMSE": 1.1094}, "validation_scope": "independent_external_validation"}
    verdict = validate_nir_response("R²=0.8335，RMSE=1.1094，已完成外部验证。", wf)
    assert "metric_not_in_current_evidence:r2" in verdict.violations
    assert "validation_scope_overclaim:external" in verdict.violations
