"""Owner-scoped reports preserve saved evidence without touching an estimator."""

import json
import shutil
from unittest.mock import AsyncMock

import httpx
import pytest
from test_nir_model_service import _registered_package
from test_nir_model_service import model_library as _model_library_fixture
from test_nir_models_router import model_router_client as _model_router_fixture

from deerflow.community.nir.models.report import bounded_chart, build_report, metrics_for_report
from deerflow.community.nir.models.service import ModelError
from deerflow.community.nir.process import ProcessStore

model_library = _model_library_fixture
model_router_client = _model_router_fixture


def prediction(**updates):
    return {"type": "prediction", "scope": "holdout", "x": [1.0, 2.0, 3.0], "y": [1.1, 1.9, 3.2], "n_total": 3, "n_returned": 3, "sample_ids": ["a", "b", "c"], "sampling": "all", **updates}


def test_historical_metrics_never_pair_tuning_r2_with_test_error():
    assert metrics_for_report({"R2_val": 0.9, "RMSEP": 0.3}) == [
        {"scope": "validation", "R2": 0.9, "RMSE": None, "MAE": None, "n": None},
        {"scope": "final_test", "R2": None, "RMSE": 0.3, "MAE": None, "n": None},
    ]
    result = build_report({"test": {"R2": float("nan"), "RMSE": float("inf")}, "preprocessing_steps": []}, {})
    assert result["metrics"] == []
    assert result["preprocessing"] == []
    assert build_report({}, {})["preprocessing"] is None
    json.dumps(result, allow_nan=False)


def test_chart_boundaries_reject_nonfinite_misaligned_and_unknown_scope():
    assert bounded_chart(prediction(scope="unknown")) is None
    assert bounded_chart(prediction(y=[1.0])) is None
    assert bounded_chart(prediction(x=[1.0, float("nan"), 3.0])) is None
    chart = bounded_chart(prediction(x=list(range(2000)), y=list(range(2000)), n_total=3000))
    assert chart["n_returned"] == 500
    assert chart["x"][0] == 0 and chart["x"][-1] == 1999
    assert chart["n_total"] == 3000


@pytest.mark.asyncio
async def test_report_is_owner_scoped_verified_and_has_no_model_or_path(model_library, monkeypatch):
    service, paths = model_library
    version, workflow = _registered_package(paths)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)

    def forbidden(*args, **kwargs):
        raise AssertionError("A report must not load arrays or deserialize models")

    monkeypatch.setattr("numpy.load", forbidden)
    monkeypatch.setattr("deerflow.community.nir._common._load_trusted_model_artifact", forbidden)
    result = await service.get_report("alice", "tablet_assay", version)
    assert result["report"]["metrics"][0]["R2"] == 0.91
    assert result["report"]["preprocessing"] == [{"method": "snv", "params": {}}]
    assert result["report"]["charts"] == []
    assert str(paths.base_dir) not in json.dumps(result)
    with pytest.raises(ModelError, match="Model version not found"):
        await service.get_report("bob", "tablet_assay", version)
    stored = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    (stored / "metrics.json").write_text("{}")
    with pytest.raises(ModelError) as exc:
        await service.get_report("alice", "tablet_assay", version)
    assert exc.value.code == "artifact_unverified"


@pytest.mark.asyncio
async def test_promotion_captures_exact_call_charts_and_survives_source_deletion(model_library):
    service, paths = model_library
    version, workflow = _registered_package(paths)
    workflow["attempt_evidence"]["tool_call_id"] = "trained-call"
    store = ProcessStore(paths.thread_dir("source-thread", user_id="alice") / "nir-process")
    attempt = store.begin(workflow["project_id"], "trained-call", "nir_train_model")
    store.publish(attempt, [{"step_key": "validation", "charts": [prediction()]}])
    store.finish(attempt, "succeeded", workflow)
    other = store.begin(workflow["project_id"], "other-call", "nir_train_model")
    store.publish(other, [{"step_key": "validation", "charts": [prediction(y=[99, 99, 99])]}])
    store.finish(other, "succeeded", workflow)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    shutil.rmtree(store.root)
    report = await service.get_report("alice", "tablet_assay", version)
    assert report["report"]["charts"][0]["y"] == [1.1, 1.9, 3.2]
    attached = await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert attached["status"] == "attached"


def test_missing_call_binding_never_uses_another_attempt(tmp_path):
    store = ProcessStore(tmp_path / "process")
    attempt = store.begin("project", "call", "nir_train_model")
    store.publish(attempt, [{"step_key": "validation", "charts": [prediction()]}])
    store.finish(attempt, "succeeded", {})
    assert build_report({}, {"workflow_project_id": "project"}, store.root)["charts"] == []


def test_finalized_autonomous_report_reuses_real_cached_points(tmp_path, monkeypatch):
    from test_nir_delivery_export import final_runtime, invoke_final

    service, result, _stop, _output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    assert final.get("status") != "error", final
    monkeypatch.setattr("numpy.load", lambda *a, **k: pytest.fail("Report must not load training or test arrays"))
    report = build_report(final["metrics"], {}, service.search.process.root)
    assert report["parameters"] == result["actual_parameters"]
    assert {chart["scope"] for chart in report["charts"]} == {"final_test", "calibration_before", "calibration_after"}
    assert report["method"] == final["metrics"]["reproducibility"]["actual_configuration"]["method"]
    assert report["metrics"][-1]["scope"] == "final_test"


@pytest.mark.asyncio
async def test_report_route_uses_authenticated_owner(model_router_client):
    app, owner, service = model_router_client
    service.get_report = AsyncMock(return_value={"model": {"model_id": "tablet"}, "report": {"schema_version": 1}})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/nir/models/tablet/versions/v1/report")
    assert response.status_code == 200
    service.get_report.assert_awaited_once_with(str(owner.id), "tablet", "v1")
