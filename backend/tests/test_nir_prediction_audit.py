"""Regression contracts for NIR prediction audit persistence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from deerflow.community.nir._common import _write_trusted_model_artifact
from deerflow.community.nir._prediction_audit import (
    PredictionAuditCorruptionError,
    append_prediction_audit,
    verify_prediction_audit,
)
from deerflow.community.nir.io_tools import nir_predict_tool


def test_prediction_audit_is_hash_chained_and_concurrency_safe(tmp_path: Path) -> None:
    audit_path = tmp_path / "prediction-audit.jsonl"

    def append(index: int) -> None:
        append_prediction_audit(audit_path, {"schema_version": 1, "event_id": str(index)})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))

    verification = verify_prediction_audit(audit_path)
    assert verification["valid"] is True
    assert verification["event_count"] == 40


def test_prediction_audit_detects_modified_event(tmp_path: Path) -> None:
    audit_path = tmp_path / "prediction-audit.jsonl"
    append_prediction_audit(audit_path, {"schema_version": 1, "event_id": "original"})
    audit_path.write_text(
        audit_path.read_text(encoding="utf-8").replace("original", "modified"),
        encoding="utf-8",
    )

    with pytest.raises(PredictionAuditCorruptionError, match="hash verification"):
        verify_prediction_audit(audit_path)


def test_nir_predict_audits_success_without_raw_business_data(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    X = rng.normal(size=(12, 5))
    y = X @ np.arange(1.0, 6.0)
    model = LinearRegression().fit(X, y)
    model_file = tmp_path / "outputs" / "model.pkl"
    data_file = tmp_path / "uploads" / "batch.npz"
    audit_file = tmp_path / "outputs" / "prediction-audit.jsonl"
    data_file.parent.mkdir(parents=True)
    np.savez(data_file, X=X[:4])
    _write_trusted_model_artifact(
        {"format": "nir_model_artifact", "version": 3, "model": model},
        str(model_file),
    )
    resolved = {
        "/mnt/user-data/outputs/model.pkl": str(model_file),
        "/mnt/user-data/uploads/batch.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(audit_file),
    }
    runtime = SimpleNamespace(
        context={
            "user_id": "user-7",
            "thread_id": "thread-8",
            "run_id": "run-9",
            "deerflow_trace_id": "trace-10",
        }
    )

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        payload = json.loads(
            nir_predict_tool.func(
                runtime=runtime,
                model_path="/mnt/user-data/outputs/model.pkl",
                data_path="/mnt/user-data/uploads/batch.npz",
                detect_drift=False,
                tool_call_id="call-11",
            )
        )

    assert payload["status"] == "ok"
    assert payload["audit"]["path"] == "/mnt/user-data/outputs/prediction-audit.jsonl"
    event = json.loads(audit_file.read_text(encoding="utf-8"))
    assert event["status"] == "success"
    assert event["attribution"] == {
        "user_id": "user-7",
        "thread_id": "thread-8",
        "run_id": "run-9",
        "trace_id": "trace-10",
        "tool_call_id": "call-11",
    }
    assert event["input"]["n_samples"] == 4
    assert event["input"]["n_wavelengths"] == 5
    assert len(event["input"]["sha256"]) == 64
    assert len(event["model"]["sha256"]) == 64
    assert "predictions" not in event
    assert all(str(value) not in audit_file.read_text(encoding="utf-8") for value in y[:4])
    assert verify_prediction_audit(audit_file)["event_count"] == 1


def test_nir_predict_audits_integrity_failure(tmp_path: Path) -> None:
    model_file = tmp_path / "outputs" / "model.pkl"
    data_file = tmp_path / "uploads" / "batch.npz"
    audit_file = tmp_path / "outputs" / "prediction-audit.jsonl"
    data_file.parent.mkdir(parents=True)
    np.savez(data_file, X=np.ones((2, 3)))
    _write_trusted_model_artifact({"format": "nir_model_artifact"}, str(model_file))
    model_file.write_bytes(model_file.read_bytes() + b"tampered")
    resolved = {
        "/mnt/user-data/outputs/model.pkl": str(model_file),
        "/mnt/user-data/uploads/batch.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(audit_file),
    }

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        payload = json.loads(
            nir_predict_tool.func(
                runtime=SimpleNamespace(context={}),
                model_path="/mnt/user-data/outputs/model.pkl",
                data_path="/mnt/user-data/uploads/batch.npz",
            )
        )

    assert payload["status"] == "error"
    event = json.loads(audit_file.read_text(encoding="utf-8"))
    assert event["status"] == "error"
    assert event["error"]["type"] == "ValueError"
    assert "integrity check failed" in event["error"]["message"]
    assert len(event["model"]["sha256"]) == 64


def test_nir_predict_fails_closed_when_audit_cannot_be_persisted(
    tmp_path: Path,
) -> None:
    X = np.arange(12.0).reshape(4, 3)
    model = LinearRegression().fit(X, np.arange(4.0))
    model_file = tmp_path / "outputs" / "model.pkl"
    data_file = tmp_path / "uploads" / "batch.npz"
    data_file.parent.mkdir(parents=True)
    np.savez(data_file, X=X)
    _write_trusted_model_artifact(
        {"format": "nir_model_artifact", "version": 3, "model": model},
        str(model_file),
    )
    resolved = {
        "/mnt/user-data/outputs/model.pkl": str(model_file),
        "/mnt/user-data/uploads/batch.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(tmp_path / "outputs" / "prediction-audit.jsonl"),
    }

    with (
        patch(
            "deerflow.community.nir.io_tools._resolve",
            side_effect=lambda _runtime, path, *, read_only: resolved[path],
        ),
        patch(
            "deerflow.community.nir.io_tools.append_prediction_audit",
            side_effect=OSError("disk full"),
        ),
    ):
        payload = json.loads(
            nir_predict_tool.func(
                runtime=SimpleNamespace(context={}),
                model_path="/mnt/user-data/outputs/model.pkl",
                data_path="/mnt/user-data/uploads/batch.npz",
                detect_drift=False,
            )
        )

    assert payload["status"] == "error"
    assert "audit persistence failed" in payload["error"]
    assert "prediction_mean" not in payload


def test_nir_predict_reports_monitor_failure_without_hiding_prediction(
    tmp_path: Path,
) -> None:
    from nir_core.utils.drift import fit_monitoring_reference

    rng = np.random.RandomState(17)
    X_train = rng.normal(size=(30, 4))
    X_predict = rng.normal(size=(5, 4))
    model = LinearRegression().fit(X_train, X_train[:, 0])
    model_file = tmp_path / "outputs" / "model.pkl"
    data_file = tmp_path / "uploads" / "batch.npz"
    data_file.parent.mkdir(parents=True)
    np.savez(data_file, X=X_predict)
    _write_trusted_model_artifact(
        {
            "format": "nir_model_artifact",
            "version": 3,
            "model": model,
            "monitoring_reference": fit_monitoring_reference(X_train),
        },
        str(model_file),
    )
    audit_file = tmp_path / "outputs" / "prediction-audit.jsonl"
    resolved = {
        "/mnt/user-data/outputs/model.pkl": str(model_file),
        "/mnt/user-data/uploads/batch.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(audit_file),
    }

    with (
        patch(
            "deerflow.community.nir.io_tools._resolve",
            side_effect=lambda _runtime, path, *, read_only: resolved[path],
        ),
        patch(
            "deerflow.community.nir.io_tools.observe_prediction_drift",
            side_effect=OSError("state storage unavailable"),
        ),
    ):
        payload = json.loads(
            nir_predict_tool.func(
                runtime=SimpleNamespace(context={}),
                model_path="/mnt/user-data/outputs/model.pkl",
                data_path="/mnt/user-data/uploads/batch.npz",
                detect_drift=True,
            )
        )

    assert payload["status"] == "ok"
    assert "prediction_mean" in payload
    assert payload["drift_monitoring"] == {
        "status": "error",
        "error_type": "OSError",
    }
    audit_event = json.loads(audit_file.read_text(encoding="utf-8"))
    assert audit_event["drift_monitoring"]["error_type"] == "OSError"
