"""End-to-end regression contract for the NIR model lifecycle."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np


def _write_calibration_csv(path: Path, X: np.ndarray, y: np.ndarray, wv: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["reference", *(f"{value:.1f}" for value in wv)])
        for reference, spectrum in zip(y, X, strict=True):
            writer.writerow([float(reference), *(float(value) for value in spectrum)])


def _approved_workflow(model_path: str, metrics_path: str, evidence: dict) -> dict:
    from deerflow.community.nir.workflow import start_workflow, transition_workflow

    state = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/workspace/calibration.npz",
        analyte="synthetic_reference",
        unit="a.u.",
        domain="default",
        validation_goal="internal_holdout",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=True,
        grade="A",
        model_path=model_path,
        metrics_path=metrics_path,
        attempt_evidence={
            "schema_version": 1,
            "tool_name": "nir_train_model",
            "protocol": "random_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "model_path": model_path,
            "metrics_path": metrics_path,
            "model_sha256": evidence["model_sha256"],
            "metrics_sha256": evidence["metrics_sha256"],
            "training_data_sha256": evidence["training_data_sha256"],
        },
    )
    return transition_workflow(state, action="approve", notes="Approved by the regression fixture")


def test_nir_lifecycle_from_csv_to_registered_prediction_and_tamper_rejection(tmp_path: Path) -> None:
    """Exercise the deployable NIR path without replacing scientific components."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from deerflow.community.nir.io_tools import nir_load_data_tool, nir_predict_tool
    from deerflow.community.nir.modeling import (
        nir_register_model_tool,
        nir_train_model_tool,
    )
    from deerflow.community.nir.models.service import ModelService
    from deerflow.community.nir.workflow import transition_workflow
    from deerflow.config.nir_library_config import NIRLibraryConfig
    from deerflow.config.paths import Paths
    from deerflow.persistence.base import Base
    from deerflow.persistence.thread_meta.model import ThreadMetaRow

    rng = np.random.RandomState(20260721)
    n_samples, n_wavelengths = 96, 12
    latent = rng.normal(size=(n_samples, 3))
    loadings = rng.normal(size=(3, n_wavelengths))
    X = latent @ loadings + rng.normal(scale=0.015, size=(n_samples, n_wavelengths))
    y = 8.0 + 2.2 * latent[:, 0] - 1.4 * latent[:, 1] + 0.7 * latent[:, 2]
    y += rng.normal(scale=0.01, size=n_samples)
    wv = np.linspace(1000.0, 2100.0, n_wavelengths)

    upload_csv = tmp_path / "calibration.csv"
    calibration_npz = tmp_path / "calibration.npz"
    clean_npz = tmp_path / "clean.npz"
    shifted_npz = tmp_path / "shifted.npz"
    library_paths = Paths(tmp_path / "library-home")
    source_outputs = library_paths.sandbox_outputs_dir("source-thread", user_id="alice")
    model_file = source_outputs / "model.pkl"
    metrics_file = source_outputs / "metrics.json"
    registry_file = source_outputs / "registry.json"
    predictions_file = source_outputs / "predictions.csv"
    prediction_audit_file = source_outputs / "prediction-audit.jsonl"
    drift_state_file = source_outputs / "prediction-drift-state.json"
    drift_alerts_file = source_outputs / "prediction-drift-alerts.jsonl"
    _write_calibration_csv(upload_csv, X, y, wv)
    np.savez(clean_npz, X=X[:8], wv=wv)
    spectral_shift = np.linspace(-20.0, 20.0, n_wavelengths)
    np.savez(shifted_npz, X=X[:8] + spectral_shift, wv=wv)

    virtual = {
        "csv": "/mnt/user-data/uploads/calibration.csv",
        "calibration": "/mnt/user-data/workspace/calibration.npz",
        "clean": "/mnt/user-data/uploads/clean.npz",
        "shifted": "/mnt/user-data/uploads/shifted.npz",
        "model": "/mnt/user-data/outputs/model.pkl",
        "metrics": "/mnt/user-data/outputs/metrics.json",
        "registry": "/mnt/user-data/outputs/registry.json",
        "predictions": "/mnt/user-data/outputs/predictions.csv",
        "prediction_audit": "/mnt/user-data/outputs/prediction-audit.jsonl",
    }
    resolved = {
        virtual["csv"]: str(upload_csv),
        virtual["calibration"]: str(calibration_npz),
        virtual["clean"]: str(clean_npz),
        virtual["shifted"]: str(shifted_npz),
        virtual["model"]: str(model_file),
        virtual["metrics"]: str(metrics_file),
        virtual["registry"]: str(registry_file),
        virtual["predictions"]: str(predictions_file),
        virtual["prediction_audit"]: str(prediction_audit_file),
    }

    def resolve(_runtime, path: str, *, read_only: bool) -> str:  # noqa: ARG001
        return resolved[path]

    with ExitStack() as stack:
        stack.enter_context(patch("deerflow.community.nir.io_tools._resolve", side_effect=resolve))
        stack.enter_context(patch("deerflow.community.nir.modeling._resolve", side_effect=resolve))

        load_payload = json.loads(
            nir_load_data_tool.func(
                runtime=MagicMock(),
                file_path=virtual["csv"],
                output_path=virtual["calibration"],
                y_col=0,
                wv_row=0,
                x_cols="1:",
            )
        )
        assert "error" not in load_payload
        assert load_payload["n_samples"] == n_samples
        assert load_payload["n_wavelengths"] == n_wavelengths
        assert load_payload["y_separated"] is True
        assert load_payload["wv_separated"] is True

        train_payload = json.loads(
            nir_train_model_tool.func(
                runtime=MagicMock(),
                input_path=virtual["calibration"],
                method="pls",
                pipeline_steps='["mean_center"]',
                max_components=4,
                cv_folds=3,
                cv_strategy="fixed",
                wavelength_selection="none",
                model_output=virtual["model"],
                metrics_output=virtual["metrics"],
            )
        )
        assert train_payload["status"] == "ok"
        assert train_payload["preprocessing"] == "均值中心化"
        assert train_payload["passed"] is True
        assert model_file.is_file()
        assert Path(str(model_file) + ".manifest.json").is_file()

        approved_state = _approved_workflow(virtual["model"], virtual["metrics"], train_payload["evidence"])
        registration_payload = json.loads(
            nir_register_model_tool.func(
                runtime=SimpleNamespace(state={"nir_workflow": approved_state}),
                model_id="e2e-synthetic-pls",
                model_path=virtual["model"],
                metrics_path=virtual["metrics"],
                registry_path=virtual["registry"],
            )
        )
        assert registration_payload["status"] == "registered"
        assert registration_payload["n_versions"] == 1

        registered_state = transition_workflow(approved_state, action="registered")

        async def promote_and_attach() -> dict:
            engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'e2e-model-library.db').as_posix()}")
            try:
                async with engine.begin() as connection:
                    await connection.run_sync(Base.metadata.create_all)
                service = ModelService(
                    async_sessionmaker(engine, expire_on_commit=False),
                    library_paths,
                    NIRLibraryConfig(enabled=True, min_free_disk_bytes=0),
                )
                async with service.session_factory() as session:
                    session.add_all(
                        [
                            ThreadMetaRow(thread_id="source-thread", user_id="alice", status="idle", metadata_json={}),
                            ThreadMetaRow(thread_id="prediction-thread", user_id="alice", status="idle", metadata_json={}),
                        ]
                    )
                    await session.commit()
                promoted = await service.promote_registered(
                    "alice",
                    "source-thread",
                    "e2e-synthetic-pls",
                    registration_payload["version"],
                    workflow=registered_state,
                    source_run_id="e2e-run",
                )
                attached = await service.attach_to_thread(
                    "alice",
                    "e2e-synthetic-pls",
                    registration_payload["version"],
                    "prediction-thread",
                )
                assert attached["artifact_sha256"] == promoted["artifact_sha256"]
                return attached
            finally:
                await engine.dispose()

        attached = asyncio.run(promote_and_attach())
        attached_model_file = library_paths.resolve_virtual_path(
            "prediction-thread",
            attached["model_path"],
            user_id="alice",
        )
        resolved[attached["model_path"]] = str(attached_model_file)

        registry = json.loads(registry_file.read_text(encoding="utf-8"))
        record = registry["e2e-synthetic-pls"][0]
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        assert record["training_data_hash"] == metrics["training_data_hash"]
        assert record["artifact_sha256"] == hashlib.sha256(model_file.read_bytes()).hexdigest()

        clean_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=virtual["model"],
                data_path=virtual["clean"],
                output_path=virtual["predictions"],
                detect_drift=True,
            )
        )
        shifted_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=virtual["model"],
                data_path=virtual["shifted"],
                detect_drift=True,
            )
        )
        second_shifted_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=virtual["model"],
                data_path=virtual["shifted"],
                detect_drift=True,
            )
        )
        alerting_shifted_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=virtual["model"],
                data_path=virtual["shifted"],
                detect_drift=True,
            )
        )
        assert clean_payload["status"] == "ok"
        assert clean_payload["n_samples"] == 8
        assert clean_payload["preprocessing"]["applied"] is True
        assert clean_payload["drift"]["method"] == "pca_t2_q"
        assert predictions_file.read_text(encoding="utf-8").splitlines()[0] == "sample_index,prediction"
        assert shifted_payload["status"] == "ok"
        assert shifted_payload["drift"]["drift_score"] >= 0.9
        assert shifted_payload["drift"]["drift_score"] > clean_payload["drift"]["drift_score"]
        assert shifted_payload["drift_monitoring"]["state"] == "watch"
        assert second_shifted_payload["drift_monitoring"]["state"] == "watch"
        assert alerting_shifted_payload["drift_monitoring"]["state"] == "alert"
        assert alerting_shifted_payload["drift_monitoring"]["transition"] == "alert_started"
        assert alerting_shifted_payload["drift_monitoring"]["alert_emitted"] is True
        drift_state = json.loads(drift_state_file.read_text(encoding="utf-8"))
        assert drift_state["models"][record["artifact_sha256"]]["alert_active"] is True
        drift_alerts = [json.loads(line) for line in drift_alerts_file.read_text(encoding="utf-8").splitlines()]
        assert [event["event_type"] for event in drift_alerts] == ["drift_alert_started"]
        audit_events = [json.loads(line) for line in prediction_audit_file.read_text(encoding="utf-8").splitlines()]
        assert [event["status"] for event in audit_events] == [
            "success",
            "success",
            "success",
            "success",
        ]
        assert audit_events[1]["previous_event_hash"] == audit_events[0]["event_hash"]

        model_file.write_bytes(model_file.read_bytes() + b"tampered")
        tampered_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=virtual["model"],
                data_path=virtual["clean"],
            )
        )
        assert tampered_payload["status"] == "error"
        assert "integrity check failed" in tampered_payload["error"]
        audit_events = [json.loads(line) for line in prediction_audit_file.read_text(encoding="utf-8").splitlines()]
        assert audit_events[-1]["status"] == "error"
        assert audit_events[-1]["error"]["type"] == "ValueError"

        # The persistent copy remains independently usable after every source
        # thread model/registry file is gone (and after the source was damaged).
        for source_file in (model_file, Path(str(model_file) + ".manifest.json"), metrics_file, registry_file):
            source_file.unlink(missing_ok=True)
        target_audit = library_paths.sandbox_outputs_dir("prediction-thread", user_id="alice") / "prediction-audit.jsonl"
        resolved[virtual["prediction_audit"]] = str(target_audit)
        reused_payload = json.loads(
            nir_predict_tool.func(
                runtime=MagicMock(),
                model_path=attached["model_path"],
                data_path=virtual["clean"],
                detect_drift=True,
            )
        )
        assert reused_payload["status"] == "ok"
        assert reused_payload["preprocessing"]["applied"] is True
        assert reused_payload["drift"]["method"] == "pca_t2_q"
        assert json.loads(target_audit.read_text(encoding="utf-8").splitlines()[-1])["status"] == "success"
