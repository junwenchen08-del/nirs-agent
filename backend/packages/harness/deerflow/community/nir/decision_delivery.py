"""Phase-4 publication adapter for the existing artifact/report/delivery tools."""

import json
import time
import uuid
from pathlib import Path

import joblib
import numpy as np

from ._common import _bind_model_metrics, _sha256_file, _write_trusted_model_artifact
from ._report import _build_report, report_context
from .artifacts import _build_model_artifact
from .decision_context import InitialDecisionService
from .process import _json, _now


def export_delivery(runtime, search, final, output_root):
    """A failed export retries publication only; finalized Test is never reopened.

    Publication leases are durable. Uncertain building leases require operator
    recovery; caught failures permit a new, unique generation without overwrites.
    """
    final = {key: value for key, value in final.items() if key != "charts"}
    sid = final["session_id"]
    generation = "delivery_" + uuid.uuid4().hex
    root = Path(output_root) / sid
    staging = root / (".staging_" + generation)
    published = root / generation
    virtual = "/mnt/user-data/outputs/" + sid + "/" + generation
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        if session.get("export"):
            for relative, digest in session["export_integrity"].items():
                if _sha256_file(str(root / relative)) != digest:
                    raise ValueError("Published delivery integrity failure; no Test replay permitted")
            return session["export"]
        if session["stage"] != "finalized":
            raise ValueError("Publication requires persisted final Test evidence")
        if session.get("export_state", {}).get("status") == "building":
            raise ValueError("Publication is in-flight/uncertain; do not replay concurrently")
        session["export_state"] = {"status": "building", "generation": generation, "started_at": _now()}
        records = InitialDecisionService(search)._records(db, sid)
        search._save(db, session)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in (root, *root.parents)):
            raise ValueError("Unsafe delivery directory")
        staging.mkdir(exist_ok=False)
        frozen = session["freeze"]
        path = search._dir(sid) / (frozen["experiment_id"] + ".joblib")
        if _sha256_file(str(path)) != frozen["model_hash"]:
            raise ValueError("Frozen artifact integrity failure")
        model = joblib.load(path)
        data_path = search._dir(sid) / "search.npz"
        if _sha256_file(str(data_path)) != session["search_hash"]:
            raise ValueError("Calibration artifact integrity failure")
        with np.load(data_path, allow_pickle=False) as archive:
            X_cal, y_cal = archive["X_cal"], archive["y_cal"]
        successful = [record for record in records if record["status"] == "succeeded"]
        ledger = session["llm_budget"]
        cost = {
            "scope": "workflow intake through finalization tool; excludes subsequent narration/approval",
            "intake_model_calls_included": session.get("intake_model_calls", 0),
            "experiments_started": session["experiments"],
            "experiments_succeeded": len(successful),
            "experiments_failed": sum(record["status"] == "failed" for record in records),
            "completed_parameter_candidates": sum(len(record["execution"].get("trials", [])) for record in successful),
            "failed_candidate_count": "unavailable for interrupted numerical fits",
            "actual_model_fits": session["fits"],
            "llm_calls": sum(ledger[name]["calls"] for name in ("search", "finalization")),
            "actual_input_tokens": sum(ledger[name]["actual_input_tokens"] for name in ("search", "finalization")),
            "actual_output_tokens": sum(ledger[name]["actual_output_tokens"] for name in ("search", "finalization")),
            "calls_with_unknown_usage": sum(ledger[name]["unknown_usage_calls"] for name in ("search", "finalization")),
            "admission_tokens_estimated_or_actual_max": sum(ledger[name]["charged_tokens"] for name in ("search", "finalization")),
            "scientific_execution_seconds": session.get("execution_seconds", 0.0),
            "total_elapsed_seconds": time.time() - session["budget_started_at"],
            "llm_seconds": sum(ledger[name]["seconds"] for name in ("search", "finalization")),
            "monetary_cost": "not supplied by provider",
            "all_caller_provider_usage": session.get("provider_cost_snapshot") or "unavailable; ledger above covers governed Lead calls, not unobserved helpers",
            "token_limit_enabled": session["delivery_policy"]["max_llm_tokens"] is not None,
        }
        lifecycle = {"search_stopped": True, "model_frozen": True, "test_completed": True, "quality_passed": final["metrics"]["quality"]["passed"], "user_approved": False, "registered": False}
        audit = {
            "schema_version": 1,
            "session_id": sid,
            "task": {key: value for key, value in session["task_snapshot"].items() if key != "_workflow_snapshot"},
            "dataset_partition": session.get("library_partition"),
            "retained_experiment_id": frozen["experiment_id"],
            "stop_decision_id": session["stop_decision_id"],
            "stop_source": session["stop_source"],
            "stop_reason": session.get("stop_reason"),
            "retained_selection_source": session["retained_selection_source"],
            "freeze": frozen,
            "cost": cost,
            "lifecycle_at_delivery": lifecycle,
            "decisions": records,
            "budget_policy": session["delivery_policy"],
        }
        metrics = {
            **final["metrics"],
            "protocol": "sealed_single_target_holdout",
            "validation_scope": "independent_holdout_not_external",
            "training_data_hash": session["search_hash"],
            "scientific_validation": session["scientific_validation"],
            "selection_closed": True,
            "reproducibility": {
                "schema_version": 1,
                "input_sha256": session["search_hash"],
                "random_state": 42,
                "cv_manifest": session["cv_manifest"],
                "frozen_model_sha256": frozen["model_hash"],
                "actual_configuration": final["configuration"],
                "actual_parameters": final["parameters"],
            },
            "autonomous_decision_audit": audit,
        }
        artifact = _build_model_artifact(
            model.estimator,
            method=final["configuration"]["method"],
            preprocessing_pipeline=model.preprocessor,
            preprocessing_desc=model.preprocessor.description(),
            wavelength_selection={"method": "none", "params": {}, "n_original": X_cal.shape[1], "n_selected": X_cal.shape[1], "selected_indices": None},
            X_reference=model.preprocessor.transform(X_cal, model.wv),
        )
        artifact["autonomous_binding"] = {key: frozen[key] for key in ("configuration", "parameters", "decision_id", "stop_decision_id", "cv_manifest_id", "model_hash", "search_data_sha256", "fit_scope")}
        _write_trusted_model_artifact(artifact, str(staging / "model.joblib"))
        (staging / "metrics.json").write_text(_json(metrics), encoding="utf-8")
        _bind_model_metrics(str(staging / "model.joblib"), str(staging / "metrics.json"), training_data_hash=session["search_hash"])
        (staging / "decisions.json").write_text(_json(audit), encoding="utf-8")
        from nir_core.models import SpectralData

        from .autonomous_view import AutonomousView
        from .report_figures import build_report_figures

        report_metrics = report_context(runtime, metrics, {"calibration": session["counts"]["calibration"], "tuning": session["counts"]["validation"], "holdout": metrics["test"]["n"]})
        view = AutonomousView(search.process)
        figures = build_report_figures(staging, audit, cal_spectra=view.spectra(sid, frozen["experiment_id"]), final_chart=view.chart(sid, "final_test"), target_unit=report_metrics["report_context"].get("unit"))
        (staging / "report.md").write_text(_build_report(SpectralData(X=X_cal, y=y_cal, wv=model.wv), report_metrics, metrics["quality"], model.preprocessor, report_figures=figures), encoding="utf-8")
        from .delivery import build_delivery

        build_delivery(
            staging / "report.md", model=staging / "model.joblib", metrics=staging / "metrics.json", figures=[staging / figure["filename"] for figure in figures], additional=[("evidence/decisions.json", staging / "decisions.json")]
        )
        payload = {
            **final,
            "metrics": metrics,
            "model": virtual + "/model.joblib",
            "metrics_path": virtual + "/metrics.json",
            "report": virtual + "/report.md",
            "report_html": virtual + "/report.html",
            "bundle": virtual + "/delivery.zip",
            "decisions": virtual + "/decisions.json",
            "lifecycle": lifecycle,
        }
        (staging / "publication.json").write_text(_json(payload), encoding="utf-8")
        staging.rename(published)
        with search.process._connection(write=True) as db:
            current = search._session(db, sid)
            if current["export_state"]["generation"] != generation:
                raise ValueError("Conflicting publication lease")
            current.update(export=payload, export_state={"status": "succeeded", "generation": generation, "completed_at": _now()})
            current["export_integrity"] = {
                generation + "/" + name: _sha256_file(str(published / name))
                for name in ("model.joblib", "model.joblib.manifest.json", "metrics.json", "decisions.json", "report.md", "report.html", "delivery.zip", "publication.json", *(figure["filename"] for figure in figures))
            }
            search._save(db, current)
        return json.loads(_json(payload))
    except BaseException as exc:
        with search.process._connection(write=True) as db:
            current = search._session(db, sid)
            if current.get("export_state", {}).get("generation") == generation:
                current["export_state"].update(status="failed", error_type=type(exc).__name__, failed_at=_now())
                search._save(db, current)
        raise
