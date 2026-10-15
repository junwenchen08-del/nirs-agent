"""Phase-1 governed regression tools, separate from legacy auto entry points."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deerflow.tools.types import Runtime

from ._common import _bind_model_metrics, _load_npz_safely, _resolve, _sha256_file, _write_trusted_model_artifact
from ._report import _build_report, report_context
from ._resources import budget_for_runtime
from .artifacts import _build_model_artifact
from .dataset_session import SearchSessionStore
from .process import _json, runtime_store

logger = logging.getLogger(__name__)


def _service(runtime):
    store = runtime_store(runtime)
    if store is None:
        raise ValueError("A trusted owner/thread runtime is required")
    return SearchSessionStore(store)


def _workflow(runtime, allowed_stages):
    workflow = (runtime.state or {}).get("nir_workflow") or {}
    if workflow.get("search_protocol") != "sealed_holdout" or workflow.get("task_type") not in {"analysis", "calibration"}:
        raise ValueError("Start a sealed_holdout single-target regression workflow first")
    if workflow.get("stage") not in allowed_stages:
        raise ValueError("Invalid sealed regression workflow stage")
    return workflow


def _result(runtime, payload, workflow=None, *, artifacts=None):
    # Chart arrays stay in private finalized session evidence. Agent tools and
    # checkpoints receive only the metrics and published artifact references.
    payload = {key: value for key, value in payload.items() if key != "charts"}
    update = {"messages": [ToolMessage(content=_json(payload), tool_call_id=runtime.tool_call_id)]}
    if workflow is not None:
        update["nir_workflow"] = workflow
    if artifacts is not None:
        # Only successful publication callers provide these trusted references.
        # The thread reducer appends them without replacing earlier artifacts.
        update["artifacts"] = list(artifacts)
    return Command(update=update)


def _failure(runtime, exc):
    # Exception paths can reveal private filenames or dataset values.
    logger.error("Governed NIR operation failed (%s)", type(exc).__name__, exc_info=exc)
    return _result(runtime, {"status": "error", "code": "nir_sealed_search_failed", "error_type": type(exc).__name__, "error": "The governed operation failed; search/test state remains durable. Inspect server diagnostics before retrying."})


def _ingestion_failure(runtime, exc):
    from .datasets.workspace import WorkspaceError

    codes = {
        "Invalid single-target partition shape or numeric integrity": "nir_ingestion_numeric_integrity",
        "Scientific Cal/Val integrity checks failed": "nir_ingestion_scientific_integrity",
        "Scientific partition separation failed": "nir_ingestion_partition_integrity",
        "Only explicit CSV/TXT or numeric NPZ ingestion is supported": "nir_ingestion_format_unsupported",
        "X must contain at least 2 samples and 5 wavelengths.": "nir_ingestion_dimensions_unsupported",
    }
    workflow = (runtime.state or {}).get("nir_workflow") or {}
    workspace_error = isinstance(exc, WorkspaceError)
    code = "nir_ingestion_workspace_invalid" if workspace_error else codes.get(str(exc))
    if not code or not workspace_error and not workflow.get("autonomous_delivery"):
        return _failure(runtime, exc)
    logger.error("Governed NIR ingestion failed (%s)", code, exc_info=exc)
    evidence = {
        "source_tool": "nir_prepare_search",
        "status": "failed",
        "code": code,
        "error_type": type(exc).__name__,
        "retryable": False,
        "next_step": "Check the explicit target mapping, supported format and numeric input integrity with server diagnostics. Do not guess a different target or retry unchanged input.",
    }
    if workspace_error:
        # WorkspaceError contains bounded application messages, never raw paths
        # or values. Preserve the useful reason instead of the generic failure.
        evidence.update(error=str(exc), next_step="请在数据集页面补齐提示的设置，保存后重新挂载并开始建模；不要重复提交同一份未修改的数据。")
    if not workflow.get("autonomous_delivery"):
        return _result(runtime, {"status": "error", **evidence})
    updated = {**workflow, "stage": "blocked", "audit_status": "failed", "revision": workflow["revision"] + 1, "next_action": "resolve_ingestion", "ingestion_evidence": evidence}
    return _result(runtime, {"status": "error", **evidence}, updated)


class _MATMappingRequired(ValueError):
    def __init__(self, mapping, field=None):
        self.mapping = {key: mapping.get(key) for key in ("status", "x_candidates", "y_candidates", "wv_candidates")}
        self.field = field
        super().__init__("MAT requires an unambiguous single-target mapping")


def _load_search_mat(source, *, x_var=None, y_var=None, wv_var=None, transpose=None):
    """Reuse numeric MAT ingestion, never silently choose an ambiguous target."""
    from nir_core.io.loaders import load_mat
    from nir_core.io.sniffers import inspect_file

    info = json.loads(inspect_file(source))
    mapping = info.get("schema_mapping") or {}
    explicit = any(value is not None for value in (x_var, y_var, wv_var, transpose))
    if not explicit:
        if not info.get("labeled_matrix") and (mapping.get("status") != "auto" or not mapping.get("y_variable") or len(mapping.get("compatible_y_candidates", [])) != 1):
            raise _MATMappingRequired(mapping)
        loaded = load_mat(source)
    else:
        if not x_var or not y_var:
            raise _MATMappingRequired(mapping)
        for field, value, candidates in (("x_var", x_var, "x_candidates"), ("y_var", y_var, "y_candidates"), ("wv_var", wv_var, "wv_candidates")):
            if value is not None and value not in mapping.get(candidates, []):
                raise _MATMappingRequired(mapping, field)
        try:
            loaded = load_mat(source, x_var=x_var, y_var=y_var, wv_var=wv_var, transpose=bool(transpose))
        except ValueError as exc:
            raise _MATMappingRequired(mapping, "x_var/y_var/wv_var/transpose") from exc
    if loaded.y is None or np.asarray(loaded.y).ndim > 1 and np.asarray(loaded.y).shape[1] != 1:
        raise _MATMappingRequired(mapping)
    data = {"X": loaded.X, "y": np.asarray(loaded.y).reshape(-1)}
    if loaded.wv is not None:
        data["wv"] = loaded.wv
    return data


def publish_final_delivery(runtime, service, workflow, payload):
    """One publication adapter shared by legacy tools and server orchestration."""
    from .decision_delivery import export_delivery

    base = Path(_resolve(runtime, "/mnt/user-data/outputs", read_only=False))
    payload = export_delivery(runtime, service, payload, base)
    model_file = base / payload["model"].removeprefix("/mnt/user-data/outputs/")
    manifest = json.loads(Path(str(model_file) + ".manifest.json").read_text(encoding="utf-8"))
    evidence = {
        "schema_version": 1,
        "source_tool": "nir_finalize_search",
        "experiment_id": payload["freeze"]["experiment_id"],
        "decision_id": payload["freeze"]["decision_id"],
        "model_path": payload["model"],
        "metrics_path": payload["metrics_path"],
        "protocol": payload["metrics"]["protocol"],
        "validation_scope": payload["metrics"]["validation_scope"],
        "model_sha256": manifest["sha256"],
        "metrics_sha256": manifest["metrics_sha256"],
        "training_data_sha256": manifest["training_data_sha256"],
        "metrics_summary": {key: payload["metrics"].get(key) for key in ("train", "train_cv", "val", "test", "RMSECV", "RMSEP", "RPD", "evaluation_scope", "test_status")},
    }
    evidence["metrics_summary"].update(passed=payload["metrics"]["quality"]["passed"], grade=payload["metrics"]["quality"]["grade"])
    updated = {
        **workflow,
        "stage": "review",
        "revision": workflow["revision"] + 1,
        "next_action": "review_final_result",
        "approval_status": "pending",
        "attempt_evidence": evidence,
        "model_path": payload["model"],
        "metrics_path": payload["metrics_path"],
    }
    return payload, updated


@tool("nir_prepare_search")
def nir_prepare_search_tool(runtime: Runtime, cv_folds: int = 5, y_col: int = 0, x_var: str | None = None, y_var: str | None = None, wv_var: str | None = None, transpose: bool | None = None) -> Command:
    """Bind the workflow source once, seal Test, and return only Cal/Val metadata.

    Supports numeric NPZ (X/y or named X_cal/y_cal, X_val/y_val,
    X_test/y_test), numeric MAT with a unique single-target X/y/wv mapping,
    and an explicitly mapped CSV/TXT target column. Ambiguous MAT files require
    x_var and y_var (dotted paths supported), optional wv_var and transpose.
    Target units come from the user/source documentation, never file names.
    Call only in a fresh sealed_holdout regression workflow.
    """
    if type(cv_folds) is not int or not 2 <= cv_folds <= 10:
        return _result(
            runtime, {"status": "error", "code": "nir_prepare_arguments_invalid", "field": "cv_folds", "allowed_range": [2, 10], "next_step": "Submit an integer in the allowed range; frozen CV must also fit the Cal sample count."}
        )
    try:
        from nir_core.io.loaders import load_csv
        from nir_core.model.evaluation import split_dataset

        workflow = _workflow(runtime, {"data_audit", "planning", "execution"})
        service = _service(runtime)
        if workflow.get("search_session_id"):
            return _result(runtime, {"session_id": workflow["search_session_id"], "test_status": "sealed", "stage": "searching"})
        budget = budget_for_runtime(runtime)
        source = _resolve(runtime, workflow["data_path"], read_only=True)
        budget.check_file(source, stage="sealed_ingestion")
        source_file = {"virtual_path": workflow["data_path"], "sha256": _sha256_file(source), "size_bytes": Path(source).stat().st_size}
        if Path(source).suffix.lower() == ".npz":
            data = _load_npz_safely(source)
        elif Path(source).suffix.lower() == ".mat":
            data = _load_search_mat(source, x_var=x_var, y_var=y_var, wv_var=wv_var, transpose=transpose)
        elif Path(source).suffix.lower() in {".csv", ".txt", ".tsv"}:
            from .datasets.workspace import fixed_regression_partitions, load_workspace_export

            workspace = load_workspace_export(source)
            if workspace is not None:
                data = fixed_regression_partitions(workspace)
            else:
                loaded = load_csv(source, y_col=y_col)
                data = {"X": loaded.X, "y": loaded.y}
                if loaded.wv is not None:
                    data["wv"] = loaded.wv
        else:
            raise ValueError("Only explicit CSV/TXT or numeric NPZ ingestion is supported")
        if "X_cal" not in data:
            if "groups" in data:
                raise ValueError("Grouped datasets require explicit named partitions in phase 1")
            X, y = np.asarray(data["X"]), np.asarray(data["y"])
            budget.check_array(X, target_count=1, stage="sealed_split")
            cal, val, holdout = split_dataset(X, y, test_ratio=0.2, val_ratio=0.2, random_state=42)
            data = {"X_cal": cal[0], "y_cal": cal[1], "X_val": val[0], "y_val": val[1], "X_test": holdout[0], "y_test": holdout[1], **({"wv": data["wv"]} if "wv" in data else {})}
        for name in ("cal", "val", "test"):
            budget.check_array(data["X_" + name], target_count=1, stage="sealed_partition_integrity")
        options = {"autonomous_rounds": True, "max_decisions": workflow.get("max_decisions", 20), "max_fits": workflow.get("max_fits", 200)} if workflow.get("autonomous_rounds") else {}
        if workflow.get("autonomous_delivery"):
            if _sha256_file(source) != source_file["sha256"] or Path(source).stat().st_size != source_file["size_bytes"]:
                raise ValueError("Source changed during ingestion")
            options.update(
                delivery_policy=workflow["delivery_policy"],
                task_snapshot={
                    key: workflow.get(key)
                    for key in ("project_id", "task_type", "analyte", "unit", "domain", "validation_goal", "model_constraint", "pipeline_constraint", "dataset_id", "dataset_profile_id", "dataset_sha256", "workflow_start_call_id")
                },
            )
            options["task_snapshot"]["source_file"] = source_file
            if workflow["delivery_policy"].get("automatic_delivery"):
                options["task_snapshot"]["_workflow_snapshot"] = workflow
        payload = service.prepare(
            workflow["project_id"], data, cv_folds=cv_folds, max_experiments=workflow["max_attempts"], domain=workflow.get("domain") or "default", decision_strategy=workflow.get("decision_strategy", "legacy"), **options
        )
        updated = {
            **workflow,
            "search_session_id": payload["session_id"],
            "stage": "execution",
            "revision": workflow["revision"] + 1,
            "audit_status": "passed",
            "next_action": "search_or_freeze",
            "missing_inputs": [],
            "audit_evidence": {"source_tool": "nir_prepare_search", "passed": True},
        }
        return _result(runtime, payload, updated)
    except _MATMappingRequired as exc:
        return _result(
            runtime,
            {
                "status": "error",
                "code": "nir_mat_mapping_required",
                "field": exc.field,
                "fields": ["x_var", "y_var", "wv_var", "transpose"],
                "mapping": exc.mapping,
                "next_step": "Ask the user which target and spectra to use if uncertain. Resubmit with explicit x_var/y_var, optional wv_var and transpose; no session or Test was evaluated.",
            },
        )
    except Exception as exc:
        return _ingestion_failure(runtime, exc)


@tool("nir_search_model")
def nir_search_model_tool(runtime: Runtime, method: str, pipeline_steps: str = "[]", max_components: int = 10) -> Command:
    """Execute an explicit PLS/Ridge/SVR/ET pipeline against Cal/Val only.

    Phase 1 uses existing bounded numerical grids, with no auto family or
    preprocessing selection. No Test handle or alternate dataset is accepted.
    """
    try:
        workflow = _workflow(runtime, {"execution"})
        if workflow.get("decision_strategy") == "llm_autonomous":
            raise ValueError("Autonomous mode requires nir_submit_decision then nir_execute_decision; direct search is forbidden")
        steps = json.loads(pipeline_steps)
        if not isinstance(steps, list) or len(steps) > 5:
            raise ValueError("Expected a bounded explicit pipeline array")
        budget = budget_for_runtime(runtime)
        payload = _service(runtime).search(workflow["search_session_id"], runtime.tool_call_id, method=method, steps=steps, max_components=max_components, checkpoint=budget.checkpoint)
        return _result(runtime, payload)
    except Exception as exc:
        return _failure(runtime, exc)


@tool("nir_freeze_search")
def nir_freeze_search_tool(runtime: Runtime, experiment_id: str) -> Command:
    """Close search and freeze the explicitly selected successful Cal model."""
    try:
        workflow = _workflow(runtime, {"execution", "awaiting_final_confirmation", "frozen"})
        service = _service(runtime)
        if workflow.get("autonomous_delivery"):
            from .decision_budget import final_checkpoint

            final_checkpoint(service, workflow["search_session_id"], budget_for_runtime(runtime).checkpoint)
            if workflow["delivery_policy"].get("automatic_delivery"):
                raise ValueError("New tasks freeze only through the trusted terminal executor")
        frozen = service.freeze(workflow["search_session_id"], experiment_id)
        updated = {**workflow, "stage": "frozen", "revision": workflow["revision"] + 1, "next_action": "finalize_search"}
        return _result(runtime, {"freeze": frozen, "test_status": "sealed"}, updated)
    except Exception as exc:
        return _failure(runtime, exc)


@tool("nir_finalize_search")
def nir_finalize_search_tool(runtime: Runtime) -> Command:
    """Evaluate the frozen model on Test once and export new verified artifacts.

    Final evaluation cannot trigger another search. Outputs use a unique
    session directory; existing reports and model artifacts are preserved.
    """
    try:
        workflow = _workflow(runtime, {"frozen", "review"})
        service = _service(runtime)
        if workflow.get("autonomous_delivery"):
            from .decision_budget import final_checkpoint

            final_checkpoint(service, workflow["search_session_id"], budget_for_runtime(runtime).checkpoint)
            if workflow["delivery_policy"].get("automatic_delivery"):
                raise ValueError("New tasks finalize only through the trusted terminal executor")
        payload = service.finalize(workflow["search_session_id"], checkpoint=budget_for_runtime(runtime).checkpoint)
        session_id = payload["session_id"]
        virtual_dir = "/mnt/user-data/outputs/" + session_id
        base = Path(_resolve(runtime, "/mnt/user-data/outputs", read_only=False))
        output = base / session_id
        if workflow.get("autonomous_delivery"):
            payload, updated = publish_final_delivery(runtime, service, workflow, payload)
            return _result(runtime, payload, updated, artifacts=[payload[key] for key in ("model", "report_html", "bundle")])
        with service.process._connection(write=True) as db:
            session = service._session(db, session_id)
            # Serialize exports and reuse an already completed export.
            if session.get("export"):
                payload = session["export"]
            else:
                output.mkdir(parents=True, exist_ok=False)
                private = service._dir(session_id)
                selected_path = private / (payload["freeze"]["experiment_id"] + ".joblib")
                if _sha256_file(str(selected_path)) != payload["freeze"]["model_hash"]:
                    raise ValueError("Frozen artifact integrity failure")
                selected = joblib.load(selected_path)
                with np.load(private / "search.npz", allow_pickle=False) as archive:
                    X_cal, y_cal = archive["X_cal"], archive["y_cal"]
                    val_count = len(archive["y_val"])
                from nir_core.models import SpectralData

                metrics = {
                    **payload["metrics"],
                    "protocol": "sealed_single_target_holdout",
                    "validation_scope": "independent_holdout_not_external",
                    "training_data_hash": session["search_hash"],
                    "scientific_validation": {"schema_version": 1, "passed": True, "source": "sealed_session_ingestion"},
                }
                artifact = _build_model_artifact(
                    selected.estimator,
                    method=metrics["method"],
                    preprocessing_pipeline=selected.preprocessor,
                    preprocessing_desc=selected.preprocessor.description(),
                    wavelength_selection={"method": "none", "params": {}, "n_original": X_cal.shape[1], "n_selected": X_cal.shape[1], "selected_indices": None},
                    X_reference=selected.preprocessor.transform(X_cal, selected.wv),
                )
                _write_trusted_model_artifact(artifact, str(output / "model.joblib"))
                (output / "metrics.json").write_text(_json(metrics), encoding="utf-8")
                _bind_model_metrics(str(output / "model.joblib"), str(output / "metrics.json"), training_data_hash=session["search_hash"])
                report_metrics = report_context(runtime, metrics, {"calibration": len(y_cal), "tuning": val_count, "holdout": metrics["test"].get("n")})
                report = _build_report(SpectralData(X=X_cal, y=y_cal, wv=selected.wv), report_metrics, metrics["quality"], selected.preprocessor)
                (output / "report.md").write_text(report, encoding="utf-8")
                payload = {**{key: value for key, value in payload.items() if key != "charts"}, "metrics": metrics, "model": virtual_dir + "/model.joblib", "metrics_path": virtual_dir + "/metrics.json", "report": virtual_dir + "/report.md"}
                session["export"] = payload
                service._save(db, session)
        updated = {**workflow, "stage": "review", "revision": workflow["revision"] + 1, "next_action": "review_final_result", "model_path": payload["model"], "metrics_path": payload["metrics_path"]}
        return _result(runtime, payload, updated)
    except Exception as exc:
        return _failure(runtime, exc)
