"""Versioned NIR model registration tool."""

from __future__ import annotations

import os
import sys
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _ok, _resolve, _sha256_file


def _registration_gate(metrics: dict) -> tuple[bool, str]:
    """Require persisted scientific, quality, lineage, and replay evidence."""

    data_hash = str(metrics.get("training_data_hash", "")).lower()
    if len(data_hash) != 64 or any(character not in "0123456789abcdef" for character in data_hash):
        return False, "training_data_hash is missing or is not a SHA-256 digest"

    science = metrics.get("scientific_validation")
    if not isinstance(science, dict) or science.get("passed") is not True:
        return False, "scientific_validation is missing or did not pass"
    dataset = science.get("dataset")
    separation = science.get("partition_separation")
    if not isinstance(dataset, dict) or dataset.get("passed") is not True:
        return False, "dataset scientific validation did not pass"
    if not isinstance(separation, dict) or separation.get("passed") is not True:
        return False, "partition-separation leakage validation did not pass"

    reproducibility = metrics.get("reproducibility")
    if not isinstance(reproducibility, dict):
        return False, "reproducibility manifest is missing"
    if reproducibility.get("schema_version") != 1:
        return False, "reproducibility manifest schema is unsupported"
    if str(reproducibility.get("input_sha256", "")).lower() != data_hash:
        return False, "reproducibility input hash does not match training_data_hash"
    if not isinstance(reproducibility.get("random_state"), int):
        return False, "reproducibility random_state is missing"

    if isinstance(metrics.get("overall"), dict):
        quality_passed = metrics["overall"].get("passed") is True
    else:
        quality = metrics.get("quality")
        quality_passed = isinstance(quality, dict) and quality.get("passed") is True
    if not quality_passed:
        return False, "model quality gate did not pass"
    return True, "all registration gates passed"


def _validation_goal_gate(metrics: dict, workflow: dict) -> tuple[bool, str]:
    """Require persisted metrics to implement the workflow's declared goal."""

    goal = str(workflow.get("validation_goal") or "").strip().lower()
    required_scope = {
        "internal_holdout": "independent_holdout_not_external",
        "external_validation": "independent_external_validation",
        "production": "independent_external_validation",
    }.get(goal)
    if required_scope is None:
        return False, f"validation_goal {goal!r} does not permit registration"
    actual_scope = str(metrics.get("validation_scope") or "").strip()
    if actual_scope != required_scope:
        return False, f"validation_scope {actual_scope!r} does not satisfy {goal!r}; expected {required_scope!r}"
    protocol = str(metrics.get("protocol") or "").strip()
    if not protocol:
        return False, "validation protocol is missing from metrics"
    evidence = workflow.get("attempt_evidence")
    if not isinstance(evidence, dict):
        return False, "approved attempt evidence is missing"
    if evidence.get("schema_version") != 1:
        return False, "approved attempt evidence schema is unsupported"
    if str(evidence.get("validation_scope") or "") != actual_scope:
        return False, "workflow attempt evidence and metrics have different validation scopes"
    if str(evidence.get("protocol") or "") != protocol:
        return False, "workflow attempt evidence and metrics have different validation protocols"
    return True, "validation goal and protocol match"


def _resolve_registration_path(runtime: Runtime, path: str, *, read_only: bool) -> str:
    """Honor the historical modeling-facade resolver for compatible callers."""
    facade = sys.modules.get("deerflow.community.nir.modeling")
    resolver = getattr(facade, "_resolve", _resolve) if facade is not None else _resolve
    return resolver(runtime, path, read_only=read_only)


@tool("nir_register_model", parse_docstring=True)
def nir_register_model_tool(
    runtime: Runtime,
    model_id: str,
    model_path: str,
    metrics_path: str,
    registry_path: str = "/mnt/user-data/outputs/registry.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Register a trained model into the versioned model registry.

    ★ v3: This tool is designed to be called by the coordinator agent
    *after* the reflection loop finishes, in a serial (non-parallel) manner.
    Sub-agents (nir-modeler) never call this — they only produce .pkl and
    metrics.json files. The coordinator collects all results, picks the best,
    and calls this tool once to register it.

    Args:
        model_id: Logical model identifier (e.g. ``"corn_protein_pls"``).
        model_path: Virtual path to the serialized model (.pkl).
        metrics_path: Virtual path to the metrics JSON file.
        registry_path: Virtual path to the registry JSON file.
        domain: Application domain tag for the registry record.

    Returns:
        JSON with the assigned version tag and registry summary.
    """
    from .workflow import registration_is_approved

    workflow_state = (runtime.state or {}).get("nir_workflow")
    if not registration_is_approved(workflow_state):
        return _err("Model registration requires an approved NIR workflow. Ask the user for explicit approval, then call nir_workflow(action='approve') before registering.")
    evidence = workflow_state["attempt_evidence"]
    if model_path != evidence.get("model_path") or metrics_path != evidence.get("metrics_path"):
        return _err("Model registration rejected: model_path and metrics_path must exactly match the approved modeling attempt.")

    try:
        import json as _json

        try:
            from nir_core.utils.registry import ModelRegistry
        except ImportError:
            return _err("nir_core V3 features (ModelRegistry) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core.")

        # Resolve paths.
        real_model = _resolve_registration_path(runtime, model_path, read_only=True)
        real_metrics = _resolve_registration_path(runtime, metrics_path, read_only=True)
        real_registry = _resolve_registration_path(runtime, registry_path, read_only=False)
        os.makedirs(os.path.dirname(real_registry), exist_ok=True)

        # Load metrics from file.
        with open(real_metrics, encoding="utf-8") as f:
            metrics = _json.load(f)
        approved, gate_reason = _registration_gate(metrics)
        if not approved:
            return _err(f"Model registration rejected by mandatory scientific gate: {gate_reason}.")
        validation_approved, validation_reason = _validation_goal_gate(metrics, workflow_state)
        if not validation_approved:
            return _err(f"Model registration rejected by validation protocol gate: {validation_reason}.")
        manifest_path = real_model + ".manifest.json"
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                model_manifest = _json.load(handle)
        except (OSError, _json.JSONDecodeError) as exc:
            return _err(f"Model registration rejected: the model integrity manifest is missing or invalid ({type(exc).__name__}).")
        artifact_hash = _sha256_file(real_model)
        if str(model_manifest.get("sha256", "")).lower() != artifact_hash:
            return _err("Model registration rejected: the model artifact does not match its integrity manifest.")
        if model_manifest.get("metrics_sha256") != _sha256_file(real_metrics):
            return _err("Model registration rejected: metrics are not the exact provenance record bound to this model artifact.")
        if str(model_manifest.get("training_data_sha256", "")).lower() != str(metrics["training_data_hash"]).lower():
            return _err("Model registration rejected: model and metrics reference different training data.")
        evidence_hashes = {
            "model_sha256": str(model_manifest.get("sha256") or "").lower(),
            "metrics_sha256": str(model_manifest.get("metrics_sha256") or "").lower(),
            "training_data_sha256": str(model_manifest.get("training_data_sha256") or "").lower(),
        }
        for key, actual in evidence_hashes.items():
            expected = str(evidence.get(key) or "").lower()
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                return _err(f"Model registration rejected: approved attempt {key} is missing or invalid.")
            if expected != actual:
                return _err(f"Model registration rejected: approved attempt {key} does not match the current artifact evidence.")

        # Compute a cryptographic artifact fingerprint. Training-data lineage
        # remains a separate field and is populated when metrics provide it.
        # Extract preprocessing steps from metrics if available.
        pp_steps = []
        if isinstance(metrics.get("preprocessing_steps"), list):
            pp_steps = metrics["preprocessing_steps"]
        elif isinstance(metrics.get("preprocessing"), str):
            pp_steps = [{"method": metrics["preprocessing"]}]
        elif isinstance(metrics.get("preprocessing"), list):
            pp_steps = metrics["preprocessing"]

        # Register.
        registry = ModelRegistry(registry_path=real_registry)
        version = registry.register(
            model_id=model_id,
            method=metrics.get("method", "unknown"),
            metrics=metrics,
            preprocessing_steps=pp_steps,
            data_hash=str(metrics.get("training_data_hash", "unknown")),
            model_path=model_path,
            artifact_hash=artifact_hash,
        )

        # Return a summary (no matrix data).
        all_versions = registry.list_versions(model_id)
        return _ok(
            {
                "status": "registered",
                "model_id": model_id,
                "version": version,
                "n_versions": len(all_versions),
                "domain": domain,
                "n_targets": int(metrics.get("n_targets", 1)),
                "component_names": metrics.get("component_names"),
                "registry_path": registry_path,
                "scientific_gate": "passed",
                "reproducibility_protocol": metrics["reproducibility"].get("protocol"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
