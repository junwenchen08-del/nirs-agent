"""Versioned NIR model registration tool."""

from __future__ import annotations

import os
import sys
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _ok, _resolve


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

    try:
        import hashlib
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

        # Compute a cryptographic artifact fingerprint. Training-data lineage
        # remains a separate field and is populated when metrics provide it.
        with open(real_model, "rb") as f:
            model_bytes = f.read()
        artifact_hash = hashlib.sha256(model_bytes).hexdigest()

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
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
