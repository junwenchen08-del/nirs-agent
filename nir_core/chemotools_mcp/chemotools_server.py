"""Controlled stdio MCP server for the pinned public Chemotools API.

The server exposes a dynamic catalog, but execution is restricted to public
objects discovered from an allowlist in :mod:`nir_core.chemotools_mcp.catalog`. It does
not provide arbitrary Python import, attribute access, or code execution.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from mcp.server.fastmcp import FastMCP

from .catalog import (
    CHEMOTOOLS_PINNED_VERSION,
    ChemotoolsCatalogError,
    build_capability_catalog,
    capability_object,
    chemotools_version,
    describe_capability,
    validate_arguments,
)
from .workspace import (
    ArtifactStore,
    ChemotoolsWorkspaceError,
    file_sha256,
    load_array,
    parse_json_object,
    persist_result,
    resolve_references,
    save_figure_result,
)

SERVER_CONTRACT_VERSION = 1

mcp = FastMCP(
    name="chemotools",
    instructions=(
        "Version-pinned Chemotools execution component for chemometric agents. "
        "Always call list_capabilities and describe_capability before choosing "
        "an algorithm. Use only calibration/training data for fit operations; "
        "never tune against a final holdout. Calibration transfer and data "
        "augmentation are explicit-only capabilities."
    ),
)


def _store() -> ArtifactStore:
    # Stdio MCP processes are scoped to a thread workspace by DeerFlow.  Build
    # the store lazily so tests and callers can safely change cwd before a call.
    return ArtifactStore()


def _numeric_matrix(path: str, key: str) -> tuple[np.ndarray, Path]:
    array, resolved = load_array(path, key=key)
    if array.dtype.kind not in "biufc":
        raise ChemotoolsWorkspaceError(f"Chemotools estimator inputs must be numeric; received {array.dtype}.")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ChemotoolsWorkspaceError(f"Chemotools estimator inputs must be 2D; received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ChemotoolsWorkspaceError("Chemotools estimator inputs must be finite.")
    return np.asarray(array), resolved


def _target_array(path: str | None, key: str) -> tuple[np.ndarray | None, Path | None]:
    if not path:
        return None, None
    target, resolved = load_array(path, key=key)
    if target.ndim == 2 and target.shape[1] == 1:
        target = target.ravel()
    if target.ndim not in {1, 2}:
        raise ChemotoolsWorkspaceError(f"Target input must be 1D or 2D; received shape {target.shape}.")
    if target.dtype.kind in "biufc" and not np.isfinite(target).all():
        raise ChemotoolsWorkspaceError("Target input must be finite.")
    return target, resolved


def _resolved_json(value: str, *, field_name: str, store: ArtifactStore) -> dict[str, Any]:
    return resolve_references(
        parse_json_object(value, field_name=field_name),
        store,
    )


def _fit(estimator: Any, X: np.ndarray, y: np.ndarray | None, kwargs: dict[str, Any]) -> Any:
    if not callable(getattr(estimator, "fit", None)):
        raise ChemotoolsCatalogError(f"{type(estimator).__name__} does not provide a fit operation.")
    return estimator.fit(X, **kwargs) if y is None else estimator.fit(X, y, **kwargs)


def _apply(
    estimator: Any,
    operation: str,
    X: np.ndarray | None,
    y: np.ndarray | None,
    kwargs: dict[str, Any],
) -> Any:
    method = getattr(estimator, operation, None)
    if not callable(method):
        raise ChemotoolsCatalogError(f"{type(estimator).__name__} does not support operation {operation!r}.")
    if operation == "get_support":
        return method(**kwargs)
    if X is None:
        raise ChemotoolsWorkspaceError(f"Operation {operation!r} requires input_path.")
    if operation == "score":
        if y is None:
            raise ChemotoolsWorkspaceError("The score operation requires y_path.")
        return method(X, y, **kwargs)
    return method(X, **kwargs)


def _auto_result_operation(capability: Mapping[str, Any]) -> str:
    operations = set(capability.get("operations") or [])
    for candidate in ("transform", "predict", "decision_function"):
        if candidate in operations:
            return candidate
    return "none"


def _persist_visual_result(result: Any, output_path: str | None) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ChemotoolsWorkspaceError("Chemotools visualization requires matplotlib.") from exc

    if isinstance(result, Figure):
        payload = save_figure_result(result, output_path)
        plt.close(result)
        return payload

    figures: list[Figure] = []
    labels: list[str] = []
    if isinstance(result, Mapping):
        for key, value in result.items():
            if isinstance(value, Figure):
                labels.append(str(key))
                figures.append(value)
    elif isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
        for index, value in enumerate(result):
            if isinstance(value, Figure):
                labels.append(str(index))
                figures.append(value)
    if not figures:
        return persist_result(result, output_path)

    base = Path(output_path) if output_path else None
    outputs: list[dict[str, Any]] = []
    for index, (label, figure) in enumerate(zip(labels, figures, strict=True)):
        if base is None:
            path = None
        else:
            safe_label = "".join(character if character.isalnum() else "-" for character in label).strip("-")
            safe_label = safe_label or str(index)
            path = str(base.with_name(f"{base.stem}-{safe_label}{base.suffix}"))
        outputs.append({"label": label, **save_figure_result(figure, path)})
        plt.close(figure)
    return {"figures": outputs, "figure_count": len(outputs)}


@mcp.tool(
    name="health",
    description="Verify the pinned Chemotools provider and MCP contract version.",
)
def health() -> dict[str, Any]:
    catalog = build_capability_catalog()
    return {
        "status": "ok",
        "server": "chemotools",
        "server_contract_version": SERVER_CONTRACT_VERSION,
        "provider_version": chemotools_version(),
        "capability_count": catalog["capability_count"],
    }


@mcp.tool(
    name="list_capabilities",
    description=("List public Chemotools capabilities from the pinned runtime. Filter by category/kind/query before requesting a detailed parameter schema."),
)
def list_capabilities(
    category: str | None = None,
    kind: str | None = None,
    query: str | None = None,
    executable_only: bool = False,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    catalog = build_capability_catalog()
    capabilities = catalog["capabilities"]
    if category:
        capabilities = [item for item in capabilities if item["category"] == category]
    if kind:
        capabilities = [item for item in capabilities if item["kind"] == kind]
    if query:
        needle = query.casefold()
        capabilities = [item for item in capabilities if needle in " ".join([item["id"], item["name"], item["description"]]).casefold()]
    if executable_only:
        capabilities = [item for item in capabilities if item["execution_supported"]]
    if offset < 0:
        raise ChemotoolsCatalogError("offset must be >= 0.")
    if not 1 <= limit <= 200:
        raise ChemotoolsCatalogError("limit must be between 1 and 200.")
    selected = capabilities[offset : offset + limit]
    return {
        "status": "success",
        "provider": catalog["provider"],
        "provider_version": catalog["provider_version"],
        "contract_version": catalog["contract_version"],
        "total": len(capabilities),
        "offset": offset,
        "limit": limit,
        "capabilities": [
            {
                key: item[key]
                for key in (
                    "id",
                    "name",
                    "category",
                    "kind",
                    "description",
                    "operations",
                    "execution_supported",
                    "explicit_only",
                )
            }
            for item in selected
        ],
    }


@mcp.tool(
    name="describe_capability",
    description=("Return the exact constructor/function parameter schema, supported operations, provider version, and safety classification for one capability."),
)
def describe_capability_tool(capability_id: str) -> dict[str, Any]:
    return {"status": "success", **describe_capability(capability_id)}


@mcp.tool(
    name="validate_operation",
    description=("Validate an operation and JSON arguments against the pinned Chemotools signature without executing the capability."),
)
def validate_operation(
    capability_id: str,
    operation: str,
    arguments_json: str = "{}",
) -> dict[str, Any]:
    capability = describe_capability(capability_id)
    if operation not in capability["operations"]:
        raise ChemotoolsCatalogError(f"Operation {operation!r} is not supported by {capability_id}; allowed operations: {capability['operations']}.")
    arguments = parse_json_object(arguments_json, field_name="arguments_json")
    validate_arguments(capability_id, arguments)
    return {
        "status": "valid",
        "capability_id": capability_id,
        "operation": operation,
        "provider_version": chemotools_version(),
    }


@mcp.tool(
    name="fit_estimator",
    description=("Fit one public Chemotools estimator on training/calibration data only, persist a hash-verified thread-scoped artifact, and optionally emit its training transform/prediction. Never use a final holdout for this call."),
)
def fit_estimator(
    capability_id: str,
    input_path: str,
    parameters_json: str = "{}",
    y_path: str | None = None,
    input_key: str = "X",
    y_key: str = "y",
    fit_parameters_json: str = "{}",
    result_parameters_json: str = "{}",
    result_operation: Literal["auto", "none", "transform", "predict", "decision_function"] = "auto",
    output_path: str | None = None,
) -> dict[str, Any]:
    capability = describe_capability(capability_id)
    if "fit" not in capability["operations"] or capability["kind"] in {
        "plot",
        "inspector",
        "function",
        "constant",
    }:
        raise ChemotoolsCatalogError(f"{capability_id} is not a fit-capable Chemotools estimator.")
    store = _store()
    raw_parameters = parse_json_object(parameters_json, field_name="parameters_json")
    validate_arguments(capability_id, raw_parameters)
    parameters = resolve_references(raw_parameters, store)
    estimator = capability_object(capability_id)(**parameters)
    X, input_resolved = _numeric_matrix(input_path, input_key)
    y, y_resolved = _target_array(y_path, y_key)
    if y is not None and y.shape[0] != X.shape[0]:
        raise ChemotoolsWorkspaceError(f"X and y row counts differ: {X.shape[0]} != {y.shape[0]}.")
    fit_parameters = _resolved_json(
        fit_parameters_json,
        field_name="fit_parameters_json",
        store=store,
    )
    _fit(estimator, X, y, fit_parameters)
    artifact = store.save(
        estimator,
        {
            "artifact_type": "chemotools_estimator",
            "capability_id": capability_id,
            "provider": "chemotools",
            "provider_version": chemotools_version(),
            "parameters": raw_parameters,
            "training_input_sha256": file_sha256(input_resolved),
            "training_target_sha256": file_sha256(y_resolved) if y_resolved else None,
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
        },
    )

    operation = _auto_result_operation(capability) if result_operation == "auto" else result_operation
    payload: dict[str, Any] = {
        "status": "success",
        "capability_id": capability_id,
        "provider_version": chemotools_version(),
        "artifact_id": artifact["artifact_id"],
        "artifact_path": artifact["artifact_path"],
        "artifact_sha256": artifact["object_sha256"],
        "result_operation": operation,
    }
    if operation != "none":
        if operation not in capability["operations"]:
            raise ChemotoolsCatalogError(f"Operation {operation!r} is not supported by {capability_id}.")
        result_parameters = _resolved_json(
            result_parameters_json,
            field_name="result_parameters_json",
            store=store,
        )
        result = _apply(estimator, operation, X, y, result_parameters)
        payload["result"] = persist_result(result, output_path)
    return payload


@mcp.tool(
    name="apply_estimator",
    description=("Apply a hash-verified Chemotools estimator artifact to new data using one of the operations declared by its capability."),
)
def apply_estimator(
    artifact_id: str,
    operation: Literal[
        "transform",
        "predict",
        "decision_function",
        "score",
        "score_samples",
        "get_support",
    ],
    input_path: str | None = None,
    y_path: str | None = None,
    input_key: str = "X",
    y_key: str = "y",
    operation_parameters_json: str = "{}",
    output_path: str | None = None,
) -> dict[str, Any]:
    store = _store()
    estimator, artifact = store.load(artifact_id)
    capability_id = str(artifact.get("capability_id") or "")
    capability = describe_capability(capability_id)
    if operation not in capability["operations"]:
        raise ChemotoolsCatalogError(f"Operation {operation!r} is not supported by {capability_id}.")
    X = None
    input_resolved = None
    if input_path:
        X, input_resolved = _numeric_matrix(input_path, input_key)
    y, _ = _target_array(y_path, y_key)
    if X is not None and y is not None and X.shape[0] != y.shape[0]:
        raise ChemotoolsWorkspaceError(f"X and y row counts differ: {X.shape[0]} != {y.shape[0]}.")
    parameters = _resolved_json(
        operation_parameters_json,
        field_name="operation_parameters_json",
        store=store,
    )
    result = _apply(estimator, operation, X, y, parameters)
    return {
        "status": "success",
        "artifact_id": artifact_id,
        "capability_id": capability_id,
        "provider_version": artifact["provider_version"],
        "operation": operation,
        "input_sha256": file_sha256(input_resolved) if input_resolved else None,
        "result": persist_result(result, output_path),
    }


@mcp.tool(
    name="call_function",
    description=("Call an allowlisted public Chemotools function. Array inputs must use explicit $array references and estimator inputs must use $artifact references."),
)
def call_function(
    capability_id: str,
    arguments_json: str = "{}",
    output_path: str | None = None,
) -> dict[str, Any]:
    capability = describe_capability(capability_id)
    if capability["kind"] != "function" or "call" not in capability["operations"]:
        raise ChemotoolsCatalogError(f"{capability_id} is not a public function.")
    store = _store()
    raw_arguments = parse_json_object(arguments_json, field_name="arguments_json")
    validate_arguments(capability_id, raw_arguments)
    arguments = resolve_references(raw_arguments, store)
    result = capability_object(capability_id)(**arguments)
    return {
        "status": "success",
        "capability_id": capability_id,
        "provider_version": chemotools_version(),
        "result": _persist_visual_result(result, output_path),
    }


@mcp.tool(
    name="render_plot",
    description=("Instantiate and render one public Chemotools plotting class to a PNG inside the current thread workspace."),
)
def render_plot(
    capability_id: str,
    arguments_json: str,
    show_parameters_json: str = "{}",
    output_path: str | None = None,
) -> dict[str, Any]:
    capability = describe_capability(capability_id)
    if capability["kind"] != "plot" or "render" not in capability["operations"]:
        raise ChemotoolsCatalogError(f"{capability_id} is not an executable Chemotools plot.")
    store = _store()
    raw_arguments = parse_json_object(arguments_json, field_name="arguments_json")
    validate_arguments(capability_id, raw_arguments)
    arguments = resolve_references(raw_arguments, store)
    plot = capability_object(capability_id)(**arguments)
    show_parameters = _resolved_json(
        show_parameters_json,
        field_name="show_parameters_json",
        store=store,
    )
    figure = plot.show(**show_parameters)
    return {
        "status": "success",
        "capability_id": capability_id,
        "provider_version": chemotools_version(),
        "result": _persist_visual_result(figure, output_path),
    }


@mcp.tool(
    name="run_inspector",
    description=("Run an allowlisted public inspect* method on a Chemotools Inspector and persist all returned figures inside the current thread workspace."),
)
def run_inspector(
    capability_id: str,
    method: str,
    arguments_json: str,
    method_parameters_json: str = "{}",
    output_path: str | None = None,
) -> dict[str, Any]:
    capability = describe_capability(capability_id)
    if capability["kind"] != "inspector" or method not in capability["operations"]:
        raise ChemotoolsCatalogError(f"Inspector method {method!r} is not allowed for {capability_id}; allowed methods: {capability['operations']}.")
    store = _store()
    raw_arguments = parse_json_object(arguments_json, field_name="arguments_json")
    validate_arguments(capability_id, raw_arguments)
    arguments = resolve_references(raw_arguments, store)
    inspector = capability_object(capability_id)(**arguments)
    method_arguments = _resolved_json(
        method_parameters_json,
        field_name="method_parameters_json",
        store=store,
    )
    result = getattr(inspector, method)(**method_arguments)
    return {
        "status": "success",
        "capability_id": capability_id,
        "provider_version": chemotools_version(),
        "method": method,
        "result": _persist_visual_result(result, output_path),
    }


@mcp.tool(
    name="get_artifact_metadata",
    description=("Verify a thread-scoped Chemotools artifact hash and return its bounded provenance metadata without loading the Python object."),
)
def get_artifact_metadata(artifact_id: str) -> dict[str, Any]:
    return {"status": "success", **_store().metadata(artifact_id)}


@mcp.resource(
    "chemotools://catalog",
    name="chemotools-catalog",
    description="Complete public capability catalog for the pinned Chemotools runtime.",
    mime_type="application/json",
)
def catalog_resource() -> str:
    return json.dumps(build_capability_catalog(), ensure_ascii=False, sort_keys=True)


@mcp.resource(
    "chemotools://capability/{capability_id}",
    name="chemotools-capability",
    description="Detailed schema and metadata for one Chemotools capability.",
    mime_type="application/json",
)
def capability_resource(capability_id: str) -> str:
    return json.dumps(
        describe_capability(capability_id),
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    """Run the thread-scoped stdio MCP server."""
    installed = chemotools_version()
    if installed != CHEMOTOOLS_PINNED_VERSION:  # pragma: no cover - defensive
        raise SystemExit(f"Chemotools {CHEMOTOOLS_PINNED_VERSION} is required; found {installed}.")
    mcp.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover - stdio entry point
    main()


__all__ = [
    "SERVER_CONTRACT_VERSION",
    "apply_estimator",
    "call_function",
    "describe_capability_tool",
    "fit_estimator",
    "get_artifact_metadata",
    "health",
    "list_capabilities",
    "main",
    "mcp",
    "render_plot",
    "run_inspector",
    "validate_operation",
]
