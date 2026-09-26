"""Version-pinned, read-only catalog for public Chemotools capabilities.

The MCP server intentionally exposes only objects exported by the public
``__all__`` declarations of an allowlisted set of Chemotools modules.  It never
accepts arbitrary module, class, or attribute names from a caller.
"""

from __future__ import annotations

import importlib
import inspect
import json
import types
from functools import lru_cache
from importlib import metadata
from typing import Any, Literal, Union, get_args, get_origin

CHEMOTOOLS_PINNED_VERSION = "0.4.4"

# Compatibility modules (``models`` and ``cross_decomposition``) intentionally
# do not appear here: their public objects are represented by their canonical
# ``regression`` / ``projection`` modules instead of being duplicated.
PUBLIC_MODULES = (
    "adaptation",
    "adaptation.functions",
    "augmentation",
    "baseline",
    "datasets",
    "derivative",
    "feature_selection",
    "inspector",
    "outliers",
    "physics",
    "plotting",
    "projection",
    "regression",
    "scale",
    "scatter",
    "smooth",
)

EXPLICIT_ONLY_CATEGORIES = frozenset(
    {
        "adaptation",
        "augmentation",
        "feature_selection",
        "inspector",
        "outliers",
        "plotting",
        "projection",
    }
)


class ChemotoolsCatalogError(RuntimeError):
    """Raised when the pinned Chemotools catalog cannot be built safely."""


def chemotools_version() -> str:
    """Return and enforce the provider version used by the MCP contract."""
    try:
        installed = metadata.version("chemotools")
    except metadata.PackageNotFoundError as exc:  # pragma: no cover - environment guard
        raise ChemotoolsCatalogError("Chemotools is not installed. Install nir-core with its pinned dependencies.") from exc
    if installed != CHEMOTOOLS_PINNED_VERSION:
        raise ChemotoolsCatalogError(f"Chemotools version mismatch: expected {CHEMOTOOLS_PINNED_VERSION}, found {installed}.")
    return installed


def _first_paragraph(doc: str | None) -> str:
    if not doc:
        return ""
    lines = [line.strip() for line in inspect.cleandoc(doc).splitlines()]
    paragraph: list[str] = []
    for line in lines:
        if not line and paragraph:
            break
        if line:
            paragraph.append(line)
    return " ".join(paragraph)


def _numpy_parameter_descriptions(doc: str | None) -> dict[str, str]:
    """Extract concise parameter descriptions from a NumPy-style docstring."""
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "Parameters")
    except StopIteration:
        return {}
    if start + 1 >= len(lines) or set(lines[start + 1].strip()) != {"-"}:
        return {}

    descriptions: dict[str, str] = {}
    index = start + 2
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped in {"Returns", "Raises", "Attributes", "Notes", "Examples"}:
            break
        if stripped and not line.startswith((" ", "\t")) and ":" in stripped:
            names = [name.strip() for name in stripped.split(":", 1)[0].split(",")]
            detail: list[str] = []
            index += 1
            while index < len(lines):
                next_line = lines[index]
                next_stripped = next_line.strip()
                if next_stripped and not next_line.startswith((" ", "\t")):
                    break
                if next_stripped:
                    detail.append(next_stripped)
                index += 1
            text = " ".join(detail)
            for name in names:
                descriptions[name] = text
            continue
        index += 1
    return descriptions


def _json_default(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        raise TypeError
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(item) for key, item in value.items()}
    raise TypeError


def _artifact_or_array_reference_schema() -> dict[str, Any]:
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {"$artifact": {"type": "string"}},
                "required": ["$artifact"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "$array": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "key": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    }
                },
                "required": ["$array"],
                "additionalProperties": False,
            },
            {"type": "array"},
        ]
    }


def _annotation_schema(annotation: Any, parameter_name: str) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty:
        if parameter_name in {"model", "estimator", "pipeline"}:
            return _artifact_or_array_reference_schema()
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in {Union, types.UnionType}:
        variants = [_annotation_schema(arg, parameter_name) for arg in args]
        return {"anyOf": variants}
    if origin in {list, tuple, set}:
        item_schema = _annotation_schema(args[0], parameter_name) if args else {}
        return {"type": "array", "items": item_schema}

    if annotation in {str}:
        return {"type": "string"}
    if annotation in {bool}:
        return {"type": "boolean"}
    if annotation in {int}:
        return {"type": "integer"}
    if annotation in {float}:
        return {"type": "number"}
    if annotation in {Any}:
        return {}

    annotation_name = getattr(annotation, "__name__", str(annotation))
    if "ndarray" in annotation_name.lower() or parameter_name in {
        "X",
        "x",
        "y",
        "x_axis",
        "reference",
        "weights",
        "interferences",
        "color_by",
    }:
        return _artifact_or_array_reference_schema()
    if parameter_name in {"model", "estimator", "pipeline"}:
        return _artifact_or_array_reference_schema()
    return {"x-python-type": annotation_name}


def _signature_schema(obj: Any) -> dict[str, Any]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    descriptions = _numpy_parameter_descriptions(inspect.getdoc(obj))
    properties: dict[str, Any] = {}
    required: list[str] = []
    accepts_kwargs = False
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        schema = _annotation_schema(parameter.annotation, parameter.name)
        if description := descriptions.get(parameter.name):
            schema["description"] = description
        if parameter.default is inspect.Parameter.empty:
            required.append(parameter.name)
        else:
            try:
                schema["default"] = _json_default(parameter.default)
            except TypeError:
                schema["x-python-default"] = repr(parameter.default)
        properties[parameter.name] = schema

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": accepts_kwargs,
    }
    if required:
        result["required"] = required
    return result


def _is_subclass(obj: Any, class_or_tuple: Any) -> bool:
    try:
        return inspect.isclass(obj) and issubclass(obj, class_or_tuple)
    except TypeError:
        return False


def _class_kind(obj: type, category: str) -> str:
    if category == "plotting":
        return "plot"
    if category == "inspector":
        return "inspector"

    from sklearn.base import OutlierMixin, RegressorMixin, TransformerMixin
    from sklearn.feature_selection import SelectorMixin

    if _is_subclass(obj, OutlierMixin):
        return "outlier"
    if _is_subclass(obj, SelectorMixin):
        return "selector"
    if _is_subclass(obj, RegressorMixin):
        return "regressor"
    if _is_subclass(obj, TransformerMixin):
        return "transformer"
    return "class"


def _operations(obj: Any, kind: str) -> list[str]:
    if inspect.isfunction(obj):
        return ["call"]
    if kind == "plot" and not inspect.isabstract(obj):
        return ["render"] if callable(getattr(obj, "show", None)) else []
    if kind == "inspector" and not inspect.isabstract(obj):
        return sorted(name for name, member in inspect.getmembers(obj, callable) if name.startswith("inspect") and not name.startswith("_"))
    if inspect.isclass(obj):
        allowed = (
            "fit",
            "transform",
            "fit_transform",
            "predict",
            "fit_predict",
            "decision_function",
            "score",
            "score_samples",
            "get_support",
        )
        return [name for name in allowed if callable(getattr(obj, name, None))]
    return ["read"]


def _safe_constant(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@lru_cache(maxsize=1)
def _catalog_objects() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    version = chemotools_version()
    entries: dict[str, dict[str, Any]] = {}
    objects: dict[str, Any] = {}

    for relative_module in PUBLIC_MODULES:
        module_name = f"chemotools.{relative_module}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional dependency guard
            raise ChemotoolsCatalogError(f"Unable to import public Chemotools module {module_name}: {exc}") from exc

        exports = getattr(module, "__all__", None)
        if exports is None:
            exports = [name for name, value in vars(module).items() if not name.startswith("_") and (inspect.isclass(value) or inspect.isfunction(value))]

        for export_name in exports:
            if not hasattr(module, export_name):
                raise ChemotoolsCatalogError(f"Public export {module_name}.{export_name} is missing.")
            obj = getattr(module, export_name)
            if inspect.ismodule(obj):
                # ``chemotools.adaptation.functions`` is cataloged separately.
                continue
            capability_id = f"{module_name}.{export_name}"
            if inspect.isclass(obj):
                kind = _class_kind(obj, relative_module.split(".", 1)[0])
            elif inspect.isfunction(obj):
                kind = "function"
            else:
                kind = "constant"
            operations = _operations(obj, kind)
            abstract = bool(inspect.isclass(obj) and inspect.isabstract(obj))
            category = relative_module.split(".", 1)[0]
            entry: dict[str, Any] = {
                "id": capability_id,
                "name": export_name,
                "module": module_name,
                "category": category,
                "kind": kind,
                "provider": "chemotools",
                "provider_version": version,
                "description": _first_paragraph(inspect.getdoc(obj)),
                "parameters": _signature_schema(obj) if callable(obj) else {"type": "object", "properties": {}},
                "operations": operations,
                "execution_supported": bool(operations) and not abstract,
                "explicit_only": category in EXPLICIT_ONLY_CATEGORIES,
                "abstract": abstract,
            }
            if kind == "constant":
                entry["value"] = _safe_constant(obj)
            entries[capability_id] = entry
            objects[capability_id] = obj

    return entries, objects


def build_capability_catalog() -> dict[str, Any]:
    """Return a compact, deterministic catalog of public capabilities."""
    entries, _ = _catalog_objects()
    categories: dict[str, int] = {}
    kinds: dict[str, int] = {}
    for entry in entries.values():
        categories[entry["category"]] = categories.get(entry["category"], 0) + 1
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    return {
        "provider": "chemotools",
        "provider_version": chemotools_version(),
        "contract_version": 1,
        "capability_count": len(entries),
        "categories": dict(sorted(categories.items())),
        "kinds": dict(sorted(kinds.items())),
        "capabilities": [entries[key] for key in sorted(entries)],
    }


def describe_capability(capability_id: str) -> dict[str, Any]:
    entries, _ = _catalog_objects()
    try:
        return entries[capability_id]
    except KeyError as exc:
        raise ChemotoolsCatalogError(f"Unknown Chemotools capability {capability_id!r}. Call list_capabilities before choosing a capability.") from exc


def capability_object(capability_id: str) -> Any:
    describe_capability(capability_id)
    return _catalog_objects()[1][capability_id]


def validate_arguments(capability_id: str, arguments: dict[str, Any]) -> None:
    """Validate names and required constructor/function arguments."""
    obj = capability_object(capability_id)
    try:
        inspect.signature(obj).bind(**arguments)
    except TypeError as exc:
        raise ChemotoolsCatalogError(f"Invalid arguments for {capability_id}: {exc}") from exc


def clear_catalog_cache() -> None:
    """Test helper used when a process changes its installed provider."""
    _catalog_objects.cache_clear()


__all__ = [
    "CHEMOTOOLS_PINNED_VERSION",
    "PUBLIC_MODULES",
    "ChemotoolsCatalogError",
    "build_capability_catalog",
    "capability_object",
    "chemotools_version",
    "clear_catalog_cache",
    "describe_capability",
    "validate_arguments",
]
