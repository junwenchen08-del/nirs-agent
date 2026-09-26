"""Workspace IO and tamper-evident artifacts for the Chemotools MCP server."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ChemotoolsWorkspaceError(ValueError):
    """Raised for unsafe paths, malformed inputs, or invalid artifacts."""


def parse_json_object(value: str, *, field_name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise ChemotoolsWorkspaceError(f"{field_name} must be valid JSON: {exc.msg}.") from exc
    if not isinstance(decoded, dict):
        raise ChemotoolsWorkspaceError(f"{field_name} must decode to a JSON object.")
    return decoded


def _allowed_roots() -> tuple[Path, ...]:
    roots = [Path.cwd().resolve()]
    configured = os.getenv("CHEMOTOOLS_MCP_ALLOWED_ROOTS", "")
    for raw in configured.split(os.pathsep):
        if raw.strip():
            roots.append(Path(raw).expanduser().resolve())
    return tuple(dict.fromkeys(roots))


def resolve_workspace_path(
    path: str | Path,
    *,
    must_exist: bool,
    allowed_suffixes: set[str] | None = None,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    resolved = candidate.resolve()
    if not any(resolved == root or root in resolved.parents for root in _allowed_roots()):
        raise ChemotoolsWorkspaceError(f"Path {str(path)!r} is outside the MCP thread workspace.")
    if must_exist and not resolved.is_file():
        raise ChemotoolsWorkspaceError(f"Input file does not exist: {resolved}")
    if allowed_suffixes is not None and resolved.suffix.lower() not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ChemotoolsWorkspaceError(f"Unsupported file suffix {resolved.suffix!r}; expected one of {allowed}.")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path: str, *, key: str = "X") -> tuple[np.ndarray, Path]:
    resolved = resolve_workspace_path(
        path,
        must_exist=True,
        allowed_suffixes={".npy", ".npz", ".csv", ".txt", ".tsv"},
    )
    suffix = resolved.suffix.lower()
    if suffix == ".npy":
        data = np.load(resolved, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as archive:
            if key not in archive.files:
                raise ChemotoolsWorkspaceError(f"NPZ key {key!r} is missing from {resolved.name}; available keys: {archive.files}.")
            data = archive[key]
    else:
        delimiter = "\t" if suffix == ".tsv" else "," if suffix == ".csv" else None
        try:
            data = np.loadtxt(resolved, delimiter=delimiter, dtype=float)
        except ValueError as exc:
            raise ChemotoolsWorkspaceError("Text inputs must contain only numeric values with no ambiguous header. Use the NIR ingestion tools first for labeled or mixed-type files.") from exc
    array = np.asarray(data)
    if array.ndim == 0:
        raise ChemotoolsWorkspaceError("Array inputs must have at least one dimension.")
    if array.dtype.kind not in "biufcUS":
        raise ChemotoolsWorkspaceError(f"Object arrays are forbidden; received dtype {array.dtype}.")
    return array, resolved


def array_summary(array: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
    }
    if array.dtype.kind in "biufc" and array.size:
        finite = np.isfinite(array)
        result["finite"] = bool(finite.all())
        result["finite_count"] = int(finite.sum())
        if finite.any():
            finite_values = np.asarray(array[finite], dtype=float)
            result.update(
                {
                    "minimum": float(finite_values.min()),
                    "maximum": float(finite_values.max()),
                    "mean": float(finite_values.mean()),
                }
            )
    return result


def _result_path(output_path: str | None, *, suffix: str) -> Path:
    if output_path:
        target = resolve_workspace_path(
            output_path,
            must_exist=False,
            allowed_suffixes={suffix},
        )
    else:
        target = resolve_workspace_path(
            Path("outputs") / "chemotools-mcp" / "results" / f"{uuid.uuid4().hex}{suffix}",
            must_exist=False,
            allowed_suffixes={suffix},
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def save_array_result(array: np.ndarray, output_path: str | None = None) -> dict[str, Any]:
    target = _result_path(output_path, suffix=".npz")
    with tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}-",
        suffix=".npz",
        dir=target.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, data=np.asarray(array))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "output_path": str(target),
        "output_sha256": file_sha256(target),
        "output": array_summary(np.asarray(array)),
    }


def save_figure_result(figure: Any, output_path: str | None = None) -> dict[str, Any]:
    target = _result_path(output_path, suffix=".png")
    figure.savefig(target, dpi=150, bbox_inches="tight")
    return {
        "output_path": str(target),
        "output_sha256": file_sha256(target),
        "media_type": "image/png",
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return repr(value)


def persist_result(result: Any, output_path: str | None = None) -> dict[str, Any]:
    """Persist large scientific results while keeping MCP responses bounded."""
    if isinstance(result, np.ndarray):
        return save_array_result(result, output_path)

    try:
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover - optional dependency guard
        Figure = ()  # type: ignore[assignment]
    if isinstance(result, Figure):
        return save_figure_result(result, output_path)

    if isinstance(result, Mapping):
        arrays = {str(key): np.asarray(value) for key, value in result.items() if isinstance(value, np.ndarray)}
        if arrays:
            target = _result_path(output_path, suffix=".npz")
            np.savez_compressed(target, **arrays)
            return {
                "output_path": str(target),
                "output_sha256": file_sha256(target),
                "arrays": {key: array_summary(value) for key, value in arrays.items()},
                "metadata": _json_safe({key: value for key, value in result.items() if key not in arrays}),
            }
        return {"value": _json_safe(result)}

    if isinstance(result, (list, tuple)) and result and all(isinstance(item, np.ndarray) for item in result):
        target = _result_path(output_path, suffix=".npz")
        arrays = {f"array_{index}": item for index, item in enumerate(result)}
        np.savez_compressed(target, **arrays)
        return {
            "output_path": str(target),
            "output_sha256": file_sha256(target),
            "arrays": {key: array_summary(value) for key, value in arrays.items()},
        }
    return {"value": _json_safe(result)}


class ArtifactStore:
    """Persist only server-created objects and verify them before unpickling."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            root
            if root is not None
            else resolve_workspace_path(
                Path("outputs") / "chemotools-mcp" / "artifacts",
                must_exist=False,
            )
        )

    def _directory(self, artifact_id: str) -> Path:
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise ChemotoolsWorkspaceError("Invalid Chemotools artifact id.")
        return self.root / artifact_id

    def save(self, obj: Any, metadata_payload: Mapping[str, Any]) -> dict[str, Any]:
        artifact_id = uuid.uuid4().hex
        directory = self._directory(artifact_id)
        directory.mkdir(parents=True, exist_ok=False)
        object_path = directory / "object.joblib"
        metadata_path = directory / "metadata.json"
        joblib.dump(obj, object_path, compress=3)
        payload = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "object_sha256": file_sha256(object_path),
            **_json_safe(dict(metadata_payload)),
        }
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {**payload, "artifact_path": str(directory)}

    def metadata(self, artifact_id: str) -> dict[str, Any]:
        directory = self._directory(artifact_id)
        metadata_path = directory / "metadata.json"
        object_path = directory / "object.joblib"
        if not metadata_path.is_file() or not object_path.is_file():
            raise ChemotoolsWorkspaceError(f"Chemotools artifact {artifact_id!r} does not exist in this thread.")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ChemotoolsWorkspaceError("Artifact metadata is unreadable.") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("artifact_id") != artifact_id:
            raise ChemotoolsWorkspaceError("Artifact metadata is invalid.")
        expected = str(payload.get("object_sha256") or "")
        actual = file_sha256(object_path)
        if not expected or actual != expected:
            raise ChemotoolsWorkspaceError("Artifact hash verification failed; the object may have been modified.")
        return payload

    def load(self, artifact_id: str) -> tuple[Any, dict[str, Any]]:
        payload = self.metadata(artifact_id)
        object_path = self._directory(artifact_id) / "object.joblib"
        return joblib.load(object_path), payload


def resolve_references(value: Any, store: ArtifactStore) -> Any:
    """Resolve explicit data and artifact references inside JSON arguments."""
    if isinstance(value, dict):
        if set(value) == {"$artifact"}:
            obj, _ = store.load(str(value["$artifact"]))
            return obj
        if set(value) == {"$array"}:
            spec = value["$array"]
            if not isinstance(spec, dict) or "path" not in spec:
                raise ChemotoolsWorkspaceError("$array must contain an object with path and optional key.")
            array, _ = load_array(str(spec["path"]), key=str(spec.get("key", "X")))
            return array
        return {key: resolve_references(item, store) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_references(item, store) for item in value]
    return value


__all__ = [
    "ArtifactStore",
    "ChemotoolsWorkspaceError",
    "array_summary",
    "file_sha256",
    "load_array",
    "parse_json_object",
    "persist_result",
    "resolve_references",
    "resolve_workspace_path",
    "save_array_result",
    "save_figure_result",
]
