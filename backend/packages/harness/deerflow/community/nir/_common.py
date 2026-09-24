"""Shared helpers for NIR community tools.

Centralises path-resolution, JSON formatting, and pipeline-step parsing so
the individual tool modules (io_tools / preprocess / modeling / reflect /
knowledge) can stay focused on their domain logic.

Kept deliberately dependency-light: only stdlib + numpy + the DeerFlow
sandbox/runtime imports that every tool already needs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from pathlib import Path, PurePosixPath

import numpy as np

from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

_MODEL_MANIFEST_SUFFIX = ".manifest.json"
_MODEL_OUTPUT_PREFIX = ("/", "mnt", "user-data", "outputs")


# ---------------------------------------------------------------------------
# Pipeline-step parsing
# ---------------------------------------------------------------------------


def _parse_pipeline_step(m: str | dict):
    """Parse a pipeline step from a method-name string or a dict with params.

    Accepts both ``"snv"`` (string) and ``{"method": "sg_smooth", "params":
    {"window": 15}}`` (dict) forms, so the LLM can pass hyper-parameters
    through ``pipeline_steps``.
    """
    from nir_core.models import PreprocessingStep

    if isinstance(m, dict):
        return PreprocessingStep(**m)
    return PreprocessingStep(method=str(m))


# ---------------------------------------------------------------------------
# Path-resolution helpers
# ---------------------------------------------------------------------------


def _resolve(runtime: Runtime, virtual_path: str, *, read_only: bool = True) -> str:
    """Resolve a ``/mnt/user-data/...`` virtual path to a real container path.

    Args:
        runtime: DeerFlow runtime (injected by the tool framework).
        virtual_path: Virtual path as seen by the LLM (e.g.
            ``/mnt/user-data/uploads/data.mat``).
        read_only: If True, validate for read access; if False, for write.

    Returns:
        Resolved host/container path string.

    Raises:
        RuntimeError: If thread data is unavailable or the path is invalid.
    """
    from deerflow.sandbox.exceptions import SandboxRuntimeError
    from deerflow.sandbox.tools import (
        get_thread_data,
        resolve_and_validate_user_data_path,
        validate_local_tool_path,
    )

    thread_data = get_thread_data(runtime)
    if thread_data is None:
        raise SandboxRuntimeError(f"Thread data not available; cannot resolve virtual path {virtual_path!r}.")
    validate_local_tool_path(virtual_path, thread_data, read_only=read_only)
    return resolve_and_validate_user_data_path(virtual_path, thread_data)


def _resolve_writable_dir(runtime: Runtime, virtual_dir: str) -> str:
    """Resolve a virtual directory for writing, creating it if needed.

    Used for ``output_dir``-style parameters. The directory is resolved via
    the user-data workspace root and created on disk.
    """
    real_dir = _resolve(runtime, virtual_dir, read_only=False)
    os.makedirs(real_dir, exist_ok=True)
    return real_dir


# ---------------------------------------------------------------------------
# Safe artifact helpers
# ---------------------------------------------------------------------------


def _load_npz_safely(path: str) -> dict[str, np.ndarray]:
    """Load a numeric/string NPZ archive without enabling pickle.

    Object arrays require Python pickle and are therefore not accepted at the
    user-data boundary. New NIR archives store labels as Unicode arrays.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            raise ValueError("Unsafe legacy NPZ: object arrays require pickle. Re-export the archive with the current NIR loader so labels are stored as Unicode arrays.") from exc
        raise


def _model_manifest_path(real_model_path: str) -> Path:
    return Path(real_model_path + _MODEL_MANIFEST_SUFFIX)


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_file(path: str) -> str:
    """Return a streaming SHA-256 digest for lineage/provenance records."""
    return _artifact_digest(Path(path))


def _write_trusted_model_artifact(artifact, real_model_path: str) -> None:
    """Atomically write a joblib artifact and its integrity manifest."""
    import joblib

    destination = Path(real_model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(artifact, temporary)
    os.replace(temporary, destination)

    digest = _artifact_digest(destination)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "nir_model_artifact",
        "sha256": digest,
        "size_bytes": destination.stat().st_size,
    }
    signing_key = os.environ.get("NIR_ARTIFACT_SIGNING_KEY")
    if signing_key:
        manifest["hmac_sha256"] = hmac.new(signing_key.encode("utf-8"), digest.encode("ascii"), hashlib.sha256).hexdigest()

    manifest_path = _model_manifest_path(real_model_path)
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)


def _write_trusted_transfer_artifact(artifact, real_artifact_path: str) -> None:
    """Atomically persist a calibration-transfer artifact and integrity manifest."""
    import joblib

    destination = Path(real_artifact_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    joblib.dump(artifact, temporary)
    os.replace(temporary, destination)

    digest = _artifact_digest(destination)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "nir_calibration_transfer_artifact",
        "sha256": digest,
        "size_bytes": destination.stat().st_size,
    }
    signing_key = os.environ.get("NIR_ARTIFACT_SIGNING_KEY")
    if signing_key:
        manifest["hmac_sha256"] = hmac.new(
            signing_key.encode("utf-8"),
            digest.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

    manifest_path = _model_manifest_path(real_artifact_path)
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_tmp, manifest_path)


def _bind_model_metrics(
    real_model_path: str,
    real_metrics_path: str,
    *,
    training_data_hash: str,
) -> None:
    """Bind the exact metrics/provenance record to an existing model manifest."""

    manifest_path = _model_manifest_path(real_model_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot bind metrics because the model manifest is missing or invalid.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Cannot bind metrics to an unsupported model manifest.")
    actual_artifact_hash = _artifact_digest(Path(real_model_path))
    if manifest.get("sha256") != actual_artifact_hash:
        raise ValueError("Cannot bind metrics to a model that failed integrity verification.")
    normalized_data_hash = str(training_data_hash).lower()
    if len(normalized_data_hash) != 64 or any(character not in "0123456789abcdef" for character in normalized_data_hash):
        raise ValueError("training_data_hash must be a SHA-256 digest")

    metrics_hash = _artifact_digest(Path(real_metrics_path))
    manifest["metrics_sha256"] = metrics_hash
    manifest["training_data_sha256"] = normalized_data_hash
    signing_key = os.environ.get("NIR_ARTIFACT_SIGNING_KEY")
    if signing_key:
        binding = f"{actual_artifact_hash}:{metrics_hash}:{normalized_data_hash}"
        manifest["provenance_hmac_sha256"] = hmac.new(
            signing_key.encode("utf-8"),
            binding.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()

    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)


def _model_evidence(
    real_model_path: str,
    real_metrics_path: str,
    *,
    training_data_hash: str,
    protocol: str,
    validation_scope: str,
) -> dict[str, object]:
    """Return bounded cryptographic evidence for the just-created model run."""

    return {
        "schema_version": 1,
        "protocol": protocol,
        "validation_scope": validation_scope,
        "model_sha256": _artifact_digest(Path(real_model_path)),
        "metrics_sha256": _artifact_digest(Path(real_metrics_path)),
        "training_data_sha256": str(training_data_hash).lower(),
    }


def _load_trusted_model_artifact(real_model_path: str, virtual_model_path: str):
    """Verify model provenance/integrity before invoking joblib.load.

    The outputs directory is writable only by trusted NIR tool execution in
    the supported workflow. Uploaded/workspace pickle files are rejected.
    Production deployments can additionally require an HMAC by setting
    ``NIR_REQUIRE_SIGNED_ARTIFACTS=1`` and ``NIR_ARTIFACT_SIGNING_KEY``.
    """
    import joblib

    virtual = PurePosixPath(virtual_model_path.replace("\\", "/"))
    if virtual.parts[:4] != _MODEL_OUTPUT_PREFIX:
        raise ValueError("Untrusted model path: prediction accepts only models generated under /mnt/user-data/outputs. Register or retrain the model first.")

    model_path = Path(real_model_path)
    manifest_path = _model_manifest_path(real_model_path)
    if not manifest_path.is_file():
        raise ValueError("Unverified model artifact: integrity manifest is missing. Retrain or re-register this legacy model with the current NIR tools.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid model integrity manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported model integrity manifest.")

    expected_digest = manifest.get("sha256")
    actual_digest = _artifact_digest(model_path)
    if not isinstance(expected_digest, str) or not hmac.compare_digest(expected_digest, actual_digest):
        raise ValueError("Model artifact integrity check failed.")
    if manifest.get("size_bytes") != model_path.stat().st_size:
        raise ValueError("Model artifact size does not match its integrity manifest.")

    signing_key = os.environ.get("NIR_ARTIFACT_SIGNING_KEY")
    require_signed = os.environ.get("NIR_REQUIRE_SIGNED_ARTIFACTS", "").strip().lower() in {"1", "true", "yes"}
    signature = manifest.get("hmac_sha256")
    if require_signed and (not signing_key or not isinstance(signature, str)):
        raise ValueError("A signed NIR model is required, but no valid signing configuration or signature was found.")
    if signature is not None:
        if not signing_key:
            raise ValueError("Model is signed, but NIR_ARTIFACT_SIGNING_KEY is not configured.")
        expected_signature = hmac.new(signing_key.encode("utf-8"), actual_digest.encode("ascii"), hashlib.sha256).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(signature, expected_signature):
            raise ValueError("Model artifact signature verification failed.")

    return joblib.load(model_path)


def _load_trusted_transfer_artifact(
    real_artifact_path: str,
    virtual_artifact_path: str,
):
    """Verify a generated transfer artifact before any joblib deserialization."""
    import joblib

    virtual = PurePosixPath(virtual_artifact_path.replace("\\", "/"))
    if virtual.parts[:4] != _MODEL_OUTPUT_PREFIX:
        raise ValueError("Untrusted transfer path: artifacts must be generated under /mnt/user-data/outputs.")

    artifact_path = Path(real_artifact_path)
    manifest_path = _model_manifest_path(real_artifact_path)
    if not manifest_path.is_file():
        raise ValueError("Unverified calibration-transfer artifact: manifest is missing.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid calibration-transfer integrity manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "nir_calibration_transfer_artifact":
        raise ValueError("Unsupported calibration-transfer integrity manifest.")

    expected_digest = manifest.get("sha256")
    actual_digest = _artifact_digest(artifact_path)
    if not isinstance(expected_digest, str) or not hmac.compare_digest(
        expected_digest,
        actual_digest,
    ):
        raise ValueError("Calibration-transfer artifact integrity check failed.")
    if manifest.get("size_bytes") != artifact_path.stat().st_size:
        raise ValueError("Calibration-transfer artifact size does not match its manifest.")

    signing_key = os.environ.get("NIR_ARTIFACT_SIGNING_KEY")
    require_signed = os.environ.get(
        "NIR_REQUIRE_SIGNED_ARTIFACTS",
        "",
    ).strip().lower() in {"1", "true", "yes"}
    signature = manifest.get("hmac_sha256")
    if require_signed and (not signing_key or not isinstance(signature, str)):
        raise ValueError("A signed calibration-transfer artifact is required.")
    if signature is not None:
        if not signing_key:
            raise ValueError("Calibration-transfer artifact is signed, but the signing key is missing.")
        expected_signature = hmac.new(
            signing_key.encode("utf-8"),
            actual_digest.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature,
            expected_signature,
        ):
            raise ValueError("Calibration-transfer artifact signature verification failed.")

    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("format") != ("nir_calibration_transfer_artifact"):
        raise ValueError("Invalid calibration-transfer artifact payload.")
    return artifact


# ---------------------------------------------------------------------------
# JSON formatting helpers
# ---------------------------------------------------------------------------


def _err(
    msg: str,
    *,
    code: str | None = None,
    details: dict | None = None,
) -> str:
    """Format an error as a JSON string for the LLM."""
    payload: dict = {"status": "error", "error": msg}
    if code is not None:
        payload["code"] = code
    if details is not None:
        payload["details"] = details
    return json.dumps(payload, ensure_ascii=False)


def _model_runtime_preflight_error(method: str | None) -> str | None:
    """Return a structured error before expensive work for unavailable models."""
    requested = (method or "").strip().lower()
    if requested != "cnn":
        return None
    try:
        from nir_core.model.cnn import require_cnn_runtime

        require_cnn_runtime()
    except Exception as exc:  # noqa: BLE001 - native dependency loading may fail
        return _err(
            f"1D-CNN runtime is unavailable in the Gateway: {type(exc).__name__}: {exc}",
            code="nir_model_runtime_unavailable",
            details={
                "requested_method": "cnn",
                "required_dependency": "torch",
                "action_required": "rebuild_gateway_with_deep_runtime",
                "substitution_requires_user_approval": True,
            },
        )
    return None


def _ok(payload: dict) -> str:
    """Format a success payload as a JSON string."""
    return json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
