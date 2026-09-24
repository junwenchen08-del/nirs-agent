"""JSON-backed model registry for versioned calibration artifacts.

The registry stores, per ``model_id``, an ordered list of version
records (newest appended last). Persistence is a single JSON file; the
class assumes single-process access (no file locking).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class RegistryCorruptionError(RuntimeError):
    """Raised when an existing registry cannot be parsed safely."""


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: str) -> threading.RLock:
    resolved = str(Path(path).resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(resolved, threading.RLock())


class ModelRegistry:
    """A simple file-based registry of versioned model artifacts.

    The on-disk layout is::

        {
            "<model_id>": [
                {"version": "...", "method": "...", "metrics": {...}, ...},
                ...
            ]
        }

    Versions are tagged as ``"<model_id>-v<unix_timestamp>"`` to be
    monotonically ordered. The class tolerates a missing file (returns
    empty / None) and lazily creates it on first registration.
    """

    def __init__(self, registry_path: str = "artifacts/registry.json") -> None:
        """Initialize the registry.

        Args:
            registry_path: Path to the JSON registry file. Parent
                directories are created lazily on write.
        """
        self.registry_path = str(registry_path)

    # ----- internal IO helpers -------------------------------------------------

    @contextmanager
    def _locked(self):
        """Acquire in-process and cross-process registry locks."""
        with _thread_lock_for(self.registry_path), self._file_locked():
            yield

    @contextmanager
    def _file_locked(self):
        """Acquire a small cross-process lock beside the registry file."""
        lock_path = Path(self.registry_path + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            if lock_file.tell() == 0:
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:  # pragma: no cover - exercised in Linux CI
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock_file.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - exercised in Linux CI
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, list[dict]]:
        p = Path(self.registry_path)
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryCorruptionError(
                f"Cannot read model registry {p}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RegistryCorruptionError(
                f"Model registry {p} must contain a JSON object"
            )
        return data

    def _read(self) -> dict[str, list[dict]]:
        """Read the registry under a cross-process lock."""
        with self._locked():
            return self._read_unlocked()

    def _write_unlocked(self, data: dict[str, list[dict]]) -> None:
        """Atomically persist the registry while the caller holds the lock."""
        p = Path(self.registry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=p.parent,
                prefix=p.name + ".",
                suffix=".tmp",
                delete=False,
            ) as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_name = handle.name
            os.replace(temporary_name, p)
        finally:
            if temporary_name and os.path.exists(temporary_name):
                os.unlink(temporary_name)

    # ----- public API ----------------------------------------------------------

    def register(
        self,
        model_id: str,
        method: str,
        metrics: dict,
        preprocessing_steps: list[dict],
        data_hash: str,
        model_path: str,
        wavelength_indices: list[int] | None = None,
        artifact_hash: str | None = None,
    ) -> str:
        """Register a new model version and return its version tag.

        Args:
            model_id: Logical model identifier (e.g. ``"corn_pls"``).
            method: Calibration method key (e.g. ``"pls"``).
            metrics: Metric dict (RMSE, R2, RPD, ...).
            preprocessing_steps: List of preprocessing step dicts
                (``{"method": ..., "params": {...}}``).
            data_hash: Hash of the training data the model was fit on.
            model_path: Path to the serialized model artifact.
            wavelength_indices: Optional selected wavelength indices.

        Returns:
            Version string ``"<model_id>-v<timestamp>"``.
        """
        with self._locked():
            data = self._read_unlocked()
            versions = data.get(model_id, [])
            timestamp = int(time.time() * 1000)
            version = f"{model_id}-v{timestamp}"
            existing_versions = {str(item.get("version")) for item in versions}
            while version in existing_versions:
                timestamp += 1
                version = f"{model_id}-v{timestamp}"
            record: dict[str, Any] = {
                "version": version,
                "method": method,
                "metrics": metrics,
                "preprocessing_steps": preprocessing_steps,
                "training_data_hash": data_hash,
                "artifact_sha256": artifact_hash,
                "model_path": model_path,
                "wavelength_indices": wavelength_indices,
                "registered_at": timestamp,
            }
            versions.append(record)
            data[model_id] = versions
            self._write_unlocked(data)
        return version

    def load_latest(self, model_id: str) -> dict | None:
        """Return the most recently registered version record, or None."""
        data = self._read()
        versions = data.get(model_id)
        if not versions:
            return None
        return versions[-1]

    def load_version(self, model_id: str, version: str) -> dict | None:
        """Return a specific version record by tag, or None if absent."""
        data = self._read()
        versions = data.get(model_id)
        if not versions:
            return None
        for rec in versions:
            if rec.get("version") == version:
                return rec
        return None

    def list_versions(self, model_id: str) -> list[dict]:
        """Return all version records for ``model_id`` (oldest first)."""
        data = self._read()
        versions = data.get(model_id)
        return list(versions) if versions else []

    def compare(self, model_ids: list[str]) -> dict:
        """Compare the latest version metrics across multiple models.

        Args:
            model_ids: List of model ids to compare.

        Returns:
            Dict mapping each model id to its latest version record (or
            ``None`` when the model has no versions). Callers can then
            inspect the ``metrics`` fields side by side.
        """
        data = self._read()
        result: dict[str, dict | None] = {}
        for mid in model_ids:
            versions = data.get(mid)
            result[mid] = versions[-1] if versions else None
        return result

    def best(self, model_id: str, metric: str = "RPD") -> dict | None:
        """Return the version with the best (max) value of ``metric``.

        Missing metric values are treated as negative infinity so they
        never win. Returns ``None`` when the model has no versions.

        Args:
            model_id: Model id to search.
            metric: Metric key to maximize (default ``"RPD"``).

        Returns:
            Best version record dict, or None.
        """
        data = self._read()
        versions = data.get(model_id)
        if not versions:
            return None

        def _val(rec: dict) -> float:
            m = rec.get("metrics", {}) or {}
            v = m.get(metric)
            try:
                return float(v) if v is not None else float("-inf")
            except (TypeError, ValueError):
                return float("-inf")

        best_rec = versions[0]
        best_v = _val(best_rec)
        for rec in versions[1:]:
            v = _val(rec)
            if v > best_v:
                best_rec = rec
                best_v = v
        return best_rec
