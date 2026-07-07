"""JSON-backed model registry for versioned calibration artifacts.

The registry stores, per ``model_id``, an ordered list of version
records (newest appended last). Persistence is a single JSON file; the
class assumes single-process access (no file locking).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


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

    def _read(self) -> dict[str, list[dict]]:
        """Read the full registry dict from disk, or empty dict if missing."""
        p = Path(self.registry_path)
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    def _write(self, data: dict[str, list[dict]]) -> None:
        """Persist the full registry dict to disk, creating parent dirs."""
        p = Path(self.registry_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

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
        data = self._read()
        versions = data.get(model_id, [])
        timestamp = int(time.time() * 1000)  # ms for uniqueness within a test
        version = f"{model_id}-v{timestamp}"
        record: dict[str, Any] = {
            "version": version,
            "method": method,
            "metrics": metrics,
            "preprocessing_steps": preprocessing_steps,
            "data_hash": data_hash,
            "model_path": model_path,
            "wavelength_indices": wavelength_indices,
            "registered_at": timestamp,
        }
        versions.append(record)
        data[model_id] = versions
        self._write(data)
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
