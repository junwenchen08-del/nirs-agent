"""Persistent state machine for continuous NIR prediction drift alerts."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._prediction_audit import (
    _file_lock,
    _thread_lock_for,
    append_prediction_audit,
)

_RECENT_SCORE_LIMIT = 20


class DriftMonitorStateError(RuntimeError):
    """Raised when persisted drift-monitor state is invalid."""


def _environment_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class DriftAlertPolicy:
    """Threshold and hysteresis policy for batch-level drift observations."""

    alert_threshold: float = 0.50
    consecutive_alert_batches: int = 3
    recovery_threshold: float = 0.10
    consecutive_recovery_batches: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.alert_threshold <= 1.0:
            raise ValueError("alert_threshold must be in [0, 1]")
        if not 0.0 <= self.recovery_threshold <= 1.0:
            raise ValueError("recovery_threshold must be in [0, 1]")
        if self.recovery_threshold >= self.alert_threshold:
            raise ValueError("recovery_threshold must be lower than alert_threshold")
        if not 1 <= self.consecutive_alert_batches <= 100:
            raise ValueError("consecutive_alert_batches must be in [1, 100]")
        if not 1 <= self.consecutive_recovery_batches <= 100:
            raise ValueError("consecutive_recovery_batches must be in [1, 100]")

    @classmethod
    def from_environment(cls) -> DriftAlertPolicy:
        """Load a validated policy from optional NIR environment variables."""
        return cls(
            alert_threshold=_environment_float("NIR_DRIFT_ALERT_THRESHOLD", 0.50),
            consecutive_alert_batches=_environment_int(
                "NIR_DRIFT_ALERT_CONSECUTIVE_BATCHES",
                3,
            ),
            recovery_threshold=_environment_float(
                "NIR_DRIFT_RECOVERY_THRESHOLD",
                0.10,
            ),
            consecutive_recovery_batches=_environment_int(
                "NIR_DRIFT_RECOVERY_CONSECUTIVE_BATCHES",
                2,
            ),
        )


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "models": {}}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DriftMonitorStateError("Prediction drift state cannot be read") from exc
    if not isinstance(state, dict) or state.get("schema_version") != 1 or not isinstance(state.get("models"), dict):
        raise DriftMonitorStateError("Prediction drift state has an unsupported schema")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _severity(score: float) -> str:
    if score >= 0.75:
        return "critical"
    if score >= 0.50:
        return "high"
    return "warning"


def _monitor_state(record: dict[str, Any]) -> str:
    if record["alert_active"]:
        return "recovering" if record["consecutive_clean_batches"] else "alert"
    return "watch" if record["consecutive_drift_batches"] else "normal"


def observe_prediction_drift(
    *,
    state_path: str | Path,
    alerts_path: str | Path,
    model_sha256: str,
    model_path: str,
    data_sha256: str | None,
    drift_score: float,
    n_samples: int,
    flagged_count: int | None,
    attribution: dict[str, Any],
    policy: DriftAlertPolicy | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Record a batch and emit alert/recovery events only on transitions."""
    if len(model_sha256) != 64:
        raise ValueError("model_sha256 must be a 64-character digest")
    score = float(drift_score)
    if not 0.0 <= score <= 1.0:
        raise ValueError("drift_score must be in [0, 1]")
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if flagged_count is not None and not 0 <= flagged_count <= n_samples:
        raise ValueError("flagged_count must be between zero and n_samples")

    active_policy = policy or DriftAlertPolicy.from_environment()
    state_file = Path(state_path)
    alerts_file = Path(alerts_path)
    timestamp = observed_at or datetime.now(UTC).isoformat()

    with _thread_lock_for(state_file), _file_lock(state_file):
        state = _read_state(state_file)
        models = state["models"]
        previous = models.get(model_sha256)
        if previous is not None and not isinstance(previous, dict):
            raise DriftMonitorStateError("Prediction drift model state is invalid")
        record = dict(previous or {})
        record.setdefault("model_path", model_path)
        record.setdefault("total_batches", 0)
        record.setdefault("consecutive_drift_batches", 0)
        record.setdefault("consecutive_clean_batches", 0)
        record.setdefault("alert_active", False)
        record.setdefault("active_alert_id", None)
        record.setdefault("recent_scores", [])

        record["model_path"] = model_path
        record["total_batches"] = int(record["total_batches"]) + 1
        record["last_score"] = score
        record["last_seen_at"] = timestamp
        record["recent_scores"] = [
            *[float(value) for value in record["recent_scores"]],
            score,
        ][-_RECENT_SCORE_LIMIT:]

        if score >= active_policy.alert_threshold:
            record["consecutive_drift_batches"] = int(record["consecutive_drift_batches"]) + 1
            record["consecutive_clean_batches"] = 0
        elif score <= active_policy.recovery_threshold:
            record["consecutive_drift_batches"] = 0
            record["consecutive_clean_batches"] = int(record["consecutive_clean_batches"]) + 1
        else:
            record["consecutive_drift_batches"] = 0
            record["consecutive_clean_batches"] = 0

        transition: str | None = None
        alert_event: dict[str, Any] | None = None
        if not record["alert_active"] and record["consecutive_drift_batches"] >= active_policy.consecutive_alert_batches:
            transition = "alert_started"
            alert_id = str(uuid.uuid4())
            record["alert_active"] = True
            record["active_alert_id"] = alert_id
            record["alert_started_at"] = timestamp
            alert_event = {
                "schema_version": 1,
                "event_id": alert_id,
                "event_type": "drift_alert_started",
                "occurred_at": timestamp,
                "severity": _severity(score),
                "recommended_action": "review_model_and_pause_automatic_decisions",
            }
        elif record["alert_active"] and record["consecutive_clean_batches"] >= active_policy.consecutive_recovery_batches:
            transition = "alert_recovered"
            alert_id = str(uuid.uuid4())
            related_alert_id = record.get("active_alert_id")
            record["alert_active"] = False
            record["active_alert_id"] = None
            record["alert_recovered_at"] = timestamp
            record["consecutive_clean_batches"] = 0
            alert_event = {
                "schema_version": 1,
                "event_id": alert_id,
                "event_type": "drift_alert_recovered",
                "occurred_at": timestamp,
                "severity": "info",
                "related_alert_id": related_alert_id,
                "recommended_action": "resume_only_after_operator_review",
            }

        if alert_event is not None:
            alert_event.update(
                {
                    "model": {"path": model_path, "sha256": model_sha256},
                    "input": {
                        "sha256": data_sha256,
                        "n_samples": n_samples,
                        "flagged_count": flagged_count,
                    },
                    "drift_score": score,
                    "policy": asdict(active_policy),
                    "attribution": attribution,
                }
            )
            append_prediction_audit(alerts_file, alert_event)

        models[model_sha256] = record
        state["updated_at"] = timestamp
        _write_state(state_file, state)

    current_state = _monitor_state(record)
    return {
        "status": "observed",
        "state": current_state,
        "drift_score": score,
        "consecutive_drift_batches": record["consecutive_drift_batches"],
        "consecutive_clean_batches": record["consecutive_clean_batches"],
        "alert_active": record["alert_active"],
        "active_alert_id": record["active_alert_id"],
        "alert_emitted": alert_event is not None,
        "transition": transition,
        "recommended_action": ("review_model_and_pause_automatic_decisions" if current_state in {"alert", "recovering"} else None),
        "policy": asdict(active_policy),
    }
