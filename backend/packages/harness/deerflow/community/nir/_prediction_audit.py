"""Tamper-evident persistence for NIR prediction audit events."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PredictionAuditCorruptionError(RuntimeError):
    """Raised when an existing audit chain cannot be extended safely."""


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.RLock:
    resolved = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(resolved, threading.RLock())


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Hold a one-byte cross-process lock beside the JSONL file."""
    lock_path = path.with_name(path.name + ".lock")
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
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - exercised in Linux CI
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _canonical_digest(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _last_event_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    last_line: bytes | None = None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(4096, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = buffer.splitlines()
            if len(lines) > 1 or position == 0:
                last_line = next((line for line in reversed(lines) if line.strip()), None)
                break
    if last_line is None:
        return None
    try:
        previous = json.loads(last_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionAuditCorruptionError(f"Prediction audit log {path} ends with invalid JSON") from exc
    previous_hash = previous.get("event_hash") if isinstance(previous, dict) else None
    if not isinstance(previous_hash, str) or len(previous_hash) != 64:
        raise PredictionAuditCorruptionError(f"Prediction audit log {path} has no valid terminal event hash")
    unsigned = dict(previous)
    unsigned.pop("event_hash", None)
    if _canonical_digest(unsigned) != previous_hash:
        raise PredictionAuditCorruptionError(f"Prediction audit log {path} failed terminal hash verification")
    return previous_hash


def append_prediction_audit(path: str | Path, event: dict[str, Any]) -> dict[str, Any]:
    """Append one fsynced event linked to the previous JSONL record."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock_for(destination), _file_lock(destination):
        persisted = dict(event)
        persisted["previous_event_hash"] = _last_event_hash(destination)
        persisted["event_hash"] = _canonical_digest(persisted)
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(persisted, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    return persisted


def verify_prediction_audit(path: str | Path) -> dict[str, Any]:
    """Verify every event and link in an audit file."""
    source = Path(path)
    previous_hash: str | None = None
    count = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PredictionAuditCorruptionError(f"Prediction audit log {source} has invalid JSON at line {line_number}") from exc
            if not isinstance(event, dict):
                raise PredictionAuditCorruptionError(f"Prediction audit log {source} has a non-object event at line {line_number}")
            event_hash = event.pop("event_hash", None)
            if event.get("previous_event_hash") != previous_hash:
                raise PredictionAuditCorruptionError(f"Prediction audit log {source} has a broken chain at line {line_number}")
            if not isinstance(event_hash, str) or _canonical_digest(event) != event_hash:
                raise PredictionAuditCorruptionError(f"Prediction audit log {source} failed hash verification at line {line_number}")
            previous_hash = event_hash
            count += 1
    return {"valid": True, "event_count": count, "last_event_hash": previous_hash}
