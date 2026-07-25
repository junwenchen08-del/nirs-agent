"""Process-local cooperative cancellation registry for synchronous tools."""

from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Protocol


class _CancellationSignal(Protocol):
    def is_set(self) -> bool: ...


_lock = RLock()
_signals: dict[str, _CancellationSignal | Callable[[], bool]] = {}


def register_run_cancellation(run_id: str, signal: _CancellationSignal | Callable[[], bool]) -> None:
    """Expose a run's cancellation signal to cooperative synchronous tools."""
    with _lock:
        _signals[run_id] = signal


def unregister_run_cancellation(run_id: str) -> None:
    """Remove a run signal after its worker has finalized."""
    with _lock:
        _signals.pop(run_id, None)


def is_run_cancelled(run_id: str | None) -> bool:
    """Return whether a registered run has requested cancellation."""
    if not run_id:
        return False
    with _lock:
        signal = _signals.get(run_id)
    if signal is None:
        return False
    if callable(signal):
        return bool(signal())
    return bool(signal.is_set())
