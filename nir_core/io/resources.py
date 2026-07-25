"""Resource budgets for bounded NIR file loading and model execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np


_MIB = 1024**2
_GIB = 1024**3


class ResourceLimitError(RuntimeError):
    """Raised when an input or operation exceeds its configured budget."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description for tool error responses."""
        return {
            "code": self.code,
            "message": str(self),
            "stage": self.stage,
            "details": self.details,
        }


def _read_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _read_positive_float(
    environ: Mapping[str, str], name: str, default: float
) -> float:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero")
    return value


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Configurable ceilings for one NIR load or modeling operation."""

    max_file_bytes: int = 512 * _MIB
    max_matrix_elements: int = 50_000_000
    max_samples: int = 100_000
    max_wavelengths: int = 50_000
    max_targets: int = 256
    max_estimated_peak_bytes: int = 4 * _GIB
    max_runtime_seconds: float = 900.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ResourceLimits:
        """Build limits from ``NIR_MAX_*`` environment variables."""
        env = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            max_file_bytes=_read_positive_int(
                env, "NIR_MAX_FILE_BYTES", defaults.max_file_bytes
            ),
            max_matrix_elements=_read_positive_int(
                env, "NIR_MAX_MATRIX_ELEMENTS", defaults.max_matrix_elements
            ),
            max_samples=_read_positive_int(
                env, "NIR_MAX_SAMPLES", defaults.max_samples
            ),
            max_wavelengths=_read_positive_int(
                env, "NIR_MAX_WAVELENGTHS", defaults.max_wavelengths
            ),
            max_targets=_read_positive_int(
                env, "NIR_MAX_TARGETS", defaults.max_targets
            ),
            max_estimated_peak_bytes=_read_positive_int(
                env,
                "NIR_MAX_ESTIMATED_PEAK_BYTES",
                defaults.max_estimated_peak_bytes,
            ),
            max_runtime_seconds=_read_positive_float(
                env, "NIR_MAX_RUNTIME_SECONDS", defaults.max_runtime_seconds
            ),
        )


class ResourceBudget:
    """Tracks size, memory, deadline, and cooperative cancellation limits."""

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        *,
        timeout_seconds: float | None = None,
        cancel_check: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limits = limits or ResourceLimits.from_env()
        self._clock = clock
        self._cancel_check = cancel_check
        self._started_at = clock()
        timeout = (
            self.limits.max_runtime_seconds
            if timeout_seconds is None
            else min(float(timeout_seconds), self.limits.max_runtime_seconds)
        )
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and greater than zero")
        self._timeout_seconds = timeout
        self._deadline = self._started_at + timeout
        self._last_stage = "created"
        self._max_observed_file_bytes = 0
        self._max_observed_elements = 0
        self._max_estimated_peak_bytes = 0

    def checkpoint(self, stage: str) -> None:
        """Fail at a safe boundary when cancelled or past the deadline."""
        self._last_stage = stage
        if self._cancel_check is not None and self._cancel_check():
            raise ResourceLimitError(
                "resource_cancelled",
                "The operation was cancelled before the next processing stage.",
                stage=stage,
            )
        elapsed = self._clock() - self._started_at
        if self._clock() > self._deadline:
            raise ResourceLimitError(
                "resource_deadline_exceeded",
                "The operation exceeded its configured runtime budget.",
                stage=stage,
                details={
                    "elapsed_seconds": round(elapsed, 6),
                    "limit_seconds": self._timeout_seconds,
                },
            )

    def check_file(self, path: str | Path, *, stage: str = "file_check") -> int:
        """Validate a file's size before a parser can materialize it."""
        self.checkpoint(stage)
        source = Path(path)
        actual = source.stat().st_size
        self._max_observed_file_bytes = max(self._max_observed_file_bytes, actual)
        if actual > self.limits.max_file_bytes:
            raise ResourceLimitError(
                "resource_file_too_large",
                "The input file exceeds the configured size limit.",
                stage=stage,
                details={
                    "path": str(source),
                    "actual_bytes": actual,
                    "limit_bytes": self.limits.max_file_bytes,
                },
            )
        return actual

    def check_matrix_shape(
        self,
        shape: Sequence[int],
        *,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
        target_count: int = 0,
        peak_multiplier: float = 1.0,
        stage: str = "matrix_check",
    ) -> int:
        """Validate dimensions, element count, and estimated peak memory."""
        self.checkpoint(stage)
        if len(shape) != 2:
            raise ValueError(f"Expected a two-dimensional matrix shape, got {shape!r}")
        samples, wavelengths = (int(shape[0]), int(shape[1]))
        if samples < 0 or wavelengths < 0 or target_count < 0:
            raise ValueError("Matrix dimensions and target_count cannot be negative")
        for dimension, actual, limit in (
            ("samples", samples, self.limits.max_samples),
            ("wavelengths", wavelengths, self.limits.max_wavelengths),
            ("targets", int(target_count), self.limits.max_targets),
        ):
            if actual > limit:
                raise ResourceLimitError(
                    "resource_matrix_dimension_exceeded",
                    f"The {dimension} dimension exceeds its configured limit.",
                    stage=stage,
                    details={
                        "dimension": dimension,
                        "actual": actual,
                        "limit": limit,
                        "shape": [samples, wavelengths],
                    },
                )
        elements = samples * wavelengths
        self._max_observed_elements = max(self._max_observed_elements, elements)
        if elements > self.limits.max_matrix_elements:
            raise ResourceLimitError(
                "resource_matrix_too_large",
                "The spectral matrix exceeds the configured element limit.",
                stage=stage,
                details={
                    "shape": [samples, wavelengths],
                    "actual_elements": elements,
                    "limit_elements": self.limits.max_matrix_elements,
                },
            )
        if not math.isfinite(peak_multiplier) or peak_multiplier <= 0:
            raise ValueError("peak_multiplier must be finite and greater than zero")
        itemsize = np.dtype(dtype).itemsize
        estimated = math.ceil(
            (elements + samples * int(target_count)) * itemsize * peak_multiplier
        )
        self._max_estimated_peak_bytes = max(self._max_estimated_peak_bytes, estimated)
        if estimated > self.limits.max_estimated_peak_bytes:
            raise ResourceLimitError(
                "resource_memory_estimate_exceeded",
                "The estimated peak memory exceeds the configured limit.",
                stage=stage,
                details={
                    "shape": [samples, wavelengths],
                    "dtype_itemsize": itemsize,
                    "peak_multiplier": peak_multiplier,
                    "estimated_peak_bytes": estimated,
                    "limit_bytes": self.limits.max_estimated_peak_bytes,
                },
            )
        return estimated

    def check_array(
        self,
        array: np.ndarray,
        *,
        target_count: int = 0,
        peak_multiplier: float = 1.0,
        stage: str = "matrix_check",
    ) -> int:
        """Validate an already materialized two-dimensional array."""
        return self.check_matrix_shape(
            array.shape,
            dtype=array.dtype,
            target_count=target_count,
            peak_multiplier=peak_multiplier,
            stage=stage,
        )

    def evidence(self) -> dict[str, int | float | str]:
        """Return measured high-water marks for logs or successful results."""
        return {
            "elapsed_seconds": round(self._clock() - self._started_at, 6),
            "runtime_limit_seconds": self._timeout_seconds,
            "max_observed_file_bytes": self._max_observed_file_bytes,
            "max_observed_elements": self._max_observed_elements,
            "max_estimated_peak_bytes": self._max_estimated_peak_bytes,
            "last_stage": self._last_stage,
        }
