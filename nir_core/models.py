"""Core Pydantic data models for nir_core.

All I/O contracts use these strongly-typed models. Placed in a dedicated
``models`` module (not ``__init__``) to avoid circular imports when submodules
import from nir_core.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, field_validator


class SpectralData(BaseModel):
    """Standardized near-infrared spectral data container.

    Attributes:
        X: Spectral matrix of shape (n_samples, n_wavelengths).
        y: Reference values of shape (n_samples,) or None if absent.
        wv: Wavelengths of shape (n_wavelengths,) or None if absent.
        sample_names: Optional sample identifiers.
        source_file: Original file path the data was loaded from.
        original_format: One of "mat" | "csv" | "txt".
    """

    model_config = {"arbitrary_types_allowed": True}

    X: np.ndarray
    y: np.ndarray | None = None
    wv: np.ndarray | None = None
    sample_names: list[str] = Field(default_factory=list)
    source_file: str = ""
    original_format: str = ""

    @field_validator("X", mode="before")
    @classmethod
    def _ensure_X_2d(cls, v):
        """Ensure X is a 2D array; reshape 1D input to (1, n)."""
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"X must be 1D or 2D, got {arr.ndim}D with shape {arr.shape}"
            )
        return arr

    def summary(self) -> dict:
        """Return a compact summary dict suitable for LLM context."""
        return {
            "n_samples": int(self.X.shape[0]),
            "n_wavelengths": int(self.X.shape[1]),
            "wavelength_range": (
                [float(self.wv.min()), float(self.wv.max())]
                if self.wv is not None
                else None
            ),
            "has_reference": self.y is not None,
            "y_range": (
                [float(self.y.min()), float(self.y.max())]
                if self.y is not None
                else None
            ),
            "source_file": self.source_file,
            "original_format": self.original_format,
        }


class PreprocessingStep(BaseModel):
    """A single preprocessing step description.

    Attributes:
        method: Method key in PRESTEP_METHODS (e.g. "snv", "sg_smooth").
        params: Keyword arguments forwarded to the method function.
    """

    method: str
    params: dict = Field(default_factory=dict)


class PreprocessingResult(BaseModel):
    """Result of applying a preprocessing pipeline."""

    model_config = {"arbitrary_types_allowed": True}

    data: SpectralData
    steps: list[PreprocessingStep]
    processing_time: float


class ModelResult(BaseModel):
    """Result of training a calibration model."""

    model_config = {"arbitrary_types_allowed": True}

    method: str
    n_components: int | None = None
    metrics: dict
    model_path: str | None = None
    preprocessing_steps: list[PreprocessingStep] = Field(default_factory=list)
    wavelength_indices: list[int] | None = None


class ReflectionRecord(BaseModel):
    """A single reflection-loop attempt record."""

    attempt: int
    pipeline: list[PreprocessingStep]
    metrics: dict
    decision: str  # "retry" | "pass" | "best_effort"
    reason: str
