"""Core Pydantic data models for nir_core.

All I/O contracts use these strongly-typed models. Placed in a dedicated
``models`` module (not ``__init__``) to avoid circular imports when submodules
import from nir_core.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field, field_validator

from nir_core.utils.spectral_axis import summarize_spectral_axis


class SpectralData(BaseModel):
    """Standardized near-infrared spectral data container.

    Attributes:
        X: Spectral matrix of shape (n_samples, n_wavelengths).
        y: Reference values of shape (n_samples,), (n_samples, n_targets),
            or None if absent.
        y_names: Optional names for the reference-value columns.
        wv: Wavelengths of shape (n_wavelengths,) or None if absent.
        sample_names: Optional sample identifiers.
        source_file: Original file path the data was loaded from.
        original_format: One of "mat" | "csv" | "txt".
    """

    model_config = {"arbitrary_types_allowed": True}

    X: np.ndarray
    y: np.ndarray | None = None
    y_names: list[str] | None = None
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

    @field_validator("y", mode="before")
    @classmethod
    def _ensure_y_1d_or_2d(cls, v):
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim not in {1, 2}:
            raise ValueError(
                f"y must be 1D or 2D, got {arr.ndim}D with shape {arr.shape}"
            )
        return arr

    def summary(self) -> dict:
        """Return a compact summary dict suitable for LLM context."""
        n_components = 0
        y_names: list[str] = []
        y_ranges: dict[str, list[float]] | None = None
        if self.y is not None:
            y_arr = np.asarray(self.y, dtype=float)
            n_components = 1 if y_arr.ndim == 1 else int(y_arr.shape[1])
            y_names = self.y_names or [f"y{i}" for i in range(n_components)]
            y_2d = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
            y_ranges = {
                name: [float(y_2d[:, i].min()), float(y_2d[:, i].max())]
                for i, name in enumerate(y_names)
            }
        raw_wavelength_range: list[float] | None = None
        usable_wavelength_range: list[float] | None = None
        constant_wavelength_count = 0
        usable_wavelength_count = int(self.X.shape[1])
        axis_summary = summarize_spectral_axis(self.wv)
        if self.wv is not None:
            wavelengths = np.asarray(self.wv, dtype=float).ravel()
            finite_wavelengths = wavelengths[np.isfinite(wavelengths)]
            if finite_wavelengths.size:
                raw_wavelength_range = [
                    float(finite_wavelengths.min()),
                    float(finite_wavelengths.max()),
                ]
            if wavelengths.size == self.X.shape[1]:
                finite_spectra = np.isfinite(self.X).all(axis=0)
                column_ranges = np.ptp(self.X, axis=0)
                usable_mask = finite_spectra & (column_ranges > 1e-12)
                constant_wavelength_count = int(
                    np.count_nonzero(finite_spectra & ~usable_mask)
                )
                usable_wavelength_count = int(np.count_nonzero(usable_mask))
                usable_wavelengths = wavelengths[usable_mask & np.isfinite(wavelengths)]
                if usable_wavelengths.size:
                    usable_wavelength_range = [
                        float(usable_wavelengths.min()),
                        float(usable_wavelengths.max()),
                    ]
        return {
            "n_samples": int(self.X.shape[0]),
            "n_wavelengths": int(self.X.shape[1]),
            "wavelength_range": raw_wavelength_range,
            "raw_wavelength_range": raw_wavelength_range,
            "usable_wavelength_range": usable_wavelength_range,
            **axis_summary,
            "constant_wavelength_count": constant_wavelength_count,
            "usable_wavelength_count": usable_wavelength_count,
            "wavelength_range_semantics": (
                "raw includes all measured columns; usable spans non-constant columns only and does not imply those columns were removed"
            ),
            "has_reference": self.y is not None,
            "n_components": n_components,
            "y_names": y_names,
            "y_range": (
                [float(self.y.min()), float(self.y.max())]
                if self.y is not None and np.asarray(self.y).ndim == 1
                else None
            ),
            "y_ranges": y_ranges,
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
