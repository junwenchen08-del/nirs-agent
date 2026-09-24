"""Wavelength-axis validation and deterministic spectral resampling."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.interpolate import interp1d


def _validated_axis(values: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    axis = np.asarray(values, dtype=float).ravel()
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two wavelength points.")
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} must contain only finite values.")
    differences = np.diff(axis)
    ascending = bool(np.all(differences > 0.0))
    descending = bool(np.all(differences < 0.0))
    if not (ascending or descending):
        raise ValueError(
            f"{name} must be strictly monotonic with no duplicate wavelengths."
        )
    return axis, descending


def _axis_hash(axis: np.ndarray) -> str:
    canonical = np.asarray(axis, dtype="<f8").tobytes(order="C")
    return hashlib.sha256(canonical).hexdigest()


class SpectralAxisAligner:
    """Resample spectra onto a fixed, validated target wavelength axis."""

    def __init__(
        self,
        target_wv: np.ndarray,
        *,
        kind: str = "linear",
        allow_extrapolation: bool = False,
    ) -> None:
        if kind not in {"linear", "nearest", "cubic"}:
            raise ValueError(
                f"kind must be 'linear', 'nearest', or 'cubic'; got {kind!r}."
            )
        target, _ = _validated_axis(target_wv, name="target_wv")
        self.target_wv = target.copy()
        self.kind = kind
        self.allow_extrapolation = bool(allow_extrapolation)

    def fit(self, source_wv: np.ndarray) -> SpectralAxisAligner:
        source, _ = _validated_axis(source_wv, name="source_wv")
        self._validate_coverage(source)
        self.source_wavelengths_ = source.copy()
        self.target_wavelengths_ = self.target_wv.copy()
        self.source_axis_hash_ = _axis_hash(source)
        self.target_axis_hash_ = _axis_hash(self.target_wavelengths_)
        return self

    def _validate_coverage(self, source: np.ndarray) -> None:
        if self.allow_extrapolation:
            return
        source_min, source_max = float(source.min()), float(source.max())
        target_min = float(self.target_wv.min())
        target_max = float(self.target_wv.max())
        tolerance = max(abs(source_min), abs(source_max), 1.0) * 1e-12
        if target_min < source_min - tolerance or target_max > source_max + tolerance:
            raise ValueError(
                "target_wv is outside the source wavelength range and "
                "allow_extrapolation is false."
            )

    def transform(self, X: np.ndarray, source_wv: np.ndarray) -> np.ndarray:
        if not hasattr(self, "target_wavelengths_"):
            raise RuntimeError("SpectralAxisAligner must be fitted before transform().")
        source, descending = _validated_axis(source_wv, name="source_wv")
        fitted_source = self.source_wavelengths_
        if source.shape != fitted_source.shape or not np.array_equal(
            source,
            fitted_source,
        ):
            raise ValueError(
                "source_wv does not match the fitted source wavelength axis."
            )
        arr = np.asarray(X, dtype=float)
        was_1d = arr.ndim == 1
        if was_1d:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {arr.ndim}D.")
        if arr.shape[1] != source.size:
            raise ValueError(
                f"X has {arr.shape[1]} columns but source_wv has {source.size}."
            )
        if not np.isfinite(arr).all():
            raise ValueError("X must contain only finite values for resampling.")
        self._validate_coverage(source)
        if descending:
            source = source[::-1]
            arr = arr[:, ::-1]
        fill_value: str | tuple[float, float]
        fill_value = "extrapolate" if self.allow_extrapolation else (np.nan, np.nan)
        interpolator = interp1d(
            source,
            arr,
            kind=self.kind,
            axis=1,
            bounds_error=not self.allow_extrapolation,
            fill_value=fill_value,
            assume_sorted=True,
        )
        out = np.asarray(interpolator(self.target_wavelengths_), dtype=float)
        return out[0] if was_1d else out

    def fit_transform(self, X: np.ndarray, source_wv: np.ndarray) -> np.ndarray:
        return self.fit(source_wv).transform(X, source_wv)


def resample_spectra(
    X: np.ndarray,
    source_wv: np.ndarray,
    target_wv: np.ndarray,
    *,
    kind: str = "linear",
    allow_extrapolation: bool = False,
) -> np.ndarray:
    """Convenience wrapper for one-off wavelength-axis resampling."""
    return SpectralAxisAligner(
        target_wv,
        kind=kind,
        allow_extrapolation=allow_extrapolation,
    ).fit_transform(X, source_wv)
