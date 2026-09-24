"""Chemotools-backed calibration transfer with explicit direction and pairing.

Calibration transfer is intentionally separate from ordinary preprocessing:
it learns a mapping from paired spectra measured on a target instrument into
the spectral space of a source/reference instrument.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
from chemotools.adaptation import (
    DirectStandardization,
    PiecewiseDirectStandardization,
    SpectralSpaceTransform,
)

from nir_core.preprocess.alignment import SpectralAxisAligner

TRANSFER_CATALOG_VERSION = "1.0"
PROVIDER = "chemotools"
PROVIDER_VERSION = "0.4.4"
IMPLEMENTATION_VERSION = "calibration-transfer-adapter-v1"

_METHODS: dict[str, dict[str, Any]] = {
    "ds": {
        "method_id": "ds",
        "display_name_zh": "直接标准化（DS）",
        "summary_zh": "学习目标仪器到参考仪器的全局线性映射。",
        "provider_class": "chemotools.adaptation.DirectStandardization",
        "parameter_schema": {},
    },
    "pds": {
        "method_id": "pds",
        "display_name_zh": "分段直接标准化（PDS）",
        "summary_zh": "按局部波长窗口分别学习目标仪器到参考仪器的映射。",
        "provider_class": ("chemotools.adaptation.PiecewiseDirectStandardization"),
        "parameter_schema": {
            "window_length": {
                "type": "integer",
                "default": 25,
                "minimum": 1,
                "description_zh": "局部窗口半宽",
            },
            "n_components": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description_zh": "每个局部 PLS 模型的成分数",
            },
            "scale": {
                "type": "boolean",
                "default": True,
                "description_zh": "局部 PLS 是否缩放输入",
            },
            "storage": {
                "type": "string",
                "default": "band",
                "choices": ["dense", "band"],
                "description_zh": "转换系数存储方式",
            },
        },
    },
    "sst": {
        "method_id": "sst",
        "display_name_zh": "光谱空间转换（SST）",
        "summary_zh": "在低维联合光谱空间中对齐目标仪器与参考仪器。",
        "provider_class": "chemotools.adaptation.SpectralSpaceTransform",
        "parameter_schema": {
            "n_components": {
                "type": "integer",
                "default": 2,
                "minimum": 1,
                "description_zh": "联合光谱空间成分数",
            },
            "with_mean": {
                "type": "boolean",
                "default": True,
                "description_zh": "拟合前是否按仪器分别中心化",
            },
            "with_std": {
                "type": "boolean",
                "default": False,
                "description_zh": "拟合前是否按仪器分别标准化",
            },
        },
    },
}


def _matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional spectral matrix.")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError(f"{name} must contain at least 2 samples and 2 features.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _axis(values: np.ndarray, *, width: int, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float).ravel()
    if axis.shape[0] != width:
        raise ValueError(
            f"{name} length {axis.shape[0]} does not match spectral width {width}."
        )
    if not np.isfinite(axis).all():
        raise ValueError(f"{name} must contain only finite values.")
    differences = np.diff(axis)
    if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
        raise ValueError(f"{name} must be strictly monotonic without duplicates.")
    return axis


def _hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _hash_pairs(
    X_source: np.ndarray,
    X_target: np.ndarray,
    sample_ids: Sequence[str] | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(_hash_array(X_source).encode("ascii"))
    digest.update(_hash_array(X_target).encode("ascii"))
    if sample_ids is not None:
        for sample_id in sample_ids:
            digest.update(str(sample_id).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def list_transfer_methods() -> list[dict[str, Any]]:
    """Return the compact, deterministic calibration-transfer catalog."""
    return [describe_transfer_method(method_id) for method_id in sorted(_METHODS)]


def describe_transfer_method(method_id: str) -> dict[str, Any]:
    """Describe one transfer method and its public parameters."""
    normalized = str(method_id).strip().lower()
    try:
        method = _METHODS[normalized]
    except KeyError as exc:
        raise KeyError(
            f"Unknown calibration transfer method {method_id!r}; "
            f"available: {sorted(_METHODS)}"
        ) from exc
    return {
        **method,
        "provider": PROVIDER,
        "provider_version": PROVIDER_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "direction": "target_instrument_to_source_instrument",
        "requires_paired_samples": True,
        "ordinary_preprocessing": False,
    }


def pair_transfer_samples(
    X_source: np.ndarray,
    X_target: np.ndarray,
    *,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate IDs and reorder target spectra to the source sample order."""
    source = _matrix(X_source, name="X_source")
    target = _matrix(X_target, name="X_target")
    normalized_source = tuple(str(value).strip() for value in source_ids)
    normalized_target = tuple(str(value).strip() for value in target_ids)
    if (
        len(normalized_source) != source.shape[0]
        or len(normalized_target) != target.shape[0]
    ):
        raise ValueError("Sample ID count must match the corresponding row count.")
    if any(not value for value in (*normalized_source, *normalized_target)):
        raise ValueError("Sample IDs must be non-empty.")
    if len(set(normalized_source)) != len(normalized_source) or len(
        set(normalized_target)
    ) != len(normalized_target):
        raise ValueError("Sample IDs must be unique within each instrument.")
    if set(normalized_source) != set(normalized_target):
        raise ValueError(
            "Source and target instruments must contain the same sample IDs."
        )
    target_lookup = {
        sample_id: index for index, sample_id in enumerate(normalized_target)
    }
    target_order = [target_lookup[sample_id] for sample_id in normalized_source]
    return source.copy(), target[target_order].copy(), normalized_source


def paired_train_validation_indices(
    n_samples: int,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one deterministic paired split without breaking row correspondence."""
    if n_samples < 8:
        raise ValueError("Calibration transfer requires at least 8 paired samples.")
    if not 0.1 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.1 and 0.5.")
    n_validation = max(2, round(n_samples * validation_fraction))
    if n_samples - n_validation < 4:
        raise ValueError(
            "Calibration-transfer training split must contain at least 4 pairs."
        )
    indices = np.random.default_rng(random_state).permutation(n_samples)
    return np.sort(indices[n_validation:]), np.sort(indices[:n_validation])


def _build_transformer(method: str, params: Mapping[str, Any]):
    allowed = set(_METHODS[method]["parameter_schema"])
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown parameters for calibration transfer method {method!r}: {unknown}."
        )

    def integer(name: str, default: int) -> int:
        value = params.get(name, default)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer.")
        normalized = int(value)
        if normalized < 1:
            raise ValueError(f"{name} must be at least 1.")
        return normalized

    def boolean(name: str, default: bool) -> bool:
        value = params.get(name, default)
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean.")
        return bool(value)

    if method == "ds":
        return DirectStandardization()
    if method == "pds":
        storage = params.get("storage", "band")
        if storage not in {"dense", "band"}:
            raise ValueError("storage must be either 'dense' or 'band'.")
        return PiecewiseDirectStandardization(
            window_length=integer("window_length", 25),
            n_components=integer("n_components", 2),
            scale=boolean("scale", True),
            storage=storage,
        )
    return SpectralSpaceTransform(
        n_components=integer("n_components", 2),
        with_mean=boolean("with_mean", True),
        with_std=boolean("with_std", False),
    )


class CalibrationTransfer:
    """Fitted target-to-source instrument transfer with axis provenance."""

    def __init__(
        self,
        *,
        method: str,
        source_instrument_id: str,
        target_instrument_id: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_method = str(method).strip().lower()
        if normalized_method not in _METHODS:
            raise ValueError(
                f"method must be one of {sorted(_METHODS)}, got {method!r}."
            )
        self.method = normalized_method
        self.source_instrument_id = str(source_instrument_id).strip()
        self.target_instrument_id = str(target_instrument_id).strip()
        if not self.source_instrument_id or not self.target_instrument_id:
            raise ValueError("Both source and target instrument IDs are required.")
        if self.source_instrument_id == self.target_instrument_id:
            raise ValueError("Source and target instrument IDs must be different.")
        self.parameters = dict(parameters or {})
        _build_transformer(self.method, self.parameters)

    def fit(
        self,
        X_target: np.ndarray,
        X_source: np.ndarray,
        *,
        target_wv: np.ndarray,
        source_wv: np.ndarray,
        sample_ids: Sequence[str] | None = None,
    ) -> CalibrationTransfer:
        """Fit the mapping from target spectra to source/reference spectra."""
        target = _matrix(X_target, name="X_target")
        source = _matrix(X_source, name="X_source")
        if target.shape[0] != source.shape[0]:
            raise ValueError(
                "X_target and X_source must contain the same number of pairs."
            )
        target_axis = _axis(target_wv, width=target.shape[1], name="target_wv")
        source_axis = _axis(source_wv, width=source.shape[1], name="source_wv")

        axes_equal = target_axis.shape == source_axis.shape and np.array_equal(
            target_axis,
            source_axis,
        )
        self.axis_aligner_ = None
        target_for_fit = target
        if not axes_equal:
            self.axis_aligner_ = SpectralAxisAligner(
                source_axis,
                kind="linear",
                allow_extrapolation=False,
            ).fit(target_axis)
            target_for_fit = self.axis_aligner_.transform(target, target_axis)
        if target_for_fit.shape != source.shape:
            raise ValueError(
                "Aligned target spectra and source spectra must have the same shape."
            )

        transformer = _build_transformer(self.method, self.parameters)
        transformer.fit(target_for_fit, X_source=source)
        self.transformer_ = transformer
        self.source_wavelengths_ = source_axis.copy()
        self.target_wavelengths_ = target_axis.copy()
        self.source_axis_hash_ = _hash_array(source_axis)
        self.target_axis_hash_ = _hash_array(target_axis)
        self.paired_training_data_hash_ = _hash_pairs(source, target, sample_ids)
        self.n_pairs_ = int(source.shape[0])
        self.n_source_features_ = int(source.shape[1])
        self.n_target_features_ = int(target.shape[1])
        return self

    def transform(
        self,
        X_target: np.ndarray,
        *,
        target_wv: np.ndarray,
        target_instrument_id: str | None = None,
    ) -> np.ndarray:
        """Transform target-instrument spectra into the source instrument space."""
        if not hasattr(self, "transformer_"):
            raise RuntimeError("CalibrationTransfer must be fitted before transform().")
        if (
            target_instrument_id is not None
            and str(target_instrument_id).strip() != self.target_instrument_id
        ):
            raise ValueError(
                "Target instrument does not match the fitted calibration-transfer artifact."
            )
        target = _matrix(X_target, name="X_target")
        axis = _axis(target_wv, width=target.shape[1], name="target_wv")
        if axis.shape != self.target_wavelengths_.shape or not np.array_equal(
            axis,
            self.target_wavelengths_,
        ):
            raise ValueError(
                "target_wv does not match the target axis bound to this transfer artifact."
            )
        if self.axis_aligner_ is not None:
            target = self.axis_aligner_.transform(target, axis)
        result = np.asarray(self.transformer_.transform(target), dtype=float)
        if result.shape[1] != self.n_source_features_ or not np.isfinite(result).all():
            raise ValueError("Calibration transfer produced invalid output.")
        return result

    def manifest(self) -> dict[str, Any]:
        """Return compact provenance for persistence and model binding."""
        if not hasattr(self, "transformer_"):
            raise RuntimeError("CalibrationTransfer must be fitted before manifest().")
        method = describe_transfer_method(self.method)
        return {
            "schema_version": 1,
            "artifact_type": "nir_calibration_transfer",
            "catalog_version": TRANSFER_CATALOG_VERSION,
            "transfer_method": self.method,
            "direction": (
                f"{self.target_instrument_id}_to_{self.source_instrument_id}"
            ),
            "source_instrument_id": self.source_instrument_id,
            "target_instrument_id": self.target_instrument_id,
            "source_axis_hash": self.source_axis_hash_,
            "target_axis_hash": self.target_axis_hash_,
            "paired_training_data_hash": self.paired_training_data_hash_,
            "n_pairs": self.n_pairs_,
            "source_feature_count": self.n_source_features_,
            "target_feature_count": self.n_target_features_,
            "provider": PROVIDER,
            "provider_version": PROVIDER_VERSION,
            "provider_class": method["provider_class"],
            "implementation_version": IMPLEMENTATION_VERSION,
            "parameters": dict(self.parameters),
            "axis_alignment": {
                "applied": self.axis_aligner_ is not None,
                "method": "linear" if self.axis_aligner_ is not None else None,
                "allow_extrapolation": False,
            },
        }


def _agreement_metrics(
    reference: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(reference, dtype=float).ravel()
    predicted = np.asarray(candidate, dtype=float).ravel()
    residual = predicted - truth
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    centered_truth = truth - truth.mean()
    denominator = float(centered_truth @ centered_truth)
    r2 = float(1.0 - (residual @ residual) / denominator) if denominator > 0 else 0.0
    slope = (
        float((centered_truth @ (predicted - predicted.mean())) / denominator)
        if denominator > 0
        else 0.0
    )
    return {"rmse": rmse, "bias": bias, "slope": slope, "r2": r2}


def evaluate_spectral_transfer(
    X_source: np.ndarray,
    X_target_aligned: np.ndarray,
    X_transferred: np.ndarray,
) -> dict[str, Any]:
    """Compare target/source agreement before and after transfer."""
    source = _matrix(X_source, name="X_source")
    before = _matrix(X_target_aligned, name="X_target_aligned")
    after = _matrix(X_transferred, name="X_transferred")
    if source.shape != before.shape or source.shape != after.shape:
        raise ValueError("Source, aligned target, and transferred matrices must match.")
    before_metrics = _agreement_metrics(source, before)
    after_metrics = _agreement_metrics(source, after)
    baseline = before_metrics["rmse"]
    improvement = (
        100.0 * (baseline - after_metrics["rmse"]) / baseline if baseline > 0.0 else 0.0
    )
    return {
        "before": before_metrics,
        "after": after_metrics,
        "improvement_percent": float(improvement),
    }


def evaluate_prediction_transfer(
    y_true: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
) -> dict[str, Any]:
    """Compare reference-model predictions before and after spectral transfer."""
    truth = np.asarray(y_true, dtype=float).ravel()
    before = np.asarray(y_before, dtype=float).ravel()
    after = np.asarray(y_after, dtype=float).ravel()
    if truth.size < 2 or before.shape != truth.shape or after.shape != truth.shape:
        raise ValueError(
            "y_true, y_before, and y_after must be equal-length vectors with at "
            "least two samples."
        )
    if not (
        np.isfinite(truth).all()
        and np.isfinite(before).all()
        and np.isfinite(after).all()
    ):
        raise ValueError("Prediction-transfer metrics require finite values.")

    def metrics(predicted: np.ndarray) -> dict[str, float]:
        agreement = _agreement_metrics(truth, predicted)
        return {
            "rmsep": agreement["rmse"],
            "bias": agreement["bias"],
            "slope": agreement["slope"],
            "r2": agreement["r2"],
        }

    before_metrics = metrics(before)
    after_metrics = metrics(after)
    baseline = before_metrics["rmsep"]
    improvement = (
        100.0 * (baseline - after_metrics["rmsep"]) / baseline
        if baseline > 0.0
        else 0.0
    )
    return {
        "before": before_metrics,
        "after": after_metrics,
        "improvement_percent": float(improvement),
    }


__all__ = [
    "IMPLEMENTATION_VERSION",
    "PROVIDER",
    "PROVIDER_VERSION",
    "TRANSFER_CATALOG_VERSION",
    "CalibrationTransfer",
    "describe_transfer_method",
    "evaluate_prediction_transfer",
    "evaluate_spectral_transfer",
    "list_transfer_methods",
    "pair_transfer_samples",
    "paired_train_validation_indices",
]
