"""Deterministic preprocessing diagnostics and bounded candidate generation.

The public function accepts only the calibration matrix supplied by its
caller. It cannot inspect a tuning or final holdout partition, which keeps
candidate generation independent from final performance evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
from nir_core.preprocess.registry import CATALOG_VERSION, METHOD_REGISTRY, catalog_hash


@dataclass(frozen=True)
class CandidateBudget:
    max_candidates: int
    max_steps: int


BUDGETS = {
    "small": CandidateBudget(max_candidates=4, max_steps=2),
    "standard": CandidateBudget(max_candidates=8, max_steps=3),
    "extended": CandidateBudget(max_candidates=12, max_steps=3),
}


@dataclass(frozen=True)
class PreprocessingDataProfile:
    n_samples: int
    n_wavelengths: int
    has_wavelength_axis: bool
    regular_wavelength_axis: bool
    spike_row_fraction: float
    baseline_drift_score: float
    high_frequency_noise_score: float
    scatter_variation_score: float
    tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_wavelengths": self.n_wavelengths,
            "has_wavelength_axis": self.has_wavelength_axis,
            "regular_wavelength_axis": self.regular_wavelength_axis,
            "spike_row_fraction": self.spike_row_fraction,
            "baseline_drift_score": self.baseline_drift_score,
            "high_frequency_noise_score": self.high_frequency_noise_score,
            "scatter_variation_score": self.scatter_variation_score,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class PreprocessingCandidate:
    candidate_id: str
    steps: tuple[PreprocessingStep, ...]
    reasons: tuple[str, ...]

    def to_pipeline(self) -> PreprocessingPipeline:
        return PreprocessingPipeline(list(self.steps))

    def as_dict(self) -> dict[str, Any]:
        pipeline = self.to_pipeline()
        return {
            "candidate_id": self.candidate_id,
            "steps": [
                {"method": step.method, "params": dict(step.params)}
                for step in self.steps
            ],
            "description": pipeline.description() if self.steps else "raw",
            "reasons": list(self.reasons),
            "providers": pipeline.provider_manifest(),
        }


@dataclass(frozen=True)
class PreprocessingRecommendation:
    budget: str
    limits: CandidateBudget
    profile: PreprocessingDataProfile
    candidates: tuple[PreprocessingCandidate, ...]
    manual_recommendations: tuple[str, ...]
    exclusions: tuple[dict[str, str], ...]

    def pipelines(self) -> list[PreprocessingPipeline]:
        return [candidate.to_pipeline() for candidate in self.candidates]

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": CATALOG_VERSION,
            "catalog_sha256": catalog_hash(),
            "budget": self.budget,
            "limits": {
                "max_candidates": self.limits.max_candidates,
                "max_steps": self.limits.max_steps,
            },
            "profile": self.profile.as_dict(),
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "manual_recommendations": list(self.manual_recommendations),
            "exclusions": list(self.exclusions),
        }


def _validated_matrix(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"X must be a 2D spectral matrix, got shape {array.shape}.")
    if array.shape[0] < 2 or array.shape[1] < 5:
        raise ValueError("X must contain at least 2 samples and 5 wavelengths.")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite values.")
    return array


def _axis_is_regular(wv: np.ndarray | None, width: int) -> tuple[bool, bool]:
    if wv is None:
        return False, False
    axis = np.asarray(wv, dtype=float).ravel()
    if axis.shape[0] != width:
        raise ValueError(
            f"wavelength axis length {axis.shape[0]} does not match X width {width}."
        )
    if not np.isfinite(axis).all():
        raise ValueError("wavelength axis must contain only finite values.")
    differences = np.diff(axis)
    monotonic = np.all(differences > 0.0) or np.all(differences < 0.0)
    regular = bool(
        monotonic and np.allclose(differences, differences[0], rtol=1e-5, atol=1e-9)
    )
    return True, regular


def profile_preprocessing_data(
    X: np.ndarray,
    wv: np.ndarray | None = None,
) -> PreprocessingDataProfile:
    """Summarize bounded diagnostics without retaining any spectra."""
    array = _validated_matrix(X)
    has_axis, regular_axis = _axis_is_regular(wv, array.shape[1])
    eps = np.finfo(float).eps

    neighbor_residual = array[:, 1:-1] - 0.5 * (array[:, :-2] + array[:, 2:])
    row_scale = np.median(np.abs(np.diff(array, axis=1)), axis=1) + eps
    spike_z = np.max(np.abs(neighbor_residual), axis=1) / row_scale
    spike_row_fraction = float(np.mean(spike_z > 12.0))

    index_axis = np.linspace(-1.0, 1.0, array.shape[1])
    centered_axis = index_axis - index_axis.mean()
    centered_rows = array - array.mean(axis=1, keepdims=True)
    slopes = centered_rows @ centered_axis / float(centered_axis @ centered_axis)
    row_amplitude = np.ptp(array, axis=1)
    baseline_drift_score = float(
        np.median(np.abs(slopes)) / (np.median(row_amplitude) + eps)
    )

    second_difference = np.diff(array, n=2, axis=1)
    high_frequency_noise_score = float(
        np.median(np.abs(second_difference)) / (np.median(row_amplitude) + eps)
    )

    row_std = array.std(axis=1)
    scatter_variation_score = float(np.std(row_std) / (np.median(row_std) + eps))

    tags: list[str] = []
    if spike_row_fraction >= 0.05:
        tags.append("isolated_spikes")
    if baseline_drift_score >= 0.35:
        tags.append("baseline_drift")
    if high_frequency_noise_score >= 0.30:
        tags.extend(["high_frequency_noise", "low_signal_to_noise"])
    elif high_frequency_noise_score >= 0.08:
        tags.append("high_frequency_noise")
    if scatter_variation_score >= 0.10:
        tags.append("scatter_variation")

    return PreprocessingDataProfile(
        n_samples=int(array.shape[0]),
        n_wavelengths=int(array.shape[1]),
        has_wavelength_axis=has_axis,
        regular_wavelength_axis=regular_axis,
        spike_row_fraction=round(spike_row_fraction, 8),
        baseline_drift_score=round(baseline_drift_score, 8),
        high_frequency_noise_score=round(high_frequency_noise_score, 8),
        scatter_variation_score=round(scatter_variation_score, 8),
        tags=tuple(tags),
    )


def _candidate_id(steps: tuple[PreprocessingStep, ...]) -> str:
    if not steps:
        return "raw"
    payload = [
        {"method": step.method, "params": dict(sorted(step.params.items()))}
        for step in steps
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _steps(*items: tuple[str, dict[str, Any]]) -> tuple[PreprocessingStep, ...]:
    return tuple(
        PreprocessingStep(method=method, params=dict(params))
        for method, params in items
    )


def _sg_window(n_wavelengths: int) -> int:
    for candidate in (11, 9, 7, 5):
        if candidate <= n_wavelengths:
            return candidate
    return 5


def recommend_preprocessing(
    X_calibration: np.ndarray,
    wv: np.ndarray | None = None,
    *,
    budget: str = "standard",
) -> PreprocessingRecommendation:
    """Generate deterministic candidates from calibration spectra only."""
    normalized_budget = str(budget).strip().lower()
    try:
        limits = BUDGETS[normalized_budget]
    except KeyError as exc:
        raise ValueError(
            f"budget must be one of {sorted(BUDGETS)}, got {budget!r}."
        ) from exc

    profile = profile_preprocessing_data(X_calibration, wv)
    tags = set(profile.tags)
    window = _sg_window(profile.n_wavelengths)
    proposed: list[tuple[tuple[PreprocessingStep, ...], tuple[str, ...]]] = [
        ((), ("mandatory_raw_baseline",)),
    ]
    manual: list[str] = []
    exclusions: list[dict[str, str]] = []

    if "isolated_spikes" in tags:
        manual.extend(["despike", "median_filter"])
        exclusions.extend(
            [
                {
                    "method": "despike",
                    "reason": "explicit_only_requires_user_confirmation",
                },
                {
                    "method": "median_filter",
                    "reason": "explicit_only_requires_user_confirmation",
                },
            ]
        )
    if "baseline_drift" in tags:
        manual.append("rubberband")
        exclusions.append(
            {
                "method": "rubberband",
                "reason": "explicit_only_requires_user_confirmation",
            }
        )
        proposed.extend(
            [
                (
                    _steps(
                        ("asls", {"lambda_": 1e5, "p": 0.001}),
                        ("snv", {}),
                    ),
                    ("baseline_drift", "scatter_robustness"),
                ),
                (
                    _steps(("detrend", {}), ("snv", {})),
                    ("baseline_drift",),
                ),
            ]
        )
        if normalized_budget == "extended":
            proposed.extend(
                [
                    (
                        _steps(("airpls", {"lambda_": 1e7}), ("snv", {})),
                        ("baseline_drift", "extended_budget"),
                    ),
                    (
                        _steps(
                            ("arpls", {"lambda_": 1e4, "ratio": 0.01}),
                            ("snv", {}),
                        ),
                        ("baseline_drift", "extended_budget"),
                    ),
                ]
            )

    if "high_frequency_noise" in tags:
        proposed.append(
            (
                _steps(
                    ("snv", {}),
                    ("sg_smooth", {"window": window, "order": 2}),
                ),
                ("high_frequency_noise",),
            )
        )
        if normalized_budget == "extended":
            proposed.append(
                (
                    _steps(("whittaker_smooth", {"lambda_": 1e4})),
                    ("high_frequency_noise", "extended_budget"),
                )
            )
    if "scatter_variation" in tags:
        proposed.extend(
            [
                (
                    _steps(("snv", {}), ("mean_center", {})),
                    ("scatter_variation",),
                ),
                (
                    _steps(("msc", {}), ("mean_center", {})),
                    ("scatter_variation",),
                ),
            ]
        )
        if normalized_budget == "extended":
            proposed.append(
                (
                    _steps(("rnv", {"percentile": 25.0})),
                    ("scatter_variation", "heavy_tail_robustness", "extended_budget"),
                )
            )

    proposed.extend(
        [
            (_steps(("snv", {})), ("stable_scatter_baseline",)),
            (_steps(("msc", {})), ("stateful_scatter_baseline",)),
            (
                _steps(("sg_smooth", {"window": window, "order": 2})),
                ("stable_noise_baseline",),
            ),
            (
                _steps(
                    ("snv", {}),
                    ("sg_smooth", {"window": window, "order": 2}),
                    ("mean_center", {}),
                ),
                ("combined_default",),
            ),
        ]
    )
    if "low_signal_to_noise" not in tags:
        proposed.append(
            (
                _steps(
                    (
                        "derivative1",
                        {
                            "window": window,
                            "order": 2,
                            "spacing_mode": "index",
                        },
                    ),
                    ("mean_center", {}),
                ),
                ("baseline_shape_separation",),
            )
        )
    else:
        exclusions.append(
            {
                "method": "derivative1",
                "reason": "low_signal_to_noise",
            }
        )

    candidates: list[PreprocessingCandidate] = []
    seen: set[str] = set()
    for steps, reasons in proposed:
        if len(steps) > limits.max_steps:
            continue
        invalid_auto_methods = [
            step.method
            for step in steps
            if METHOD_REGISTRY[step.method].auto_level not in {"default", "conditional"}
        ]
        if invalid_auto_methods:
            for method in invalid_auto_methods:
                exclusions.append(
                    {"method": method, "reason": "not_enabled_for_automatic_selection"}
                )
            continue
        is_valid, reason = validate_pipeline(list(steps))
        if not is_valid:
            exclusions.append({"method": "pipeline", "reason": reason})
            continue
        candidate_id = _candidate_id(steps)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        candidates.append(
            PreprocessingCandidate(
                candidate_id=candidate_id,
                steps=steps,
                reasons=reasons,
            )
        )
        if len(candidates) >= limits.max_candidates:
            break

    return PreprocessingRecommendation(
        budget=normalized_budget,
        limits=limits,
        profile=profile,
        candidates=tuple(candidates),
        manual_recommendations=tuple(dict.fromkeys(manual)),
        exclusions=tuple(
            {"method": item["method"], "reason": item["reason"]}
            for item in {
                (entry["method"], entry["reason"]): entry for entry in exclusions
            }.values()
        ),
    )


__all__ = [
    "BUDGETS",
    "CandidateBudget",
    "PreprocessingCandidate",
    "PreprocessingDataProfile",
    "PreprocessingRecommendation",
    "profile_preprocessing_data",
    "recommend_preprocessing",
]
