"""Pure candidate-policy decisions for NIR wavelength and model selection."""

from __future__ import annotations

import numpy as np

_MODEL_METHODS = (
    "pls",
    "pcr",
    "svr",
    "rf",
    "et",
    "gbm",
    "ridge",
    "lasso",
    "elasticnet",
    "knn",
    "mlp",
    "cnn",
)


def _decide_autonomous_wavelength_selection(
    *,
    compare_cars: bool | None,
    n_calibration: int,
    n_wavelengths: int,
    baseline_rmse: float,
    y_tuning: np.ndarray,
    max_selection_samples: int = 750,
) -> dict:
    """Decide whether CARS is worth evaluating without consulting test data."""
    n_calibration = int(n_calibration)
    n_wavelengths = int(n_wavelengths)
    max_selection_samples = max(30, int(max_selection_samples))
    y_tuning = np.asarray(y_tuning, dtype=float).ravel()
    tuning_variance = float(np.var(y_tuning)) if y_tuning.size else 0.0
    baseline_r2 = float(1.0 - float(baseline_rmse) ** 2 / tuning_variance) if tuning_variance > 0 else None
    ratio = float(n_wavelengths / max(1, n_calibration))
    cars_fit_samples = min(n_calibration, max_selection_samples)
    common = {
        "n_calibration": n_calibration,
        "n_wavelengths": n_wavelengths,
        "wavelength_to_sample_ratio": ratio,
        "baseline_RMSE_tuning": float(baseline_rmse),
        "baseline_R2_tuning": baseline_r2,
        "cars_fit_samples": cars_fit_samples,
        "sample_cap_applied": cars_fit_samples < n_calibration,
        "max_selection_samples": max_selection_samples,
    }

    if compare_cars is False:
        return {
            **common,
            "mode": "disabled",
            "evaluate_cars": False,
            "reason_code": "explicitly_disabled",
            "reason": "CARS comparison was explicitly disabled by the caller.",
            "signals": [],
        }
    if compare_cars is True:
        return {
            **common,
            "mode": "forced",
            "evaluate_cars": True,
            "reason_code": "explicitly_enabled",
            "reason": "CARS comparison was explicitly requested by the caller.",
            "signals": ["explicit_request"],
        }
    if n_calibration < 30:
        return {
            **common,
            "mode": "auto",
            "evaluate_cars": False,
            "reason_code": "insufficient_calibration_samples",
            "reason": "Fewer than 30 calibration samples make autonomous CARS unstable.",
            "signals": [],
        }
    if n_wavelengths < 40:
        return {
            **common,
            "mode": "auto",
            "evaluate_cars": False,
            "reason_code": "insufficient_wavelengths",
            "reason": "Fewer than 40 wavelengths provide little scope for useful variable reduction.",
            "signals": [],
        }

    signals: list[str] = []
    if n_wavelengths >= 150:
        signals.append("many_wavelengths")
    if ratio >= 0.5:
        signals.append("high_wavelength_to_sample_ratio")
    if baseline_r2 is not None and baseline_r2 < 0.85:
        signals.append("weak_full_spectrum_tuning_fit")
    evaluate = bool(signals)
    return {
        **common,
        "mode": "auto",
        "evaluate_cars": evaluate,
        "reason_code": "selection_signals_detected" if evaluate else "full_spectrum_sufficient",
        "reason": ("CARS will be compared because: " + ", ".join(signals) + "." if evaluate else "The full-spectrum problem is small and well-conditioned; CARS is unlikely to justify its cost."),
        "signals": signals,
    }


def _choose_wavelength_candidate(
    candidate_results: list[dict],
    *,
    min_relative_improvement: float = 0.005,
) -> tuple[dict, dict]:
    """Adopt CARS only when it materially improves tuning RMSE."""
    full = next((item for item in candidate_results if item["method"] == "none"), None)
    cars = next((item for item in candidate_results if item["method"] == "cars"), None)
    if full is None:
        chosen = min(candidate_results, key=lambda item: item["RMSE_tuning"])
        return chosen, {"reason_code": "no_full_spectrum_candidate"}
    if cars is None:
        return full, {
            "reason_code": "cars_not_evaluated",
            "relative_RMSE_improvement": None,
        }
    denominator = max(abs(float(full["RMSE_tuning"])), np.finfo(float).eps)
    improvement = (float(full["RMSE_tuning"]) - float(cars["RMSE_tuning"])) / denominator
    if improvement >= float(min_relative_improvement):
        return cars, {
            "reason_code": "cars_improved_tuning_rmse",
            "relative_RMSE_improvement": float(improvement),
            "minimum_required_improvement": float(min_relative_improvement),
        }
    return full, {
        "reason_code": "cars_improvement_below_threshold",
        "relative_RMSE_improvement": float(improvement),
        "minimum_required_improvement": float(min_relative_improvement),
    }


def _wavelength_candidate_summary(candidate_results: list[dict]) -> list[dict]:
    """Remove large index arrays while retaining auditable tuning evidence."""
    return [
        {
            "method": item["method"],
            "n_selected": int(item["n_selected"]),
            "n_components": int(item["n_components"]),
            "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
        }
        for item in candidate_results
    ]


def _decide_autonomous_model_selection(
    *,
    requested_method: str | None,
    n_calibration: int,
    n_features: int,
    baseline_rmse: float,
    y_tuning: np.ndarray,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
) -> dict:
    """Choose a bounded set of model families without consulting test data."""
    requested = (requested_method or "auto").strip().lower()
    if requested != "auto" and requested not in _MODEL_METHODS:
        raise ValueError(f"Unknown method {requested_method!r}; use auto or one of: {', '.join(_MODEL_METHODS)}")

    n_calibration = int(n_calibration)
    n_features = int(n_features)
    max_svr_samples = max(30, int(max_svr_samples))
    max_tree_samples = max(80, int(max_tree_samples))
    y_tuning = np.asarray(y_tuning, dtype=float).ravel()
    tuning_variance = float(np.var(y_tuning)) if y_tuning.size else 0.0
    baseline_r2 = float(1.0 - float(baseline_rmse) ** 2 / tuning_variance) if tuning_variance > 0 else None
    feature_ratio = float(n_features / max(1, n_calibration))
    common = {
        "n_calibration": n_calibration,
        "n_features": n_features,
        "feature_to_sample_ratio": feature_ratio,
        "pls_baseline_RMSE_tuning": float(baseline_rmse),
        "pls_baseline_R2_tuning": baseline_r2,
        "max_svr_samples": max_svr_samples,
        "max_tree_samples": max_tree_samples,
    }
    if requested != "auto":
        return {
            **common,
            "mode": "forced",
            "candidate_methods": [requested],
            "reason_code": "explicit_method",
            "reason": f"The caller explicitly requested method={requested!r}.",
            "signals": ["explicit_request"],
            "runtime_caps_applied": [],
        }
    if n_calibration < 30 or baseline_r2 is None:
        return {
            **common,
            "mode": "auto",
            "candidate_methods": ["pls"],
            "reason_code": "insufficient_model_selection_evidence",
            "reason": "The calibration/tuning data are insufficient for stable model-family comparison.",
            "signals": [],
            "runtime_caps_applied": [],
        }

    signals: list[str] = []
    if feature_ratio >= 0.5:
        signals.append("high_feature_to_sample_ratio")
    if baseline_r2 < 0.9:
        signals.append("weak_pls_tuning_fit")
    if baseline_r2 < 0.75:
        signals.append("possible_nonlinearity")

    candidates = ["pls"]
    if "high_feature_to_sample_ratio" in signals or "weak_pls_tuning_fit" in signals:
        candidates.append("ridge")

    runtime_caps: list[str] = []
    if "weak_pls_tuning_fit" in signals:
        if n_calibration <= max_svr_samples:
            candidates.append("svr")
        else:
            runtime_caps.append("svr")
    if "possible_nonlinearity" in signals and n_calibration >= 80:
        if n_calibration <= max_tree_samples:
            candidates.append("et")
        else:
            runtime_caps.append("et")

    return {
        **common,
        "mode": "auto",
        "candidate_methods": candidates,
        "reason_code": "model_comparison_signals_detected" if len(candidates) > 1 else "pls_baseline_sufficient",
        "reason": ("Alternative model families will be compared because: " + ", ".join(signals) + "." if len(candidates) > 1 else "The compact, strong PLS baseline does not justify additional model-family searches."),
        "signals": signals,
        "runtime_caps_applied": runtime_caps,
    }


def _choose_model_candidate(
    candidate_results: list[dict],
    *,
    min_relative_improvement: float = 0.01,
) -> tuple[dict, dict]:
    """Prefer PLS unless an alternative materially improves tuning RMSE."""
    if not candidate_results:
        raise ValueError("No successful model candidates are available")
    pls = next((item for item in candidate_results if item["method"] == "pls"), None)
    if pls is None:
        chosen = min(candidate_results, key=lambda item: float(item["RMSE_tuning"]))
        return chosen, {
            "reason_code": "explicit_or_no_pls_baseline",
            "relative_RMSE_improvement": None,
            "minimum_required_improvement": float(min_relative_improvement),
        }
    alternatives = [item for item in candidate_results if item["method"] != "pls"]
    if not alternatives:
        return pls, {
            "reason_code": "pls_only_candidate",
            "relative_RMSE_improvement": None,
            "minimum_required_improvement": float(min_relative_improvement),
        }
    best_alternative = min(alternatives, key=lambda item: float(item["RMSE_tuning"]))
    denominator = max(abs(float(pls["RMSE_tuning"])), np.finfo(float).eps)
    improvement = (float(pls["RMSE_tuning"]) - float(best_alternative["RMSE_tuning"])) / denominator
    if improvement >= float(min_relative_improvement):
        return best_alternative, {
            "reason_code": "alternative_improved_tuning_rmse",
            "relative_RMSE_improvement": float(improvement),
            "minimum_required_improvement": float(min_relative_improvement),
        }
    return pls, {
        "reason_code": "alternative_improvement_below_threshold",
        "relative_RMSE_improvement": float(improvement),
        "minimum_required_improvement": float(min_relative_improvement),
    }


def _model_candidate_summary(candidate_results: list[dict]) -> list[dict]:
    """Return JSON-safe, decision-grade model comparison evidence."""
    return [
        {
            "method": item["method"],
            "n_components": item.get("n_components"),
            "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            "RMSECV": None if item.get("RMSECV") is None else round(float(item["RMSECV"]), 5),
        }
        for item in candidate_results
    ]
