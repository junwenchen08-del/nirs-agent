"""Minimal single-target Cal/Val execution; this API cannot accept Test.

Reuse the existing sklearn estimators and bounded grids. Each CV fold fits
the complete preprocessing/model pipeline using only its calibration rows.
High-level family/pipeline choice is deliberately left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from nir_core.diagnostics import compute_residual_diagnostics
from nir_core.models import PreprocessingStep
from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
from nir_core.utils.metrics import compute_metrics
from nir_core.utils.scientific_validation import (
    validate_calibration_dataset,
    validate_partition_separation,
)


@dataclass
class CalibratedRegressor:
    """A fitted complete pipeline, also usable for the final evaluation."""

    preprocessor: PreprocessingPipeline
    estimator: object
    wv: np.ndarray | None

    def predict(self, X):
        return np.asarray(
            self.estimator.predict(self.preprocessor.transform(X, self.wv))
        ).ravel()


@dataclass
class SearchFit:
    model: CalibratedRegressor
    metrics: dict
    parameters: dict
    folds: list[tuple[np.ndarray, np.ndarray]]
    trials: list[dict]
    fits: int
    cv_manifest: dict
    predictions: dict[str, np.ndarray] = field(default_factory=dict)


def _candidates(method, upper):
    if method == "pls":
        return [{"n_components": value} for value in range(1, upper + 1)]
    if method == "ridge":
        from nir_core.model.linear_reg import DEFAULT_ALPHA_GRID

        return list(ParameterGrid({"alpha": DEFAULT_ALPHA_GRID}))
    if method == "svr":
        from nir_core.model.svr import DEFAULT_C_GRID, DEFAULT_GAMMA_GRID

        return list(ParameterGrid({"C": DEFAULT_C_GRID, "gamma": DEFAULT_GAMMA_GRID}))
    if method == "et":
        from nir_core.model.rf import DEFAULT_MAX_DEPTH, DEFAULT_N_ESTIMATORS

        return list(
            ParameterGrid(
                {"max_depth": DEFAULT_MAX_DEPTH, "n_estimators": DEFAULT_N_ESTIMATORS}
            )
        )
    raise ValueError("Only pls, ridge, svr and et are supported by safe search")


def _estimator(method, params):
    if method == "pls":
        return PLSRegression(**({"scale": False} | params))
    if method == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("estimator", Ridge(**params))])
    if method == "svr":
        return Pipeline(
            [("scaler", StandardScaler()), ("svr", SVR(**({"kernel": "rbf"} | params)))]
        )
    return ExtraTreesRegressor(random_state=42, n_jobs=1, **params)


def make_cv_manifest(X_cal, *, cv_folds, groups=None):
    """Exact calibration row membership; no holdout or target inputs."""
    if groups is not None:
        groups = np.asarray(groups)
        if (
            groups.ndim != 1
            or len(groups) != len(X_cal)
            or len(np.unique(groups)) < cv_folds
        ):
            raise ValueError("Insufficient or misaligned calibration groups for CV")
        folds = list(GroupKFold(n_splits=cv_folds).split(X_cal, groups=groups))
    else:
        folds = list(
            KFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X_cal)
        )
    body = {
        "scope": "calibration_cv",
        "splitter": "GroupKFold" if groups is not None else "KFold",
        "random_state": None if groups is not None else 42,
        "n_splits": cv_folds,
        "folds": [
            {"train": train.tolist(), "validation": valid.tolist()}
            for train, valid in folds
        ],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**body, "manifest_id": "cv_" + digest, "sha256": digest}


def fit_search_model(
    X_cal,
    y_cal,
    X_val,
    y_val,
    *,
    method: str,
    steps: list[PreprocessingStep],
    wv=None,
    groups=None,
    cv_folds: int = 5,
    max_components: int = 10,
    parameter_plan: dict | None = None,
    cv_manifest: dict | None = None,
    checkpoint: Callable[[str], None] = lambda _: None,
) -> SearchFit:
    """Select numerical parameters by Cal CV, then evaluate once on Val.

    Phase 1 retains existing bounded grids; custom LLM search spaces belong
    to a later stage. No implicit family or wavelength search is performed.
    """
    X_cal, X_val = np.asarray(X_cal, dtype=float), np.asarray(X_val, dtype=float)
    y_cal, y_val = np.asarray(y_cal, dtype=float), np.asarray(y_val, dtype=float)
    if y_cal.ndim != 1 or y_val.ndim != 1:
        raise ValueError("Safe search requires a single regression target")
    for X, y, minimum in ((X_cal, y_cal, 6), (X_val, y_val, 2)):
        gate = validate_calibration_dataset(X, y, wv, minimum_samples=minimum)
        if not gate["passed"]:
            raise ValueError(
                "Scientific calibration/validation checks failed: "
                + str(gate["errors"])
            )
    if X_cal.shape[1] != X_val.shape[1]:
        raise ValueError("Calibration and validation axes differ")
    if not validate_partition_separation({"calibration": X_cal, "validation": X_val})[
        "passed"
    ]:
        raise ValueError("Calibration and validation spectra overlap")
    if not 2 <= cv_folds <= min(10, len(X_cal)) or not 1 <= max_components <= 20:
        raise ValueError("Invalid CV/component bounds")
    valid, errors = validate_pipeline(steps)
    if not valid:
        raise ValueError("Invalid preprocessing pipeline: " + str(errors))
    expected_cv = make_cv_manifest(X_cal, cv_folds=cv_folds, groups=groups)
    if cv_manifest is not None and cv_manifest != expected_cv:
        raise ValueError("Frozen CV manifest differs from actual calibration folds")
    cv_manifest = expected_cv
    folds = [
        (np.asarray(fold["train"]), np.asarray(fold["validation"]))
        for fold in cv_manifest["folds"]
    ]
    upper = min(
        20 if parameter_plan is not None else max_components,
        X_cal.shape[1],
        min(len(train) - 1 for train, _ in folds),
    )
    for train, valid_rows in folds:
        if not validate_partition_separation(
            {"fold_train": X_cal[train], "fold_validation": X_cal[valid_rows]}
        )["passed"]:
            raise ValueError(
                "Duplicate spectra cross calibration CV folds; supply isolated groups"
            )
    if parameter_plan is None:
        candidates = _candidates(method, upper)
    else:
        from .parameter_control import authorized_candidates

        candidates = authorized_candidates(
            method, parameter_plan, component_upper=upper, cv_folds=cv_folds
        )
    trials, fitted_count = [], 0
    best_params, best_score, best_predictions = None, float("inf"), None
    for params in candidates:
        prediction = np.empty_like(y_cal)
        scores = []
        for train, valid_rows in folds:
            checkpoint("search_cv_fit")
            preprocessor = PreprocessingPipeline(steps).fit(X_cal[train], wv)
            estimator = _estimator(method, params)
            fitted_count += 1
            estimator.fit(preprocessor.transform(X_cal[train], wv), y_cal[train])
            fold_pred = np.asarray(
                estimator.predict(preprocessor.transform(X_cal[valid_rows], wv))
            ).ravel()
            if not np.all(np.isfinite(fold_pred)):
                raise ValueError("Non-finite CV predictions")
            prediction[valid_rows] = fold_pred
            scores.append(float(np.sqrt(np.mean((fold_pred - y_cal[valid_rows]) ** 2))))
        score = float(np.mean(scores))
        trials.append(
            {
                "parameters": params,
                "RMSECV": score,
                "fold_rmse": scores,
                "scope": "calibration_cv",
            }
        )
        if score < best_score:
            best_score, best_params, best_predictions = score, params, prediction.copy()
    if best_params is None:
        raise ValueError("No valid parameter candidate")
    checkpoint("search_calibration_refit")
    preprocessor = PreprocessingPipeline(steps).fit(X_cal, wv)
    estimator = _estimator(method, best_params)
    estimator.fit(preprocessor.transform(X_cal, wv), y_cal)
    fitted_count += 1
    model = CalibratedRegressor(preprocessor, estimator, wv)
    val_prediction = model.predict(X_val)
    cal_prediction = model.predict(X_cal)
    metrics = {
        "evaluation_scope": "validation",
        "test_status": "sealed",
        "method": method,
        "n_components": best_params.get("n_components"),
        "n_samples": len(X_cal),
        "train": {
            **compute_metrics(y_cal, cal_prediction),
            "scope": "calibration",
        },
        "train_cv": {
            **compute_metrics(y_cal, best_predictions),
            "scope": "calibration_cv",
        },
        "val": {**compute_metrics(y_val, val_prediction), "scope": "validation"},
        "RMSECV": best_score,
        "diagnostics": {
            **compute_residual_diagnostics(y_val, val_prediction),
            "scope": "validation",
        },
    }
    actual = (
        estimator.steps[-1][1].get_params(deep=False)
        if isinstance(estimator, Pipeline)
        else estimator.get_params(deep=False)
    )
    return SearchFit(
        model,
        metrics,
        actual if parameter_plan is not None else dict(best_params),
        folds,
        trials,
        fitted_count,
        cv_manifest,
        {
            "calibration": cal_prediction,
            "calibration_cv": best_predictions,
            "validation": val_prediction,
        },
    )
