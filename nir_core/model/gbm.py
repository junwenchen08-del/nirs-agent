"""Gradient Boosting regression.

Uses :class:`sklearn.ensemble.GradientBoostingRegressor` with grid-searched
hyper-parameters (``n_estimators``, ``learning_rate``, ``max_depth``).
Tree-based methods are scale-invariant, so no ``StandardScaler`` is needed.

If XGBoost is installed (``pip install xgboost``), :func:`train_xgboost`
provides an XGBoost-based alternative with the same API.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from nir_core.model.pls import _resolve_cv_splitter

# Keep the interactive default bounded: 8 candidates instead of 27. Deep
# trees are especially expensive and prone to overfit typical small NIR sets.
DEFAULT_N_ESTIMATORS: list[int] = [100, 200]
DEFAULT_LEARNING_RATE: list[float] = [0.05, 0.1]
DEFAULT_MAX_DEPTH: list[int] = [2, 3]


def train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train a Gradient Boosting model with grid-searched hyper-parameters.

    The bounded interactive grid covers ``n_estimators in [100, 200]``,
    ``learning_rate in [0.05, 0.1]``, and ``max_depth in [2, 3]``.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` adapts to sample size,
            ``"loocv"`` forces Leave-One-Out, ``"fixed"`` uses cv_folds.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})"
        )
    n_samples = X_train.shape[0]

    estimator = GradientBoostingRegressor(random_state=random_state)
    param_grid = {
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_depth": DEFAULT_MAX_DEPTH,
    }

    splitter, strategy_label = _resolve_cv_splitter(
        cv_strategy, cv_folds, n_samples, random_state
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=splitter,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_score = float(grid.best_score_)
    best_rmse = float(np.sqrt(-best_score)) if np.isfinite(best_score) else float("inf")

    grid_results = {
        "best_params": dict(grid.best_params_),
        "best_score": best_score,
        "best_rmse": best_rmse,
        "cv_strategy": strategy_label,
        "param_grid": {
            "n_estimators": list(DEFAULT_N_ESTIMATORS),
            "learning_rate": list(DEFAULT_LEARNING_RATE),
            "max_depth": list(DEFAULT_MAX_DEPTH),
        },
    }
    return grid.best_estimator_, grid_results


def predict_gbm(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained Gradient Boosting model.

    Args:
        model: A fitted object exposing ``.predict``.
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train an XGBoost model with grid-searched hyper-parameters.

    Requires the ``xgboost`` package (``pip install xgboost``).

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.

    Raises:
        ImportError: If ``xgboost`` is not installed.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for train_xgboost. Install with: pip install xgboost"
        ) from exc

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})"
        )
    n_samples = X_train.shape[0]

    estimator = XGBRegressor(random_state=random_state, n_jobs=1, verbosity=0)
    param_grid = {
        "n_estimators": DEFAULT_N_ESTIMATORS,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "max_depth": DEFAULT_MAX_DEPTH,
    }

    splitter, strategy_label = _resolve_cv_splitter(
        cv_strategy, cv_folds, n_samples, random_state
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=splitter,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best_score = float(grid.best_score_)
    best_rmse = float(np.sqrt(-best_score)) if np.isfinite(best_score) else float("inf")

    grid_results = {
        "best_params": dict(grid.best_params_),
        "best_score": best_score,
        "best_rmse": best_rmse,
        "cv_strategy": strategy_label,
        "param_grid": {
            "n_estimators": list(DEFAULT_N_ESTIMATORS),
            "learning_rate": list(DEFAULT_LEARNING_RATE),
            "max_depth": list(DEFAULT_MAX_DEPTH),
        },
    }
    return grid.best_estimator_, grid_results


def predict_xgboost(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained XGBoost model.

    Args:
        model: A fitted object exposing ``.predict``.
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_N_ESTIMATORS",
    "predict_gbm",
    "predict_xgboost",
    "train_gbm",
    "train_xgboost",
]
