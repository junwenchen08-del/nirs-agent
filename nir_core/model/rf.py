"""Random Forest regression.

Uses :class:`sklearn.ensemble.RandomForestRegressor` with grid-searched
hyper-parameters (``n_estimators`` and ``max_depth``). Tree-based methods
are scale-invariant, so no ``StandardScaler`` is needed.

Extra Trees (``et``) is also provided as a variant — it uses random splits
instead of optimal splits, which can reduce variance at the cost of
slightly higher bias.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from nir_core.model.pls import _resolve_cv_splitter

# Bounded interactive grid. Larger forests and very deep candidates add major
# latency with little value on the small calibration sets typical of NIR work.
DEFAULT_N_ESTIMATORS: list[int] = [100, 200]
DEFAULT_MAX_DEPTH: list[int | None] = [None, 10]


def _train_tree_ensemble(
    estimator_cls: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Generic tree-ensemble trainer with grid search.

    Args:
        estimator_cls: ``RandomForestRegressor`` or ``ExtraTreesRegressor``.
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
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

    estimator = estimator_cls(random_state=random_state, n_jobs=1)
    param_grid = {
        "n_estimators": DEFAULT_N_ESTIMATORS,
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
            "max_depth": [str(d) for d in DEFAULT_MAX_DEPTH],
        },
    }
    return grid.best_estimator_, grid_results


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train a Random Forest model with grid-searched hyper-parameters.

    The bounded grid covers ``n_estimators in [100, 200]`` and
    ``max_depth in [None, 10]``.

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
    return _train_tree_ensemble(
        RandomForestRegressor, X_train, y_train, cv_folds, cv_strategy, random_state
    )


def predict_rf(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained Random Forest model.

    Args:
        model: A fitted object exposing ``.predict``.
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


def train_et(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train an Extra Trees model with grid-searched hyper-parameters.

    Extra Trees uses random splits instead of optimal splits, which
    reduces variance at the cost of slightly higher bias.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    return _train_tree_ensemble(
        ExtraTreesRegressor, X_train, y_train, cv_folds, cv_strategy, random_state
    )


def predict_et(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained Extra Trees model.

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
    "train_rf",
    "predict_rf",
    "train_et",
    "predict_et",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_MAX_DEPTH",
]
