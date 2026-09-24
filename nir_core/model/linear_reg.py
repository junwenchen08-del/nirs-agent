"""Regularized linear regression: Ridge, Lasso, ElasticNet.

All three methods use :class:`sklearn.linear_model` estimators wrapped in
a :class:`sklearn.pipeline.Pipeline` with :class:`StandardScaler` so the
returned model accepts raw spectra directly. Hyper-parameter ``alpha``
(regularization strength) is tuned via grid search cross-validation.

- **Ridge**: L2 penalty, shrinks coefficients but keeps all variables.
- **Lasso**: L1 penalty, produces sparse coefficients (variable selection).
- **ElasticNet**: combines L1 and L2 penalties via ``l1_ratio``.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nir_core.model.pls import _resolve_cv_splitter

# Default alpha search grid (log-spaced).
DEFAULT_ALPHA_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# Default l1_ratio grid for ElasticNet.
DEFAULT_L1_RATIO_GRID: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]


def _train_regularized_linear(
    estimator_cls: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
    extra_name: str = "",
) -> tuple[object, dict]:
    """Generic regularized linear trainer with grid search.

    Args:
        estimator_cls: ``Ridge``, ``Lasso``, or ``ElasticNet``.
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        param_grid: Parameter grid for GridSearchCV (keys prefixed with
            the estimator name in the pipeline).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy.
        random_state: Seed for reproducibility.
        extra_name: Extra key used in the pipeline step name (unused).

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

    step_name = "estimator"
    pipe = Pipeline([("scaler", StandardScaler()), (step_name, estimator_cls())])

    # Prefix param_grid keys with the step name.
    prefixed_grid = {f"{step_name}__{k}": v for k, v in param_grid.items()}

    splitter, strategy_label = _resolve_cv_splitter(
        cv_strategy, cv_folds, n_samples, random_state
    )

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=prefixed_grid,
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
        "param_grid": {k: list(v) for k, v in param_grid.items()},
    }
    return grid.best_estimator_, grid_results


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train a Ridge regression model with grid-searched alpha.

    The grid covers ``alpha in [0.001, 0.01, 0.1, 1, 10, 100]``.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    return _train_regularized_linear(
        Ridge,
        X_train,
        y_train,
        {"alpha": DEFAULT_ALPHA_GRID},
        cv_folds,
        cv_strategy,
        random_state,
    )


def predict_ridge(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained Ridge model.

    Args:
        model: A fitted Pipeline (StandardScaler -> Ridge).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


def train_lasso(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train a Lasso regression model with grid-searched alpha.

    Lasso uses an L1 penalty, producing sparse coefficients that can
    serve as a variable-selection mechanism.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    return _train_regularized_linear(
        Lasso,
        X_train,
        y_train,
        {"alpha": DEFAULT_ALPHA_GRID},
        cv_folds,
        cv_strategy,
        random_state,
    )


def predict_lasso(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained Lasso model.

    Args:
        model: A fitted Pipeline (StandardScaler -> Lasso).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


def train_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
) -> tuple[object, dict]:
    """Train an ElasticNet model with grid-searched alpha and l1_ratio.

    ElasticNet combines L1 (Lasso) and L2 (Ridge) penalties. The grid
    covers ``alpha in [0.001, ..., 100]`` and
    ``l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]``.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for the grid search.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.

    Returns:
        Tuple ``(best_model, grid_results)``.
    """
    return _train_regularized_linear(
        ElasticNet,
        X_train,
        y_train,
        {"alpha": DEFAULT_ALPHA_GRID, "l1_ratio": DEFAULT_L1_RATIO_GRID},
        cv_folds,
        cv_strategy,
        random_state,
    )


def predict_elasticnet(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained ElasticNet model.

    Args:
        model: A fitted Pipeline (StandardScaler -> ElasticNet).
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    X = np.asarray(X, dtype=float)
    pred = model.predict(X)
    return np.asarray(pred).ravel()


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_L1_RATIO_GRID",
    "predict_elasticnet",
    "predict_lasso",
    "predict_ridge",
    "train_elasticnet",
    "train_lasso",
    "train_ridge",
]
