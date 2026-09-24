"""1D Convolutional Neural Network (1D-CNN) regression.

Implements a compact 1D-CNN using PyTorch for NIR spectral regression.
The architecture consists of two convolutional layers (with ReLU +
MaxPool) followed by one fully-connected regression head.

PyTorch is an **optional** dependency. If ``torch`` is not installed, a
clear :class:`ImportError` is raised when :func:`train_cnn` is called.

The trained model is wrapped in a :class:`CNNWrapper` that mimics the
sklearn ``fit``/``predict`` interface so it can be used interchangeably
with other nir_core model trainers.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler

from nir_core.model.pls import _resolve_cv_splitter
from nir_core.utils.metrics import rmse

DEFAULT_EPOCHS: int = 200
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 0.001


def _check_torch():
    """Import torch or raise a helpful ImportError."""
    try:
        import torch
        from torch import nn

        return torch, nn
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for 1D-CNN models. Install the nir-core 'deep' extra and rebuild the Gateway runtime."
        ) from exc


def require_cnn_runtime() -> None:
    """Fail immediately when the optional PyTorch runtime is unavailable."""
    _check_torch()


class _CNNModel:
    """1D-CNN architecture for spectral regression.

    Architecture: Conv1d(1→16, k=5, pad=2) → ReLU → MaxPool(2)
                  Conv1d(16→32, k=5, pad=2) → ReLU → MaxPool(2)
                  Flatten → Linear(… → 64) → ReLU → Linear(64 → 1)
    """

    def __init__(self, n_wavelengths: int):
        _, nn = _check_torch()

        # Compute flattened size after two MaxPool(2) operations.
        n_after_pool = n_wavelengths // 4
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(32 * n_after_pool, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def parameters(self):
        return self.net.parameters()

    def forward(self, x):
        return self.net(x)

    def __call__(self, x):
        return self.forward(x)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)


class CNNWrapper:
    """Sklearn-compatible wrapper around the PyTorch 1D-CNN.

    Exposes ``fit`` and ``predict`` so the model can be used in the same
    pipeline as other nir_core trainers. Internal scaling is handled by
    a :class:`StandardScaler` fitted on the training data.

    The wrapper stores the trained network, scaler, and model
    hyper-parameters. It is serialisable via :mod:`joblib`; imported module
    objects are deliberately kept out of instance state.
    """

    def __init__(
        self,
        n_wavelengths: int,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        random_state: int = 42,
    ):
        _check_torch()
        self.n_wavelengths = n_wavelengths
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._scaler: StandardScaler | None = None
        self._model: _CNNModel | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch, _ = _check_torch()
        torch.manual_seed(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Standardise features.
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Build model.
        self._model = _CNNModel(self.n_wavelengths)

        # Convert to tensors: (N, 1, L) for Conv1d.
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()

        n_samples = X_t.shape[0]
        self._model.net.train()
        for epoch in range(self.epochs):
            # Shuffle indices each epoch.
            perm = torch.randperm(n_samples)
            for i in range(0, n_samples, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch, _ = _check_torch()
        X = np.asarray(X, dtype=float)
        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)

        self._model.net.eval()
        with torch.no_grad():
            pred = self._model(X_t)
        return pred.numpy().ravel()


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
    cv_strategy: str = "auto",
    random_state: int = 42,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tuple[object, dict]:
    """Train a 1D-CNN model for NIR spectral regression.

    Uses a compact two-layer 1D-CNN with Adam optimiser. Features are
    standardised via :class:`StandardScaler` before feeding to the network.
    A K-fold cross-validation is performed to estimate RMSECV; the final
    model is then retrained on the full training set.

    Requires PyTorch: ``pip install torch``.

    Args:
        X_train: Training spectra, shape (n_samples, n_wavelengths).
        y_train: Reference values, shape (n_samples,).
        cv_folds: Number of CV folds for RMSECV estimation.
        cv_strategy: CV strategy: ``"auto"`` / ``"loocv"`` / ``"fixed"``.
        random_state: Seed for reproducibility.
        epochs: Number of training epochs (default 200).
        batch_size: Mini-batch size (default 32).
        learning_rate: Adam learning rate (default 0.001).

    Returns:
        Tuple ``(best_model, cv_results)`` where ``best_model`` is a fitted
        :class:`CNNWrapper` and ``cv_results`` is a dict with keys
        ``"best_rmse"``, ``"mean_rmse_cv"``, ``"cv_strategy"``,
        ``"epochs"``, ``"batch_size"``, ``"learning_rate"``.

    Raises:
        ImportError: If PyTorch is not installed.
        ValueError: On shape mismatch.
    """
    # Check torch availability early.
    _check_torch()

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"X_train rows ({X_train.shape[0]}) != y_train length ({y_train.shape[0]})"
        )
    n_samples, n_wavelengths = X_train.shape

    # K-fold CV to estimate RMSECV.
    splitter, strategy_label = _resolve_cv_splitter(
        cv_strategy, cv_folds, n_samples, random_state
    )

    fold_rmse: list[float] = []
    for train_idx, val_idx in splitter.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        try:
            wrapper = CNNWrapper(
                n_wavelengths=n_wavelengths,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_state=random_state,
            )
            wrapper.fit(X_tr, y_tr)
            pred = wrapper.predict(X_val)
            fold_rmse.append(rmse(y_val, pred))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"CNN CV fold skipped after numerical/runtime failure: {exc}",
                stacklevel=2,
            )
            continue

    mean_rmse_cv = float(np.mean(fold_rmse)) if fold_rmse else float("inf")

    # Retrain on the full training set.
    best_model = CNNWrapper(
        n_wavelengths=n_wavelengths,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    best_model.fit(X_train, y_train)

    cv_results = {
        "best_rmse": mean_rmse_cv,
        "mean_rmse_cv": mean_rmse_cv,
        "cv_strategy": strategy_label,
        "n_folds": len(fold_rmse),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "Conv1d(1→16,k5)→ReLU→Pool(2)→Conv1d(16→32,k5)→ReLU→Pool(2)→FC(→64)→ReLU→FC(→1)",
    }
    return best_model, cv_results


def predict_cnn(model: object, X: np.ndarray) -> np.ndarray:
    """Predict reference values using a trained 1D-CNN wrapper.

    Args:
        model: A fitted :class:`CNNWrapper`.
        X: Spectra, shape (n_samples, n_wavelengths).

    Returns:
        1-D array of predictions, shape (n_samples,).
    """
    return np.asarray(model.predict(X), dtype=float).ravel()


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "CNNWrapper",
    "predict_cnn",
    "require_cnn_runtime",
    "train_cnn",
]
