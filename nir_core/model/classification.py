"""Leakage-safe supervised classification helpers for qualitative NIR analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


@dataclass(frozen=True)
class ClassificationSplit:
    """Indices for calibration, tuning, and one-time holdout evaluation."""

    calibration: list[int]
    tuning: list[int]
    holdout: list[int]
    strategy: str


@dataclass
class ClassificationSelection:
    """Selected classifier and bounded tuning evidence."""

    method: str
    model: object
    candidates: list[dict]


class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """PLS-DA implemented as PLS regression against one-hot class targets."""

    def __init__(self, n_components: int = 2):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_arr = np.asarray(X, dtype=float)
        labels = np.asarray(y).astype(str)
        if X_arr.ndim != 2 or labels.ndim != 1 or X_arr.shape[0] != labels.size:
            raise ValueError("PLS-DA requires a 2D X matrix and one label per row")
        self._label_encoder = LabelEncoder().fit(labels)
        encoded = self._label_encoder.transform(labels)
        self.classes_ = self._label_encoder.classes_
        if self.classes_.size < 2:
            raise ValueError("PLS-DA requires at least two classes")
        one_hot = np.eye(self.classes_.size, dtype=float)[encoded]
        safe_components = max(
            1,
            min(int(self.n_components), X_arr.shape[0] - 1, X_arr.shape[1]),
        )
        self.n_components_ = safe_components
        self._model = PLSRegression(n_components=safe_components, scale=False)
        self._model.fit(X_arr, one_hot)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(
            self._model.predict(np.asarray(X, dtype=float)), dtype=float
        )
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(scores, -50.0, 50.0))
        denominator = np.sum(exp_scores, axis=1, keepdims=True)
        return exp_scores / np.maximum(denominator, np.finfo(float).eps)

    def predict(self, X: np.ndarray) -> np.ndarray:
        encoded = np.argmax(self.decision_function(X), axis=1)
        return self._label_encoder.inverse_transform(encoded)


def _normalized_labels(labels: np.ndarray | Iterable[object]) -> np.ndarray:
    values = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels)
    if values.ndim != 1:
        raise ValueError("Classification labels must be one-dimensional")
    normalized = values.astype(str)
    if np.any(np.char.strip(normalized) == ""):
        raise ValueError("Classification labels cannot be empty")
    return normalized


def _require_partition_coverage(
    labels: np.ndarray, partitions: Iterable[np.ndarray]
) -> None:
    expected = set(np.unique(labels))
    for name, indices in zip(
        ("calibration", "tuning", "holdout"), partitions, strict=True
    ):
        if set(labels[indices]) != expected:
            raise ValueError(f"The {name} partition does not contain every class")


def _best_group_holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    expected = set(np.unique(labels[indices]))
    splitter = GroupShuffleSplit(
        n_splits=128, test_size=ratio, random_state=random_state
    )
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    overall = {label: float(np.mean(labels[indices] == label)) for label in expected}
    for train_local, test_local in splitter.split(
        indices, labels[indices], groups[indices]
    ):
        train_indices = indices[train_local]
        test_indices = indices[test_local]
        if (
            set(labels[train_indices]) != expected
            or set(labels[test_indices]) != expected
        ):
            continue
        size_error = abs((test_indices.size / indices.size) - ratio)
        balance_error = sum(
            abs(float(np.mean(labels[test_indices] == label)) - overall[label])
            for label in expected
        )
        score = size_error + balance_error
        if best is None or score < best[0]:
            best = (score, train_indices, test_indices)
    if best is None:
        raise ValueError(
            "Unable to create group-isolated partitions containing every class"
        )
    return np.sort(best[1]), np.sort(best[2])


def stratified_three_way_split(
    labels: np.ndarray | Iterable[object],
    *,
    tuning_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    groups: np.ndarray | Iterable[object] | None = None,
) -> ClassificationSplit:
    """Create deterministic class-stratified calibration/tuning/holdout indices."""

    y = _normalized_labels(labels)
    if (
        not 0 < tuning_ratio < 1
        or not 0 < test_ratio < 1
        or tuning_ratio + test_ratio >= 1
    ):
        raise ValueError(
            "tuning_ratio and test_ratio must be positive and sum to less than 1"
        )
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("Classification requires at least two classes")
    too_small = {
        str(label): int(count)
        for label, count in zip(classes, counts, strict=True)
        if count < 5
    }
    if too_small:
        raise ValueError(
            f"Every class needs at least 5 samples for stable three-way selection; too small: {too_small}"
        )

    indices = np.arange(y.size)
    if groups is None:
        remaining, holdout = train_test_split(
            indices,
            test_size=float(test_ratio),
            random_state=int(random_state),
            stratify=y,
        )
        relative_tuning = float(tuning_ratio) / (1.0 - float(test_ratio))
        calibration, tuning = train_test_split(
            remaining,
            test_size=relative_tuning,
            random_state=int(random_state) + 1,
            stratify=y[remaining],
        )
        strategy = "stratified"
    else:
        group_values = np.asarray(
            list(groups) if not isinstance(groups, np.ndarray) else groups
        ).astype(str)
        if group_values.ndim != 1 or group_values.size != y.size:
            raise ValueError("groups must be one-dimensional with one value per sample")
        calibration_and_tuning, holdout = _best_group_holdout(
            indices,
            y,
            group_values,
            ratio=float(test_ratio),
            random_state=int(random_state),
        )
        relative_tuning = float(tuning_ratio) / (1.0 - float(test_ratio))
        calibration, tuning = _best_group_holdout(
            calibration_and_tuning,
            y,
            group_values,
            ratio=relative_tuning,
            random_state=int(random_state) + 1,
        )
        strategy = "stratified_group"

    arrays = tuple(
        np.sort(np.asarray(part, dtype=int)) for part in (calibration, tuning, holdout)
    )
    _require_partition_coverage(y, arrays)
    return ClassificationSplit(
        calibration=arrays[0].tolist(),
        tuning=arrays[1].tolist(),
        holdout=arrays[2].tolist(),
        strategy=strategy,
    )


def classification_metrics(
    y_true: np.ndarray | Iterable[object],
    y_pred: np.ndarray | Iterable[object],
    *,
    classes: list[str] | np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> dict:
    """Compute aggregate and per-class metrics without hiding minority failures."""

    true = _normalized_labels(y_true)
    predicted = _normalized_labels(y_pred)
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    ordered_classes = [
        str(value) for value in (classes if classes is not None else np.unique(true))
    ]
    matrix = confusion_matrix(true, predicted, labels=ordered_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        predicted,
        labels=ordered_classes,
        zero_division=0,
    )
    per_class: dict[str, dict] = {}
    total = int(matrix.sum())
    for index, label in enumerate(ordered_classes):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = total - tp - fn - fp
        per_class[label] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
    result = {
        "n": int(true.size),
        "classes": ordered_classes,
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, predicted)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
    }
    if probabilities is not None:
        probability_array = np.asarray(probabilities, dtype=float)
        if probability_array.shape != (true.size, len(ordered_classes)):
            raise ValueError("probabilities must have shape (n_samples, n_classes)")
        encoder = LabelEncoder().fit(ordered_classes)
        encoded = encoder.transform(true)
        try:
            if len(ordered_classes) == 2:
                result["roc_auc"] = float(
                    roc_auc_score(encoded, probability_array[:, 1])
                )
            else:
                result["roc_auc_ovr_macro"] = float(
                    roc_auc_score(
                        encoded, probability_array, multi_class="ovr", average="macro"
                    )
                )
        except ValueError:
            pass
    return result


def build_classifier(
    method: str,
    *,
    random_state: int = 42,
    max_components: int = 10,
    n_components: int | None = None,
    calibration_cv: int = 3,
):
    """Build a bounded qualitative model family by stable method name."""

    normalized = str(method).strip().lower()
    if normalized == "pls_da":
        return PLSDAClassifier(n_components=n_components or max_components)
    if normalized == "logistic":
        return LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=int(random_state),
        )
    if normalized == "linear_svm":
        return CalibratedClassifierCV(
            SVC(
                kernel="linear",
                class_weight="balanced",
                random_state=int(random_state),
            ),
            method="sigmoid",
            cv=max(2, int(calibration_cv)),
            ensemble=False,
        )
    if normalized == "rbf_svm":
        return CalibratedClassifierCV(
            SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=int(random_state),
            ),
            method="sigmoid",
            cv=max(2, int(calibration_cv)),
            ensemble=False,
        )
    raise ValueError(f"Unsupported classification method {method!r}")


def _select_pls_components(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_components: int,
    random_state: int,
) -> int:
    counts = np.unique(labels, return_counts=True)[1]
    folds = max(2, min(5, int(counts.min())))
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=int(random_state)
    )
    fold_splits = list(splitter.split(X, labels))
    smallest_inner_train = min(train_idx.size for train_idx, _ in fold_splits)
    upper = max(
        1,
        min(
            int(max_components),
            X.shape[1],
            X.shape[0] - 1,
            smallest_inner_train - 1,
        ),
    )
    best = (float("-inf"), 1)
    for components in range(1, upper + 1):
        scores: list[float] = []
        for train_idx, val_idx in fold_splits:
            model = PLSDAClassifier(n_components=components).fit(
                X[train_idx], labels[train_idx]
            )
            scores.append(
                balanced_accuracy_score(labels[val_idx], model.predict(X[val_idx]))
            )
        candidate = (float(np.mean(scores)), -components)
        if candidate > (best[0], -best[1]):
            best = (candidate[0], components)
    return best[1]


def select_classifier(
    X_calibration: np.ndarray,
    y_calibration: np.ndarray | Iterable[object],
    X_tuning: np.ndarray,
    y_tuning: np.ndarray | Iterable[object],
    *,
    methods: Iterable[str] = ("pls_da", "logistic", "linear_svm"),
    random_state: int = 42,
    max_components: int = 10,
) -> ClassificationSelection:
    """Choose a model family using tuning balanced accuracy and macro-F1 only."""

    X_cal = np.asarray(X_calibration, dtype=float)
    X_tune = np.asarray(X_tuning, dtype=float)
    y_cal = _normalized_labels(y_calibration)
    y_tune = _normalized_labels(y_tuning)
    candidates: list[dict] = []
    fitted: dict[str, object] = {}
    normalized_methods = list(
        dict.fromkeys(str(method).strip().lower() for method in methods)
    )
    if not normalized_methods:
        raise ValueError("At least one classification method is required")
    for method in normalized_methods:
        n_components = None
        if method == "pls_da":
            n_components = _select_pls_components(
                X_cal,
                y_cal,
                max_components=int(max_components),
                random_state=int(random_state),
            )
        model = build_classifier(
            method,
            random_state=int(random_state),
            max_components=int(max_components),
            n_components=n_components,
            calibration_cv=max(
                2, min(3, int(np.unique(y_cal, return_counts=True)[1].min()))
            ),
        ).fit(X_cal, y_cal)
        prediction = np.asarray(model.predict(X_tune)).astype(str)
        metrics = classification_metrics(y_tune, prediction)
        candidates.append(
            {
                "method": method,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mcc": metrics["mcc"],
                **(
                    {"n_components": int(n_components)}
                    if n_components is not None
                    else {}
                ),
            }
        )
        fitted[method] = model
    chosen = max(
        candidates,
        key=lambda item: (
            float(item["balanced_accuracy"]),
            float(item["macro_f1"]),
            float(item["mcc"]),
            -normalized_methods.index(str(item["method"])),
        ),
    )
    method = str(chosen["method"])
    return ClassificationSelection(
        method=method, model=fitted[method], candidates=candidates
    )


__all__ = [
    "ClassificationSelection",
    "ClassificationSplit",
    "PLSDAClassifier",
    "build_classifier",
    "classification_metrics",
    "select_classifier",
    "stratified_three_way_split",
]
