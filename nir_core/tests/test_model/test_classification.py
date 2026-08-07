"""Deterministic qualitative/classification modeling regressions."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.model.classification import (
    PLSDAClassifier,
    classification_metrics,
    select_classifier,
    stratified_three_way_split,
)


def _three_class_data(seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray(["apple", "pear", "peach"]), 30)
    centers = {
        "apple": np.asarray([1.2, 0.0, 0.0] * 4),
        "pear": np.asarray([0.0, 1.2, 0.0] * 4),
        "peach": np.asarray([0.0, 0.0, 1.2] * 4),
    }
    X = np.vstack(
        [rng.normal(loc=centers[label], scale=0.18, size=12) for label in labels]
    )
    return X, labels


def test_pls_da_supports_string_multiclass_labels_and_probabilities():
    X, labels = _three_class_data()

    model = PLSDAClassifier(n_components=3).fit(X, labels)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert predictions.shape == labels.shape
    assert set(predictions) == set(labels)
    assert probabilities.shape == (labels.size, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.mean(predictions == labels) > 0.95


def test_stratified_three_way_split_preserves_classes_and_is_reproducible():
    _, labels = _three_class_data()

    first = stratified_three_way_split(
        labels,
        tuning_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )
    second = stratified_three_way_split(
        labels,
        tuning_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )

    assert first == second
    all_indices = first.calibration + first.tuning + first.holdout
    assert sorted(all_indices) == list(range(labels.size))
    assert len(all_indices) == len(set(all_indices))
    for indices in (first.calibration, first.tuning, first.holdout):
        assert set(labels[indices]) == {"apple", "pear", "peach"}


def test_grouped_classification_split_never_leaks_groups():
    _, labels = _three_class_data()
    groups = np.asarray([f"sample-{index // 3}" for index in range(labels.size)])

    split = stratified_three_way_split(
        labels,
        groups=groups,
        tuning_ratio=0.2,
        test_ratio=0.2,
        random_state=11,
    )

    group_sets = [
        set(groups[indices])
        for indices in (split.calibration, split.tuning, split.holdout)
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    for indices in (split.calibration, split.tuning, split.holdout):
        assert set(labels[indices]) == {"apple", "pear", "peach"}


def test_classification_metrics_report_per_class_specificity_and_confusion_matrix():
    labels = np.asarray(["a", "a", "b", "b", "c", "c"])
    predicted = np.asarray(["a", "a", "b", "c", "c", "c"])

    metrics = classification_metrics(labels, predicted, classes=["a", "b", "c"])

    assert metrics["accuracy"] == pytest.approx(5 / 6)
    assert metrics["balanced_accuracy"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert metrics["per_class"]["b"]["specificity"] == pytest.approx(1.0)


def test_classifier_selection_uses_tuning_only_and_refits_selected_family():
    X, labels = _three_class_data()
    split = stratified_three_way_split(
        labels, tuning_ratio=0.2, test_ratio=0.2, random_state=5
    )

    selected = select_classifier(
        X[split.calibration],
        labels[split.calibration],
        X[split.tuning],
        labels[split.tuning],
        methods=["pls_da", "logistic", "linear_svm"],
        random_state=5,
        max_components=5,
    )

    assert selected.method in {"pls_da", "logistic", "linear_svm"}
    assert len(selected.candidates) == 3
    assert all("balanced_accuracy" in candidate for candidate in selected.candidates)
    assert (
        np.mean(selected.model.predict(X[split.tuning]) == labels[split.tuning]) > 0.9
    )


def test_pls_da_selection_caps_components_to_inner_fold_size(monkeypatch):
    rng = np.random.default_rng(31)
    y_calibration = np.asarray(["a"] * 3 + ["b"] * 3)
    X_calibration = rng.normal(size=(6, 20))
    X_calibration[y_calibration == "b", :3] += 2.0
    y_tuning = np.asarray(["a", "b"])
    X_tuning = rng.normal(size=(2, 20))
    X_tuning[y_tuning == "b", :3] += 2.0
    attempted_components: list[int] = []
    original_fit = PLSDAClassifier.fit

    def recording_fit(self, X, y):
        attempted_components.append(int(self.n_components))
        return original_fit(self, X, y)

    monkeypatch.setattr(PLSDAClassifier, "fit", recording_fit)

    selected = select_classifier(
        X_calibration,
        y_calibration,
        X_tuning,
        y_tuning,
        methods=("pls_da",),
        max_components=10,
    )

    assert selected.method == "pls_da"
    assert max(attempted_components) <= 3


def test_classification_split_rejects_classes_too_small_for_stable_selection():
    labels = np.asarray(["a"] * 10 + ["b"] * 4)

    with pytest.raises(ValueError, match="at least 5 samples"):
        stratified_three_way_split(labels, tuning_ratio=0.2, test_ratio=0.2)
