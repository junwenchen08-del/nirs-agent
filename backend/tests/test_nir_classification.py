"""Deployable qualitative NIR classification tool regressions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd


def _write_classification_csv(path: Path, *, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    class_names = np.asarray(["authentic", "adulterated", "substitute"])
    labels = np.repeat(class_names, 30)
    centers = {
        "authentic": np.asarray([1.4, 0.1, 0.1] * 4),
        "adulterated": np.asarray([0.1, 1.4, 0.1] * 4),
        "substitute": np.asarray([0.1, 0.1, 1.4] * 4),
    }
    X = np.vstack([rng.normal(centers[label], 0.08, size=12) for label in labels])
    frame = pd.DataFrame(X, columns=[str(value) for value in np.linspace(1000, 1550, 12)])
    frame.insert(0, "sample_group", [f"{label}-{index // 2}" for index, label in enumerate(labels)])
    frame.insert(0, "class", labels)
    frame.to_csv(path, index=False)
    return X, labels


def _write_columnwise_classification_csv(path: Path, *, seed: int = 23) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(["Arabica"] * 18 + ["Robusta"] * 18)
    wavelengths = np.linspace(810, 900, 16)
    arabica = rng.normal(np.linspace(1.0, 1.8, wavelengths.size), 0.03, size=(18, wavelengths.size))
    robusta = rng.normal(np.linspace(1.8, 1.0, wavelengths.size), 0.03, size=(18, wavelengths.size))
    X = np.vstack([arabica, robusta])
    rows: list[list[object]] = [
        ["Sample Number:", *[str(index + 1) for index in range(labels.size)]],
        ["Group Code:", *[str(index + 1) for index in range(labels.size)]],
        ["Wavenumbers", *labels.tolist()],
    ]
    for feature_index, wavelength in enumerate(wavelengths):
        rows.append([f"{wavelength:.3f}", *X[:, feature_index].tolist()])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return X, labels


def test_classification_train_and_predict_lifecycle(tmp_path: Path):
    from deerflow.community.nir.classification import nir_train_classifier_tool
    from deerflow.community.nir.io_tools import nir_predict_tool

    csv_path = tmp_path / "qualitative.csv"
    X, labels = _write_classification_csv(csv_path)
    model_path = tmp_path / "qualitative.pkl"
    metrics_path = tmp_path / "qualitative.json"
    report_path = tmp_path / "qualitative_report.md"
    prediction_input = tmp_path / "unknown.npz"
    prediction_output = tmp_path / "classified.csv"
    prediction_audit = tmp_path / "prediction-audit.jsonl"
    np.savez(prediction_input, X=X[:9], wv=np.linspace(1000, 1550, 12))

    virtual_csv = "/mnt/user-data/uploads/qualitative.csv"
    virtual_model = "/mnt/user-data/outputs/qualitative.pkl"
    virtual_metrics = "/mnt/user-data/outputs/qualitative.json"
    virtual_report = "/mnt/user-data/outputs/qualitative_report.md"
    virtual_prediction_input = "/mnt/user-data/uploads/unknown.npz"
    virtual_prediction_output = "/mnt/user-data/outputs/classified.csv"
    virtual_audit = "/mnt/user-data/outputs/prediction-audit.jsonl"
    resolved = {
        virtual_csv: str(csv_path),
        virtual_model: str(model_path),
        virtual_metrics: str(metrics_path),
        virtual_report: str(report_path),
        virtual_prediction_input: str(prediction_input),
        virtual_prediction_output: str(prediction_output),
        virtual_audit: str(prediction_audit),
    }

    with patch(
        "deerflow.community.nir.classification._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path=virtual_csv,
            label_col="class",
            group_col="sample_group",
            x_cols="2:",
            pipeline_steps='["snv", "autoscale"]',
            methods='["pls_da", "logistic", "linear_svm"]',
            tuning_ratio=0.2,
            test_ratio=0.2,
            random_state=9,
            model_output=virtual_model,
            metrics_output=virtual_metrics,
            report_output=virtual_report,
        )

    payload = json.loads(result)
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    artifact = joblib.load(model_path)
    assert payload["status"] == "ok"
    assert payload["task_kind"] == "classification"
    assert payload["classes"] == ["adulterated", "authentic", "substitute"]
    assert payload["passed"] is True
    assert payload["holdout"]["balanced_accuracy"] > 0.95
    assert persisted["scientific_validation"]["passed"] is True
    assert persisted["quality"]["passed"] is True
    assert artifact["format"] == "nir_model_artifact"
    assert artifact["version"] == 4
    assert artifact["task_kind"] == "classification"
    assert (tmp_path / "qualitative.pkl.manifest.json").is_file()

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        prediction = nir_predict_tool.func(
            runtime=MagicMock(state={}),
            model_path=virtual_model,
            data_path=virtual_prediction_input,
            output_path=virtual_prediction_output,
            detect_drift=True,
        )

    predicted = json.loads(prediction)
    assert predicted["status"] == "ok", predicted
    assert predicted["task_kind"] == "classification"
    assert predicted["n_samples"] == 9
    assert predicted["class_counts"] == {"authentic": 9}
    assert predicted["accepted_count"] + predicted["needs_review_count"] == 9
    assert predicted["accepted_count"] >= 8
    assert predicted["mean_confidence"] > 0.5
    with prediction_output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() == {
        "sample_index",
        "predicted_class",
        "confidence",
        "margin",
        "decision",
    }
    assert {row["predicted_class"] for row in rows} == {"authentic"}
    assert labels[:9].tolist() == ["authentic"] * 9


def test_classification_trains_directly_from_samples_in_columns_csv(tmp_path: Path):
    from deerflow.community.nir.classification import nir_train_classifier_tool

    csv_path = tmp_path / "coffee_raw.csv"
    _write_columnwise_classification_csv(csv_path)
    model_path = tmp_path / "coffee.pkl"
    metrics_path = tmp_path / "coffee.json"
    report_path = tmp_path / "coffee.md"
    virtual_csv = "/mnt/user-data/uploads/coffee_raw.csv"
    virtual_model = "/mnt/user-data/outputs/coffee.pkl"
    virtual_metrics = "/mnt/user-data/outputs/coffee.json"
    virtual_report = "/mnt/user-data/outputs/coffee.md"
    resolved = {
        virtual_csv: str(csv_path),
        virtual_model: str(model_path),
        virtual_metrics: str(metrics_path),
        virtual_report: str(report_path),
    }

    with patch(
        "deerflow.community.nir.classification._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path=virtual_csv,
            label_row=2,
            sample_cols="1:",
            wavenumber_col=0,
            pipeline_steps='["snv", "autoscale"]',
            methods='["pls_da", "logistic", "linear_svm"]',
            random_state=11,
            model_output=virtual_model,
            metrics_output=virtual_metrics,
            report_output=virtual_report,
        )

    payload = json.loads(result)
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok", payload
    assert payload["input_layout"] == "samples_in_columns"
    assert payload["layout_parameters"]["label_row"] == 2
    assert payload["classes"] == ["Arabica", "Robusta"]
    assert payload["holdout"]["balanced_accuracy"] > 0.95
    assert persisted["input_layout"] == "samples_in_columns"
    assert persisted["layout_parameters"]["sample_cols"] == "1:"
    assert persisted["scientific_validation"]["dataset"]["n_samples"] == 36
    assert "Input layout: `samples_in_columns`" in report_path.read_text(encoding="utf-8")


def test_classification_auto_infers_samples_in_columns_label_row(tmp_path: Path):
    from deerflow.community.nir.classification import nir_train_classifier_tool

    csv_path = tmp_path / "coffee_auto.csv"
    _write_columnwise_classification_csv(csv_path)
    model_path = tmp_path / "auto.pkl"
    metrics_path = tmp_path / "auto.json"
    report_path = tmp_path / "auto.md"
    virtual_csv = "/mnt/user-data/uploads/coffee_auto.csv"
    virtual_model = "/mnt/user-data/outputs/auto.pkl"
    virtual_metrics = "/mnt/user-data/outputs/auto.json"
    virtual_report = "/mnt/user-data/outputs/auto.md"
    resolved = {
        virtual_csv: str(csv_path),
        virtual_model: str(model_path),
        virtual_metrics: str(metrics_path),
        virtual_report: str(report_path),
    }

    with patch(
        "deerflow.community.nir.classification._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path=virtual_csv,
            sample_cols="1:",
            random_state=12,
            model_output=virtual_model,
            metrics_output=virtual_metrics,
            report_output=virtual_report,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok", payload
    assert payload["layout_parameters"]["label_row"] == 2
    assert payload["layout_parameters"]["label_row_inferred"] is True
    assert payload["holdout"]["macro_f1"] > 0.95


def test_nir_inspect_reports_bounded_class_label_candidates(tmp_path: Path) -> None:
    from deerflow.community.nir.io_tools import nir_inspect_tool

    csv_path = tmp_path / "labelled.csv"
    pd.DataFrame(
        {
            "origin": ["north", "south", "north", "west", "south", "north"],
            "batch": [1, 1, 2, 2, 3, 3],
            "1000": np.linspace(0.1, 0.6, 6),
            "1002": np.linspace(0.2, 0.7, 6),
        }
    ).to_csv(csv_path, index=False)
    inspected = {
        "format": "csv",
        "shape": [6, 4],
        "column_names": ["origin", "batch", "1000", "1002"],
        "non_numeric_columns": [0],
    }

    with (
        patch("deerflow.community.nir.io_tools._resolve", return_value=str(csv_path)),
        patch("nir_core.io.sniffers.inspect_file", return_value=json.dumps(inspected)),
    ):
        result = nir_inspect_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/labelled.csv",
        )

    payload = json.loads(result)
    candidates = {item["name"]: item for item in payload["categorical_columns"]}
    assert candidates["origin"]["class_counts"] == {"north": 3, "south": 2, "west": 1}
    assert candidates["origin"]["sampled_rows"] == 6
    assert candidates["batch"]["n_unique"] == 3
    assert "1000" not in candidates


def test_nir_inspect_reports_samples_in_columns_label_row_candidates(tmp_path: Path) -> None:
    from deerflow.community.nir.io_tools import nir_inspect_tool

    csv_path = tmp_path / "coffee_raw.csv"
    _write_columnwise_classification_csv(csv_path)
    inspected = {
        "format": "csv",
        "shape": [19, 37],
        "column_names": ["0", "1", "2"],
    }

    with (
        patch("deerflow.community.nir.io_tools._resolve", return_value=str(csv_path)),
        patch("nir_core.io.sniffers.inspect_file", return_value=json.dumps(inspected)),
    ):
        result = nir_inspect_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/coffee_raw.csv",
        )

    payload = json.loads(result)
    assert payload["classification_label_rows"][0]["row_index"] == 2
    assert payload["classification_label_rows"][0]["class_counts"] == {"Arabica": 18, "Robusta": 18}
    assert "label_row=<row_index>" in payload["classification_hint"]


def test_classifier_rejects_conflicting_duplicate_spectra(tmp_path: Path):
    from deerflow.community.nir.classification import nir_train_classifier_tool

    csv_path = tmp_path / "conflict.csv"
    X, _ = _write_classification_csv(csv_path)
    frame = pd.read_csv(csv_path)
    frame.loc[1, frame.columns[2:]] = X[0]
    frame.loc[1, "class"] = "adulterated"
    frame.to_csv(csv_path, index=False)
    virtual_csv = "/mnt/user-data/uploads/conflict.csv"

    with patch(
        "deerflow.community.nir.classification._resolve",
        return_value=str(csv_path),
    ):
        result = nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path=virtual_csv,
            label_col="class",
            x_cols="2:",
        )

    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "conflicting labels" in payload["error"]


def test_reflect_handles_failed_classification_metrics(tmp_path: Path) -> None:
    from deerflow.community.nir.reflect import nir_reflect_tool

    metrics_path = tmp_path / "classification_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "task_kind": "classification",
                "n_samples": 75,
                "preprocessing_steps": ["snv", "autoscale"],
                "test": {
                    "balanced_accuracy": 0.61,
                    "macro_f1": 0.58,
                    "per_class": {
                        "authentic": {"recall": 0.8},
                        "adulterated": {"recall": 0.42},
                    },
                },
                "quality": {
                    "grade": "D",
                    "passed": False,
                    "action": "improve_or_collect_data",
                    "thresholds_used": {"min_balanced_accuracy": 0.7},
                },
            }
        ),
        encoding="utf-8",
    )

    with patch("deerflow.community.nir.reflect._resolve", return_value=str(metrics_path)):
        result = nir_reflect_tool.func(
            runtime=MagicMock(),
            metrics_path="/mnt/user-data/outputs/classification_metrics.json",
            attempt=1,
            max_retries=3,
        )

    payload = json.loads(result)
    assert payload["task_kind"] == "classification"
    assert payload["should_retry"] is True
    assert payload["diagnostics"]["weakest_class"] == "adulterated"
    assert payload["fallback_suggestion_steps"]


def test_classifier_rejects_invalid_probability_threshold_before_file_io() -> None:
    from deerflow.community.nir.classification import nir_train_classifier_tool

    payload = json.loads(
        nir_train_classifier_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/missing.csv",
            label_col="class",
            min_confidence=1.2,
        )
    )

    assert payload["status"] == "error"
    assert "min_confidence" in payload["error"]
