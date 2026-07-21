"""End-to-end contracts for multi-component NIR tools."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np


def test_load_data_accepts_multiple_reference_columns(tmp_path: Path) -> None:
    from deerflow.community.nir.io_tools import nir_load_data_tool

    csv_file = tmp_path / "multi.csv"
    csv_file.write_text(
        "protein,moisture,oil,1100,1200\n10,12,4,0.1,0.2\n11,13,5,0.2,0.3\n",
        encoding="utf-8",
    )
    npz_file = tmp_path / "multi.npz"
    paths = {
        "/mnt/user-data/uploads/multi.csv": str(csv_file),
        "/mnt/user-data/workspace/multi.npz": str(npz_file),
    }

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        result = nir_load_data_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/multi.csv",
            output_path="/mnt/user-data/workspace/multi.npz",
            y_cols="0,1,2",
            x_cols="3:",
        )

    payload = json.loads(result)
    archive = np.load(npz_file, allow_pickle=True)
    assert "error" not in payload
    assert payload["n_components"] == 3
    assert payload["y_names"] == ["protein", "moisture", "oil"]
    assert payload["wv_separated"] is True
    assert payload["wv_first_values"] == [1100.0, 1200.0]
    assert archive["y"].shape == (2, 3)
    assert archive["wv"].tolist() == [1100.0, 1200.0]


def test_train_and_predict_multi_component_model(tmp_path: Path) -> None:
    from deerflow.community.nir.io_tools import nir_predict_tool
    from deerflow.community.nir.modeling import nir_train_multi_model_tool

    rng = np.random.RandomState(42)
    X = rng.rand(60, 8)
    y = np.column_stack(
        (
            X[:, 0] * 2.0 + X[:, 1],
            X[:, 2] * -1.5 + X[:, 3] * 0.5,
            X[:, 4] + X[:, 5] * 0.8,
        )
    )
    names = np.asarray(["protein", "moisture", "oil"], dtype=object)
    train_file = tmp_path / "train.npz"
    predict_file = tmp_path / "predict.npz"
    model_file = tmp_path / "multi-model.pkl"
    metrics_file = tmp_path / "multi-metrics.json"
    predictions_file = tmp_path / "predictions.csv"
    output_dir = tmp_path / "multi-output"
    np.savez(train_file, X=X, y=y, y_names=names)
    np.savez(predict_file, X=X[:5])

    paths = {
        "/mnt/user-data/uploads/train.npz": str(train_file),
        "/mnt/user-data/uploads/predict.npz": str(predict_file),
        "/mnt/user-data/outputs/multi-model.pkl": str(model_file),
        "/mnt/user-data/outputs/multi-metrics.json": str(metrics_file),
        "/mnt/user-data/outputs/multi": str(output_dir),
        "/mnt/user-data/outputs/predictions.csv": str(predictions_file),
    }

    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        train_result = nir_train_multi_model_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/train.npz",
            method="pls",
            max_components=2,
            cv_folds=2,
            cv_strategy="fixed",
            model_output="/mnt/user-data/outputs/multi-model.pkl",
            metrics_output="/mnt/user-data/outputs/multi-metrics.json",
            output_dir="/mnt/user-data/outputs/multi",
        )

    train_payload = json.loads(train_result)
    artifact = joblib.load(model_file)
    persisted_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert train_payload["status"] == "ok"
    assert train_payload["n_targets"] == 3
    assert [item["name"] for item in train_payload["per_component"]] == names.tolist()
    assert artifact["version"] == 3
    assert artifact["multi_output"] is True
    assert len(artifact["models"]) == 3
    assert len(artifact["wavelength_selection"]) == 3
    # Preprocessing artifact must always be a list[dict] of length n_targets,
    # regardless of shared/independent mode (Issue1/Issue2 regression guard).
    assert isinstance(artifact["preprocessing"], list)
    assert len(artifact["preprocessing"]) == 3
    assert all(pp["shared"] is True for pp in artifact["preprocessing"])
    assert persisted_metrics["overall"]["n_total"] == 3

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        prediction_result = nir_predict_tool.func(
            runtime=MagicMock(),
            model_path="/mnt/user-data/outputs/multi-model.pkl",
            data_path="/mnt/user-data/uploads/predict.npz",
            output_path="/mnt/user-data/outputs/predictions.csv",
            detect_drift=False,
        )

    prediction_payload = json.loads(prediction_result)
    assert prediction_payload["status"] == "ok"
    assert prediction_payload["n_targets"] == 3
    assert prediction_payload["component_names"] == names.tolist()
    assert len(prediction_payload["predictions_summary"]) == 3
    # Shared-preprocessing artifact (list format) must still report shared=True.
    assert prediction_payload["preprocessing"]["shared"] is True
    with predictions_file.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["sample_index", "protein", "moisture", "oil"]
    assert len(rows) == 6


def test_predict_multi_component_with_independent_preprocessing(tmp_path: Path) -> None:
    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir.io_tools import nir_predict_tool

    rng = np.random.RandomState(9)
    X_train = rng.rand(30, 4)
    X_predict = rng.rand(5, 4)
    pipelines = [
        PreprocessingPipeline([PreprocessingStep(method="mean_center")]).fit(X_train),
        PreprocessingPipeline([PreprocessingStep(method="autoscale")]).fit(X_train),
    ]
    coefficients = [np.array([1.0, -0.5, 0.2, 0.8]), np.array([-0.2, 0.4, 1.1, 0.3])]
    models = [LinearRegression().fit(pipeline.transform(X_train), pipeline.transform(X_train) @ coef) for pipeline, coef in zip(pipelines, coefficients)]
    artifact = {
        "format": "nir_model_artifact",
        "version": 3,
        "multi_output": True,
        "n_targets": 2,
        "component_names": ["protein", "moisture"],
        "models": models,
        "preprocessing": [
            {
                "shared": False,
                "description": pipeline.description(),
                "pipeline": pipeline,
                "apply_on_predict": True,
            }
            for pipeline in pipelines
        ],
        "wavelength_selection": [{"method": "none"}, {"method": "none"}],
    }
    model_file = tmp_path / "independent.pkl"
    data_file = tmp_path / "predict.npz"
    joblib.dump(artifact, model_file)
    np.savez(data_file, X=X_predict)
    paths = {
        "/mnt/user-data/outputs/independent.pkl": str(model_file),
        "/mnt/user-data/uploads/predict.npz": str(data_file),
    }

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        result = nir_predict_tool.func(
            runtime=MagicMock(),
            model_path="/mnt/user-data/outputs/independent.pkl",
            data_path="/mnt/user-data/uploads/predict.npz",
            detect_drift=False,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["preprocessing"] == {"shared": False, "applied": [True, True]}


def test_parse_multi_pipeline_rejects_invalid_with_english_message() -> None:
    """Invalid pipelines must raise an English ValueError (not Chinese),
    consistent with the rest of the module's error messages."""
    import pytest

    from deerflow.community.nir.modeling import _parse_multi_pipeline

    # "unknown_method" is not in the PRESTEP_METHODS whitelist.
    with pytest.raises(ValueError) as exc_info:
        _parse_multi_pipeline('["unknown_method"]')
    msg = str(exc_info.value)
    assert msg.startswith("Invalid pipeline:")
    assert "流水线非法" not in msg
