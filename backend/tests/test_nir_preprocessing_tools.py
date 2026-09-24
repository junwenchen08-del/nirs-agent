"""Regression tests for NIR preprocessing and wavelength-alignment tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


def _resolver(mapping):
    return lambda _runtime, path, *, read_only: str(mapping[path])


def test_nir_list_preprocessing_methods_returns_compact_runtime_catalog() -> None:
    from deerflow.community.nir.preprocess import nir_list_preprocessing_methods_tool

    payload = json.loads(
        nir_list_preprocessing_methods_tool.func(
            runtime=MagicMock(),
            auto_level="default",
        )
    )
    assert payload["status"] == "ok"
    assert payload["catalog_version"] == "2.0"
    assert len(payload["catalog_sha256"]) == 64
    assert payload["count"] > 0
    assert all(item["auto_level"] == "default" for item in payload["methods"])
    assert all("parameter_schema" not in item for item in payload["methods"])


def test_nir_describe_preprocessing_method_returns_parameters_and_providers() -> None:
    from deerflow.community.nir.preprocess import (
        nir_describe_preprocessing_method_tool,
    )

    payload = json.loads(
        nir_describe_preprocessing_method_tool.func(
            runtime=MagicMock(),
            method="sg_smooth",
        )
    )
    assert payload["status"] == "ok"
    assert payload["method"]["method_id"] == "sg_smooth"
    assert payload["method"]["parameter_schema"]["window"]["default"] == 11
    providers = {item["provider"]: item for item in payload["method"]["available_providers"]}
    assert payload["method"]["preferred_provider"] == "chemotools"
    assert providers["native"]["is_default"] is False
    assert providers["chemotools"]["is_default"] is True
    assert providers["chemotools"]["compatibility"] == "verified"


def test_nir_describe_preprocessing_method_has_stable_unknown_error() -> None:
    from deerflow.community.nir.preprocess import (
        nir_describe_preprocessing_method_tool,
    )

    payload = json.loads(
        nir_describe_preprocessing_method_tool.func(
            runtime=MagicMock(),
            method="invented_method",
        )
    )
    assert payload["status"] == "error"
    assert payload["code"] == "nir_preprocessing_method_unknown"
    assert "invented_method" in payload["error"]


def test_nir_recommend_preprocessing_returns_bounded_evidence(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_recommend_preprocessing_tool

    axis = np.linspace(-1.0, 1.0, 101)
    X = np.vstack([1.0 + np.exp(-(((axis - 0.2) / 0.2) ** 2)) + 2.0 * axis for _ in range(24)])
    source = tmp_path / "source.npz"
    np.savez(source, X=X, wv=np.linspace(900.0, 1800.0, X.shape[1]))
    mapping = {"/mnt/user-data/uploads/source.npz": source}

    with patch(
        "deerflow.community.nir.preprocess._resolve",
        side_effect=_resolver(mapping),
    ):
        result = nir_recommend_preprocessing_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/source.npz",
            budget="standard",
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["scope"] == "exploratory_input_only"
    assert payload["recommendation"]["candidates"][0]["candidate_id"] == "raw"
    assert len(payload["recommendation"]["candidates"]) <= 8
    assert "baseline_drift" in payload["recommendation"]["profile"]["tags"]
    assert "X" not in payload["recommendation"]


def test_nir_recommend_preprocessing_rejects_unknown_budget(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_recommend_preprocessing_tool

    source = tmp_path / "source.npz"
    np.savez(source, X=np.ones((4, 10)))
    mapping = {"/mnt/user-data/uploads/source.npz": source}
    with patch(
        "deerflow.community.nir.preprocess._resolve",
        side_effect=_resolver(mapping),
    ):
        result = nir_recommend_preprocessing_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/source.npz",
            budget="unbounded",
        )
    payload = json.loads(result)
    assert payload["status"] == "error"
    assert payload["code"] == "nir_preprocessing_recommendation_invalid"


def test_nir_preprocess_supports_new_methods(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_preprocess_tool

    wv = np.linspace(900.0, 1800.0, 101)
    reference = np.sin(wv / 150.0) + 2.0
    X = np.vstack([0.1 + scale * reference for scale in (0.8, 1.0, 1.2)])
    X[:, 50] += 8.0
    source = tmp_path / "source.npz"
    output = tmp_path / "processed.npz"
    np.savez(source, X=X, wv=wv)
    mapping = {
        "/mnt/user-data/uploads/source.npz": source,
        "/mnt/user-data/outputs/processed.npz": output,
    }
    steps = json.dumps(
        [
            {"method": "despike", "params": {"window": 5}},
            {"method": "emsc", "params": {"polynomial_order": 1}},
            {
                "method": "norris_derivative1",
                "params": {"gap": 3, "segment": 5},
            },
            "mean_center",
        ]
    )

    with patch(
        "deerflow.community.nir.preprocess._resolve",
        side_effect=_resolver(mapping),
    ):
        result = nir_preprocess_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/source.npz",
            output_path="/mnt/user-data/outputs/processed.npz",
            pipeline_steps=steps,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert [step["method"] for step in payload["pipeline"]] == [
        "despike",
        "emsc",
        "norris_derivative1",
        "mean_center",
    ]
    assert payload["catalog_version"] == "2.0"
    assert len(payload["catalog_sha256"]) == 64
    assert [item["provider"] for item in payload["providers"]] == [
        "native",
        "chemotools",
        "chemotools",
        "chemotools",
    ]
    with np.load(output, allow_pickle=False) as saved:
        assert saved["X"].shape == X.shape
        np.testing.assert_array_equal(saved["wv"], wv)
        assert np.isfinite(saved["X"]).all()


def test_nir_preprocess_reuses_pipeline_validation(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_preprocess_tool

    source = tmp_path / "source.npz"
    output = tmp_path / "processed.npz"
    np.savez(source, X=np.ones((3, 20)))
    mapping = {
        "/mnt/user-data/uploads/source.npz": source,
        "/mnt/user-data/outputs/processed.npz": output,
    }
    with patch(
        "deerflow.community.nir.preprocess._resolve",
        side_effect=_resolver(mapping),
    ):
        result = nir_preprocess_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/source.npz",
            output_path="/mnt/user-data/outputs/processed.npz",
            pipeline_steps='["mean_center", "robust_snv"]',
        )
    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "顺序不当" in payload["error"]
    assert not output.exists()


def test_nir_align_wavelengths_updates_matrix_and_axis(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_align_wavelengths_tool

    source_wv = np.linspace(900.0, 1800.0, 91)
    target_wv = np.linspace(950.0, 1750.0, 81)
    X = np.vstack([2.0 + 0.01 * source_wv, -1.0 + 0.02 * source_wv])
    source = tmp_path / "source.npz"
    reference = tmp_path / "reference.npz"
    output = tmp_path / "aligned.npz"
    np.savez(source, X=X, wv=source_wv, y=np.array([1.0, 2.0]))
    np.savez(reference, X=np.zeros((1, target_wv.size)), wv=target_wv)
    mapping = {
        "/mnt/user-data/uploads/source.npz": source,
        "/mnt/user-data/uploads/reference.npz": reference,
        "/mnt/user-data/outputs/aligned.npz": output,
    }

    with patch(
        "deerflow.community.nir.preprocess._resolve",
        side_effect=_resolver(mapping),
    ):
        result = nir_align_wavelengths_tool.func(
            runtime=MagicMock(),
            input_path="/mnt/user-data/uploads/source.npz",
            reference_path="/mnt/user-data/uploads/reference.npz",
            output_path="/mnt/user-data/outputs/aligned.npz",
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["output_shape"] == [2, 81]
    assert len(payload["source_axis_sha256"]) == 64
    assert len(payload["target_axis_sha256"]) == 64
    with np.load(output, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["wv"], target_wv)
        np.testing.assert_array_equal(saved["y"], np.array([1.0, 2.0]))
        np.testing.assert_allclose(
            saved["X"],
            np.vstack([2.0 + 0.01 * target_wv, -1.0 + 0.02 * target_wv]),
            atol=1e-12,
        )


def test_nir_align_wavelengths_requires_exactly_one_target(tmp_path: Path) -> None:
    from deerflow.community.nir.preprocess import nir_align_wavelengths_tool

    result = nir_align_wavelengths_tool.func(
        runtime=MagicMock(),
        input_path="/mnt/user-data/uploads/source.npz",
        output_path="/mnt/user-data/outputs/aligned.npz",
    )
    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "exactly one" in payload["error"]


def test_nir_predict_replays_fitted_emsc_pipeline(tmp_path: Path) -> None:
    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir._common import _write_trusted_model_artifact
    from deerflow.community.nir.io_tools import nir_predict_tool

    rng = np.random.default_rng(42)
    wv = np.linspace(900.0, 1800.0, 120)
    base = np.sin(wv / 130.0) + 2.0
    analyte = np.exp(-(((wv - 1450.0) / 45.0) ** 2))

    y_train = np.linspace(0.0, 1.0, 40)
    X_train = np.vstack([0.2 + scale * (base + y * analyte) + 0.02 * ((wv - wv.mean()) / wv.std()) + rng.normal(0.0, 1e-4, wv.size) for y, scale in zip(y_train, np.linspace(0.8, 1.2, y_train.size))])
    y_predict = np.array([0.15, 0.45, 0.85])
    X_predict = np.vstack([0.4 + scale * (base + y * analyte) for y, scale in zip(y_predict, (0.9, 1.1, 1.3))])
    pipeline = PreprocessingPipeline([PreprocessingStep(method="emsc", params={"polynomial_order": 1})]).fit(X_train, wv)
    model = LinearRegression().fit(pipeline.transform(X_train, wv), y_train)
    expected = model.predict(pipeline.transform(X_predict, wv))

    model_file = tmp_path / "emsc-model.pkl"
    data_file = tmp_path / "predict.npz"
    np.savez(data_file, X=X_predict, wv=wv)
    _write_trusted_model_artifact(
        {
            "format": "nir_model_artifact",
            "version": 3,
            "model": model,
            "preprocessing": {
                "description": pipeline.description(),
                "pipeline": pipeline,
                "apply_on_predict": True,
            },
            "wavelength_selection": {"method": "none"},
        },
        str(model_file),
    )
    resolved = {
        "/mnt/user-data/outputs/emsc-model.pkl": str(model_file),
        "/mnt/user-data/uploads/predict.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(tmp_path / "prediction-audit.jsonl"),
    }
    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=_resolver(resolved),
    ):
        result = nir_predict_tool.func(
            runtime=MagicMock(),
            model_path="/mnt/user-data/outputs/emsc-model.pkl",
            data_path="/mnt/user-data/uploads/predict.npz",
            detect_drift=False,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["preprocessing"]["applied"] is True
    assert np.isclose(payload["prediction_mean"], float(np.mean(expected)))
