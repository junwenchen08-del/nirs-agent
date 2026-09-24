"""End-to-end contracts for the separate calibration-transfer tool family."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


def _resolver(mapping):
    return lambda _runtime, path, *, read_only: str(mapping[path])


def _paired_files(
    tmp_path: Path,
    *,
    seed: int = 91,
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tmp_path.mkdir(parents=True, exist_ok=True)
    wv = np.linspace(1000.0, 1800.0, 31)
    latent = rng.normal(size=(48, 5))
    source = latent @ rng.normal(size=(5, wv.size))
    target = source * 1.25
    y = source[:, 0] + 0.5 * source[:, 1]
    ids = np.asarray([f"sample-{index:03d}" for index in range(source.shape[0])])
    order = rng.permutation(source.shape[0])
    source_path = tmp_path / "source.npz"
    target_path = tmp_path / "target.npz"
    np.savez(source_path, X=source, y=y, wv=wv, sample_names=ids)
    np.savez(
        target_path,
        X=target[order],
        y=y[order],
        wv=wv,
        sample_names=ids[order],
    )
    return source_path, target_path, source, target


def test_transfer_catalog_is_separate_from_preprocessing_catalog():
    from deerflow.community.nir.calibration_transfer import (
        nir_list_calibration_transfer_methods_tool,
    )

    payload = json.loads(nir_list_calibration_transfer_methods_tool.func(runtime=MagicMock()))
    assert payload["status"] == "ok"
    assert payload["catalog_version"] == "1.0"
    assert {item["method_id"] for item in payload["methods"]} == {
        "ds",
        "pds",
        "sst",
    }
    assert all(item["ordinary_preprocessing"] is False for item in payload["methods"])


def test_transfer_tools_are_exported_and_registered():
    from deerflow.community.nir import nir_fit_calibration_transfer_tool as package_tool
    from deerflow.community.nir.tools import (
        nir_fit_calibration_transfer_tool as facade_tool,
    )

    config_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    if not config_path.is_file():
        pytest.skip("Repository-root config.example.yaml is not mounted in this container.")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tools = {entry["name"]: entry["use"] for entry in config["tools"]}

    assert package_tool is facade_tool
    assert tools["nir_fit_calibration_transfer"] == ("deerflow.community.nir.calibration_transfer:nir_fit_calibration_transfer_tool")
    assert "nir_apply_calibration_transfer" in tools
    assert "nir_evaluate_calibration_transfer" in tools


def test_fit_apply_and_evaluate_transfer_with_integrity_manifest(tmp_path: Path):
    from deerflow.community.nir.calibration_transfer import (
        nir_apply_calibration_transfer_tool,
        nir_evaluate_calibration_transfer_tool,
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, source, _ = _paired_files(tmp_path / "train")
    artifact_path = tmp_path / "transfer.joblib"
    output_path = tmp_path / "transferred.npz"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
        "/mnt/user-data/outputs/transferred.npz": output_path,
    }

    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        fitted = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="reference-a",
                target_instrument_id="target-b",
                method="ds",
            )
        )
        applied = json.loads(
            nir_apply_calibration_transfer_tool.func(
                runtime=MagicMock(),
                artifact_path="/mnt/user-data/outputs/transfer.joblib",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transferred.npz",
                target_instrument_id="target-b",
            )
        )
        evaluated = json.loads(
            nir_evaluate_calibration_transfer_tool.func(
                runtime=MagicMock(),
                artifact_path="/mnt/user-data/outputs/transfer.joblib",
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
            )
        )

    assert fitted["status"] == "ok"
    assert fitted["validation_status"] == "spectrally_validated"
    assert fitted["validation"]["scope"] == "internal_paired_holdout"
    assert fitted["validation"]["spectral"]["improvement_percent"] > 90.0
    assert fitted["manifest"]["direction"] == "target-b_to_reference-a"
    assert artifact_path.with_name("transfer.joblib.manifest.json").is_file()
    assert applied["status"] == "ok"
    assert applied["production_qualification"] == "requires_reference_model_validation"
    assert evaluated["status"] == "ok"
    assert evaluated["spectral"]["after"]["rmse"] < evaluated["spectral"]["before"]["rmse"]
    with np.load(output_path, allow_pickle=False) as output:
        assert output["X"].shape == source.shape
        expected_order = [int(value.rsplit("-", 1)[1]) for value in output["sample_names"]]
        np.testing.assert_allclose(
            output["X"],
            source[expected_order],
            rtol=1e-9,
            atol=1e-9,
        )


def test_fit_requires_explicit_unique_pairing_evidence(tmp_path: Path):
    from deerflow.community.nir.calibration_transfer import (
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, _, _ = _paired_files(tmp_path)
    with np.load(target_path, allow_pickle=False) as archive:
        np.savez(target_path, X=archive["X"], wv=archive["wv"])
    artifact_path = tmp_path / "transfer.joblib"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
    }
    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        payload = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="source",
                target_instrument_id="target",
                method="ds",
            )
        )

    assert payload["status"] == "error"
    assert payload["code"] == "nir_calibration_transfer_pairing_invalid"
    assert not artifact_path.exists()


def test_reference_model_validation_promotes_transfer_to_production(tmp_path: Path):
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir._common import _write_trusted_model_artifact
    from deerflow.community.nir.calibration_transfer import (
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, source, _ = _paired_files(tmp_path / "train")
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir(parents=True)
    rng = np.random.default_rng(92)
    validation_source = rng.normal(size=(20, source.shape[0])) @ source
    validation_target = validation_source * 1.25
    validation_y = validation_source[:, 0] + 0.5 * validation_source[:, 1]
    validation_ids = np.asarray([f"validation-{index:03d}" for index in range(validation_source.shape[0])])
    validation_order = rng.permutation(validation_source.shape[0])
    validation_source_path = validation_dir / "source.npz"
    validation_target_path = validation_dir / "target.npz"
    validation_wv = np.linspace(1000.0, 1800.0, source.shape[1])
    np.savez(
        validation_source_path,
        X=validation_source,
        y=validation_y,
        wv=validation_wv,
        sample_names=validation_ids,
    )
    np.savez(
        validation_target_path,
        X=validation_target[validation_order],
        y=validation_y[validation_order],
        wv=validation_wv,
        sample_names=validation_ids[validation_order],
    )
    with np.load(source_path, allow_pickle=False) as archive:
        y = archive["y"]
    model_path = tmp_path / "reference-model.joblib"
    _write_trusted_model_artifact(
        {
            "format": "nir_model_artifact",
            "version": 3,
            "model": LinearRegression().fit(source, y),
            "preprocessing": {},
            "wavelength_selection": {},
        },
        str(model_path),
    )
    artifact_path = tmp_path / "transfer.joblib"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/uploads/validation-source.npz": validation_source_path,
        "/mnt/user-data/uploads/validation-target.npz": validation_target_path,
        "/mnt/user-data/outputs/reference-model.joblib": model_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
    }
    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        payload = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="reference-a",
                target_instrument_id="target-b",
                method="ds",
                reference_model_path=("/mnt/user-data/outputs/reference-model.joblib"),
                reference_model_id="reference-model-v1",
                validation_source_path=("/mnt/user-data/uploads/validation-source.npz"),
                validation_target_path=("/mnt/user-data/uploads/validation-target.npz"),
            )
        )

    assert payload["status"] == "ok"
    assert payload["validation_status"] == "production_validated"
    assert payload["production_qualification"] == "approved"
    assert payload["validation"]["scope"] == "independent_paired_validation"
    prediction = payload["validation"]["prediction"]
    assert prediction["after"]["rmsep"] < prediction["before"]["rmsep"]
    assert prediction["improvement_percent"] > 90.0
    assert payload["reference_model_binding"]["model_id"] == "reference-model-v1"
    assert len(payload["reference_model_binding"]["artifact_sha256"]) == 64


def test_internal_model_validation_does_not_claim_production_approval(tmp_path: Path):
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir._common import _write_trusted_model_artifact
    from deerflow.community.nir.calibration_transfer import (
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, source, _ = _paired_files(tmp_path / "train")
    with np.load(source_path, allow_pickle=False) as archive:
        y = archive["y"]
    model_path = tmp_path / "reference-model.joblib"
    _write_trusted_model_artifact(
        {
            "format": "nir_model_artifact",
            "version": 3,
            "model": LinearRegression().fit(source, y),
            "preprocessing": {},
            "wavelength_selection": {},
        },
        str(model_path),
    )
    artifact_path = tmp_path / "transfer.joblib"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/outputs/reference-model.joblib": model_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
    }
    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        payload = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="reference-a",
                target_instrument_id="target-b",
                method="ds",
                reference_model_path=("/mnt/user-data/outputs/reference-model.joblib"),
                reference_model_id="reference-model-v1",
            )
        )

    assert payload["status"] == "ok"
    assert payload["validation_status"] == "model_validated_internal"
    assert payload["production_qualification"] == "requires_independent_validation"
    assert payload["validation"]["scope"] == "internal_paired_holdout"


def test_independent_validation_paths_must_be_provided_as_a_pair(tmp_path: Path):
    from deerflow.community.nir.calibration_transfer import (
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, _, _ = _paired_files(tmp_path / "train")
    artifact_path = tmp_path / "transfer.joblib"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
    }
    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        payload = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="reference-a",
                target_instrument_id="target-b",
                validation_source_path=("/mnt/user-data/uploads/validation-source.npz"),
            )
        )

    assert payload["status"] == "error"
    assert payload["code"] == "nir_calibration_transfer_invalid"
    assert "provided together" in payload["error"]
    assert not artifact_path.exists()


def test_transfer_artifact_tampering_is_rejected_before_joblib_load(tmp_path: Path):
    from deerflow.community.nir._common import _load_trusted_transfer_artifact
    from deerflow.community.nir.calibration_transfer import (
        nir_fit_calibration_transfer_tool,
    )

    source_path, target_path, _, _ = _paired_files(tmp_path)
    artifact_path = tmp_path / "transfer.joblib"
    mapping = {
        "/mnt/user-data/uploads/source.npz": source_path,
        "/mnt/user-data/uploads/target.npz": target_path,
        "/mnt/user-data/outputs/transfer.joblib": artifact_path,
    }
    with patch(
        "deerflow.community.nir.calibration_transfer._resolve",
        side_effect=_resolver(mapping),
    ):
        payload = json.loads(
            nir_fit_calibration_transfer_tool.func(
                runtime=MagicMock(),
                source_path="/mnt/user-data/uploads/source.npz",
                target_path="/mnt/user-data/uploads/target.npz",
                output_path="/mnt/user-data/outputs/transfer.joblib",
                source_instrument_id="source",
                target_instrument_id="target",
                method="ds",
            )
        )
    assert payload["status"] == "ok"
    with artifact_path.open("ab") as handle:
        handle.write(b"tamper")

    with patch("joblib.load") as unsafe_load:
        try:
            _load_trusted_transfer_artifact(
                str(artifact_path),
                "/mnt/user-data/outputs/transfer.joblib",
            )
        except ValueError as exc:
            assert "integrity" in str(exc).lower()
        else:  # pragma: no cover - safety assertion
            raise AssertionError("Tampered transfer artifact was accepted")
        unsafe_load.assert_not_called()
