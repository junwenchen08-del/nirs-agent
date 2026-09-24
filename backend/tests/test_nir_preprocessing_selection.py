"""End-to-end contracts for bounded automatic preprocessing selection."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np


def test_analyze_profiles_calibration_only_and_persists_selection(tmp_path: Path) -> None:
    from deerflow.community.nir.modeling import nir_analyze_tool

    rng = np.random.default_rng(20260924)
    n_samples = 60
    n_wavelengths = 24
    X = rng.normal(size=(n_samples, n_wavelengths))
    y = 2.0 + 1.4 * X[:, 3] - 0.8 * X[:, 11] + rng.normal(0.0, 0.04, n_samples)
    source = tmp_path / "source.npz"
    output = tmp_path / "analysis"
    output.mkdir()
    np.savez(source, X=X, y=y, wv=np.linspace(900.0, 1700.0, n_wavelengths))

    with (
        patch(
            "deerflow.community.nir.modeling._resolve",
            return_value=str(source),
        ),
        patch(
            "deerflow.community.nir.modeling._resolve_writable_dir",
            return_value=str(output),
        ),
    ):
        result = nir_analyze_tool.func(
            runtime=MagicMock(),
            data_path="/mnt/user-data/uploads/source.npz",
            method="pls",
            auto_preprocess=True,
            wavelength_selection="none",
            output_dir="/mnt/user-data/outputs/analysis",
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    selection = payload["preprocessing_selection"]
    assert selection["calibration_samples"] == 42
    assert selection["candidates"][0]["candidate_id"] == "raw"
    assert selection["candidate_count"] <= 8
    assert selection["selection_rule"] == "rmsecv_1pct"
    assert len(result) < 12_000

    artifact = joblib.load(output / "model.pkl")
    persisted = artifact["preprocessing"]["selection"]
    assert persisted["evaluation"]["best_candidate_id"] == selection["selected_candidate_id"]
    assert artifact["preprocessing"]["catalog_version"] == "2.0"
