"""Gateway-tool regressions for labeled MATLAB matrix bundles."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from deerflow.community.nir.io_tools import nir_inspect_tool, nir_load_data_tool


def test_nir_inspect_guides_labeled_mat_directly_to_nir_load_data() -> None:
    inspected = {
        "format": "mat",
        "labeled_matrix": True,
        "auto_load_supported": True,
        "shape": [310, 404],
        "target_columns": [{"index": 0, "name": "active (%w/w)"}],
        "metadata_columns": [
            {"index": 1, "name": "Type"},
            {"index": 2, "name": "Scale"},
        ],
        "spectral_column_indices": list(range(3, 407)),
    }

    with (
        patch("deerflow.community.nir.io_tools._resolve", return_value="/tmp/tablets.MAT"),
        patch("nir_core.io.sniffers.inspect_file", return_value=json.dumps(inspected)),
    ):
        result = nir_inspect_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/NIRdata_tablets.MAT",
        )

    payload = json.loads(result)
    assert payload["layout_pattern"] == "labeled_matrix_bundle"
    assert payload["spectral_column_indices"] == {
        "start": 3,
        "stop": 407,
        "count": 404,
    }
    assert "Call nir_load_data directly" in payload["hint"]
    assert "without y_col, x_cols, wv_row, or Python scripts" in payload["hint"]


def test_nir_inspect_surfaces_mapping_clarification_instead_of_guessing() -> None:
    inspected = {
        "format": "mat",
        "shape": [10, 20],
        "schema_mapping": {
            "status": "needs_user_mapping",
            "confidence": 0.5,
            "clarification_required": True,
            "x_candidates": ["experiment.block_a", "experiment.block_b"],
            "y_candidates": ["experiment.reference"],
        },
    }

    with (
        patch("deerflow.community.nir.io_tools._resolve", return_value="/tmp/ambiguous.mat"),
        patch("nir_core.io.sniffers.inspect_file", return_value=json.dumps(inspected)),
    ):
        result = nir_inspect_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/ambiguous.mat",
        )

    payload = json.loads(result)
    assert payload["action_required"] == "confirm_field_mapping"
    assert "block_a" in payload["hint"]
    assert "Do not write a script" in payload["hint"]


def test_nir_load_data_refuses_ambiguous_mapping_without_override() -> None:
    inspected = {
        "format": "mat",
        "schema_mapping": {
            "status": "needs_user_mapping",
            "x_candidates": ["block_a", "block_b"],
            "y_candidates": ["response"],
        },
    }

    with (
        patch("deerflow.community.nir.io_tools._resolve", return_value="/tmp/ambiguous.mat"),
        patch("nir_core.io.sniffers.detect_format", return_value="mat"),
        patch(
            "nir_core.io.sniffers.inspect_file",
            return_value=json.dumps(inspected),
        ),
    ):
        result = nir_load_data_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/ambiguous.mat",
        )

    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "Field mapping confirmation required" in payload["error"]
    assert "block_a" in payload["error"]
