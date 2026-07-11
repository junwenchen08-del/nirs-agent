"""Core behaviour tests for NIR community tool helpers.

Covers:
- _parse_pipeline_step: parsing method-name strings and dicts with params
- nir_reflect: diagnostics extraction, fallback_suggestion naming
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nir_core.models import PreprocessingStep

from deerflow.community.nir.tools import _parse_pipeline_step


# ---------------------------------------------------------------------------
# _parse_pipeline_step
# ---------------------------------------------------------------------------
class TestParsePipelineStep:
    """Tests for _parse_pipeline_step."""

    def test_parse_plain_string(self):
        """A plain method-name string yields a step with empty params."""
        step = _parse_pipeline_step("snv")
        assert isinstance(step, PreprocessingStep)
        assert step.method == "snv"
        assert step.params == {}

    def test_parse_dict_with_method_only(self):
        """A dict with only 'method' yields a step with empty params."""
        step = _parse_pipeline_step({"method": "mean_center"})
        assert step.method == "mean_center"
        assert step.params == {}

    def test_parse_dict_with_params(self):
        """A dict with 'method' and 'params' preserves hyper-parameters."""
        step = _parse_pipeline_step({"method": "sg_smooth", "params": {"window": 15, "order": 2}})
        assert step.method == "sg_smooth"
        assert step.params == {"window": 15, "order": 2}

    def test_parse_dict_with_airpls_params(self):
        """airPLS lambda_ param is preserved through parsing."""
        step = _parse_pipeline_step({"method": "airpls", "params": {"lambda_": 1e6}})
        assert step.method == "airpls"
        assert step.params == {"lambda_": 1e6}

    def test_parse_string_with_numeric_method(self):
        """A numeric value is coerced to string method name."""
        step = _parse_pipeline_step(123)  # type: ignore[arg-type]
        assert step.method == "123"
        assert step.params == {}

    def test_parse_dict_extra_keys_ignored(self):
        """A dict with unexpected keys is handled gracefully by Pydantic."""
        step = _parse_pipeline_step({"method": "snv", "unknown_key": 42})
        assert step.method == "snv"


# ---------------------------------------------------------------------------
# nir_reflect: diagnostics + fallback_suggestion
# ---------------------------------------------------------------------------
class TestNirReflectDiagnostics:
    """Tests for nir_reflect tool's V3.6 diagnostics and fallback naming."""

    @staticmethod
    def _make_metrics_file(tmp_path: Path, with_diagnostics: bool = True) -> str:
        """Create a temporary metrics.json and return its path."""
        metrics = {
            "method": "pls",
            "n_components": 5,
            "domain": "default",
            "n_samples": 150,
            "preprocessing": "SNV",
            "preprocessing_steps": [{"method": "snv"}],
            "R2_val": 0.65,
            "RPD": 2.1,
            "RMSEP": 0.5,
            "RMSECV": 0.4,
            "test": {"RMSE": 0.5, "R2": 0.60, "RPD": 2.0, "bias": 0.01},
            "val": {"RMSE": 0.4, "R2": 0.65, "RPD": 2.1, "bias": 0.01},
            "train": {"RMSE": 0.3, "R2": 0.75, "RPD": 3.0, "bias": 0.0},
        }
        if with_diagnostics:
            metrics["diagnostics"] = {
                "residual_trend": "upward",
                "residual_variance": "high",
                "outlier_ratio": 0.08,
                "outlier_count": 12,
                "residual_std": 0.35,
            }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics, ensure_ascii=False), encoding="utf-8")
        return str(metrics_file)

    def test_diagnostics_returned_when_present(self, tmp_path):
        """nir_reflect returns diagnostics from metrics.json."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.tools._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert "diagnostics" in result_dict
        assert result_dict["diagnostics"]["residual_trend"] == "upward"
        assert result_dict["diagnostics"]["residual_variance"] == "high"

    def test_diagnostics_empty_when_absent(self, tmp_path):
        """nir_reflect returns empty diagnostics when not in metrics."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, with_diagnostics=False)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.tools._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert result_dict["diagnostics"] == {}

    def test_fallback_suggestion_key_present(self, tmp_path):
        """nir_reflect uses 'fallback_suggestion' not 'next_pipeline'."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.tools._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert "fallback_suggestion" in result_dict
        assert "fallback_suggestion_steps" in result_dict
        assert "next_pipeline" not in result_dict

    def test_reason_includes_diagnostics(self, tmp_path):
        """The reason text mentions diagnostics when present."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.tools._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert "残差诊断" in result_dict["reason"]
        assert "trend=upward" in result_dict["reason"]

    def test_best_so_far_included(self, tmp_path):
        """best_so_far is returned with current attempt's metrics."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.tools._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert result_dict["best_so_far"] is not None
        assert result_dict["best_so_far"]["RPD"] == 2.1
