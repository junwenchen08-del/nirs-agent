"""Core behaviour tests for NIR community tool helpers.

Covers:
- _parse_pipeline_step: parsing method-name strings and dicts with params
- nir_reflect: diagnostics extraction, fallback_suggestion naming
- _build_knowledge_hint: structured knowledge-base retrieval triggers
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from nir_core.models import PreprocessingStep

from deerflow.community.nir.tools import _build_knowledge_hint, _parse_pipeline_step


def test_pls_training_exposes_raw_space_intercept(tmp_path: Path):
    """The tool response and persisted metrics expose the complete PLS equation."""
    from deerflow.community.nir.tools import nir_train_model_tool

    rng = np.random.RandomState(42)
    X = rng.rand(60, 12)
    y = X @ rng.rand(12) + 2.5
    input_file = tmp_path / "data.npz"
    model_file = tmp_path / "model.pkl"
    metrics_file = tmp_path / "metrics.json"
    np.savez(input_file, X=X, y=y)

    virtual_input = "/mnt/user-data/uploads/data.npz"
    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_metrics = "/mnt/user-data/outputs/metrics.json"
    resolved = {
        virtual_input: str(input_file),
        virtual_model: str(model_file),
        virtual_metrics: str(metrics_file),
    }

    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_train_model_tool.func(
            runtime=MagicMock(),
            input_path=virtual_input,
            method="pls",
            max_components=2,
            cv_folds=2,
            cv_strategy="fixed",
            model_output=virtual_model,
            metrics_output=virtual_metrics,
        )

    payload = json.loads(result)
    persisted = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert isinstance(payload["coef_summary"]["intercept"], float)
    assert payload["coef_summary"]["intercept"] == persisted["coef_summary"]["intercept"]


def test_train_model_cars_selection_persists_artifact_metadata(tmp_path: Path):
    """CARS selection is fitted during training and stored with the model artifact."""
    import joblib

    from deerflow.community.nir.tools import nir_train_model_tool

    rng = np.random.RandomState(7)
    X = rng.rand(48, 18)
    y = X[:, 2] * 1.8 - X[:, 9] * 0.7 + rng.normal(scale=0.02, size=48)
    wv = np.linspace(900, 1700, X.shape[1])
    input_file = tmp_path / "data.npz"
    model_file = tmp_path / "model.pkl"
    metrics_file = tmp_path / "metrics.json"
    np.savez(input_file, X=X, y=y, wv=wv)

    virtual_input = "/mnt/user-data/uploads/data.npz"
    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_metrics = "/mnt/user-data/outputs/metrics.json"
    resolved = {
        virtual_input: str(input_file),
        virtual_model: str(model_file),
        virtual_metrics: str(metrics_file),
    }

    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_train_model_tool.func(
            runtime=MagicMock(),
            input_path=virtual_input,
            method="pls",
            max_components=2,
            cv_folds=2,
            cv_strategy="fixed",
            wavelength_selection="cars",
            wavelength_selection_params='{"n_mc_samples": 6, "n_folds": 2, "random_state": 11}',
            model_output=virtual_model,
            metrics_output=virtual_metrics,
        )

    payload = json.loads(result)
    persisted = json.loads(metrics_file.read_text(encoding="utf-8"))
    artifact = joblib.load(model_file)

    assert payload["status"] == "ok"
    assert payload["wavelength_selection"]["method"] == "cars"
    assert payload["wavelength_selection"]["n_selected"] < X.shape[1]
    assert persisted["wavelength_selection"] == payload["wavelength_selection"]
    assert artifact["format"] == "nir_model_artifact"
    assert artifact["wavelength_selection"] == payload["wavelength_selection"]


def test_nir_predict_applies_artifact_wavelength_selection(tmp_path: Path):
    """Prediction accepts a full-width X matrix and slices train-selected columns."""
    import joblib
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir.io_tools import nir_predict_tool

    rng = np.random.RandomState(13)
    X = rng.rand(20, 5)
    selected_indices = [1, 3]
    y = X[:, selected_indices] @ np.array([2.0, -1.0])
    model = LinearRegression().fit(X[:, selected_indices], y)

    model_file = tmp_path / "model.pkl"
    data_file = tmp_path / "predict.npz"
    np.savez(data_file, X=X)
    joblib.dump(
        {
            "format": "nir_model_artifact",
            "version": 1,
            "model": model,
            "wavelength_selection": {
                "method": "manual",
                "selected_indices": selected_indices,
                "n_original": X.shape[1],
                "n_selected": len(selected_indices),
            },
        },
        model_file,
    )

    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_data = "/mnt/user-data/uploads/predict.npz"
    resolved = {
        virtual_model: str(model_file),
        virtual_data: str(data_file),
    }

    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_predict_tool.func(
            runtime=MagicMock(),
            model_path=virtual_model,
            data_path=virtual_data,
            detect_drift=False,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["n_samples"] == X.shape[0]
    assert payload["wavelength_selection"]["selected_indices"] == selected_indices


def test_nir_predict_applies_fitted_artifact_preprocessing(tmp_path: Path):
    """Version-2 artifacts reproduce train-time preprocessing for raw spectra."""
    import joblib
    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir.io_tools import nir_predict_tool

    rng = np.random.RandomState(21)
    X_train = rng.rand(30, 4)
    X_predict = rng.rand(8, 4)
    pipeline = PreprocessingPipeline([PreprocessingStep(method="mean_center")]).fit(X_train)
    y_train = pipeline.transform(X_train) @ np.array([1.5, -0.5, 2.0, 0.25])
    model = LinearRegression().fit(pipeline.transform(X_train), y_train)
    expected = model.predict(pipeline.transform(X_predict))

    model_file = tmp_path / "model-v2.pkl"
    data_file = tmp_path / "raw-predict.npz"
    np.savez(data_file, X=X_predict)
    joblib.dump(
        {
            "format": "nir_model_artifact",
            "version": 2,
            "model": model,
            "preprocessing": {
                "description": pipeline.description(),
                "pipeline": pipeline,
                "apply_on_predict": True,
            },
            "wavelength_selection": {"method": "none"},
        },
        model_file,
    )

    resolved = {
        "/mnt/user-data/outputs/model-v2.pkl": str(model_file),
        "/mnt/user-data/uploads/raw-predict.npz": str(data_file),
    }
    with patch(
        "deerflow.community.nir.io_tools._resolve",
        side_effect=lambda _runtime, path, *, read_only: resolved[path],
    ):
        result = nir_predict_tool.func(
            runtime=MagicMock(),
            model_path="/mnt/user-data/outputs/model-v2.pkl",
            data_path="/mnt/user-data/uploads/raw-predict.npz",
            detect_drift=False,
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["preprocessing"]["applied"] is True
    assert np.isclose(payload["prediction_mean"], float(np.mean(expected)))


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

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
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

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
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

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
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

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
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

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
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


# ---------------------------------------------------------------------------
# _build_knowledge_hint: structured knowledge-base retrieval triggers
# ---------------------------------------------------------------------------
class TestBuildKnowledgeHint:
    """Tests for _build_knowledge_hint trigger logic."""

    def test_unknown_domain_triggers_hint(self):
        """A domain outside the known set triggers a hint with domain query."""
        hint = _build_knowledge_hint(domain="textile")
        assert hint is not None
        assert hint["should_search"] is True
        assert "textile" in hint["query"]
        assert "不在已知领域列表内" in hint["reason"]

    def test_known_domain_no_hint_when_grade_ok(self):
        """Known domain with passing grade returns no hint."""
        hint = _build_knowledge_hint(
            domain="soil",
            grade="A",
            passed=True,
            attempt=1,
            r2_val=0.85,
            diagnostics={"residual_trend": "none", "residual_variance": "low"},
        )
        assert hint is None

    def test_low_r2_triggers_hint(self):
        """R²_val < 0.7 triggers a domain-typical-range query."""
        hint = _build_knowledge_hint(
            domain="soil",
            grade="B",
            passed=True,
            attempt=1,
            r2_val=0.55,
            diagnostics={"residual_trend": "none", "residual_variance": "low"},
        )
        assert hint is not None
        assert hint["should_search"] is True
        assert "typical R2 RPD" in hint["query"]
        assert "soil" in hint["query"]
        assert "0.550" in hint["reason"]

    def test_grade_c_attempt_2_upward_trend_triggers_airpls_query(self):
        """grade=C, attempt>=2, upward trend → airPLS baseline query."""
        hint = _build_knowledge_hint(
            domain="food_protein",
            grade="C",
            passed=False,
            attempt=2,
            r2_val=0.72,
            diagnostics={"residual_trend": "upward", "residual_variance": "low"},
        )
        assert hint is not None
        assert "airpls baseline" in hint["query"]
        assert "food_protein" in hint["query"]
        assert "上升趋势" in hint["reason"]

    def test_grade_d_attempt_2_downward_trend_triggers_snv_msc_query(self):
        """grade=D, attempt>=2, downward trend → SNV vs MSC query."""
        hint = _build_knowledge_hint(
            domain="pharma",
            grade="D",
            passed=False,
            attempt=3,
            r2_val=0.72,
            diagnostics={"residual_trend": "downward", "residual_variance": "low"},
        )
        assert hint is not None
        assert "snv vs msc" in hint["query"]
        assert "pharma" in hint["query"]
        assert "下降趋势" in hint["reason"]

    def test_grade_f_attempt_2_high_variance_triggers_sg_smooth_query(self):
        """grade=F, attempt>=2, high variance → sg_smooth window query."""
        hint = _build_knowledge_hint(
            domain="feed",
            grade="F",
            passed=False,
            attempt=2,
            r2_val=0.72,
            diagnostics={"residual_trend": "none", "residual_variance": "high"},
        )
        assert hint is not None
        assert "sg_smooth window" in hint["query"]
        assert "feed" in hint["query"]
        assert "方差高" in hint["reason"]

    def test_grade_c_attempt_1_no_hint(self):
        """grade=C but attempt=1 → no hint yet (give one retry first)."""
        hint = _build_knowledge_hint(
            domain="soil",
            grade="C",
            passed=False,
            attempt=1,
            r2_val=0.72,
            diagnostics={"residual_trend": "upward", "residual_variance": "low"},
        )
        assert hint is None

    def test_grade_c_attempt_2_no_diag_falls_back_to_improve_query(self):
        """grade=C, attempt>=2, no specific diag signal → generic improve query."""
        hint = _build_knowledge_hint(
            domain="soil",
            grade="C",
            passed=False,
            attempt=2,
            r2_val=0.72,
            diagnostics={"residual_trend": "none", "residual_variance": "low"},
        )
        assert hint is not None
        assert "improve RPD" in hint["query"]

    def test_unknown_domain_takes_priority_over_low_r2(self):
        """Unknown domain should win over low R² (first match wins)."""
        hint = _build_knowledge_hint(
            domain="textile",
            grade="F",
            passed=False,
            attempt=3,
            r2_val=0.3,
            diagnostics={"residual_trend": "upward", "residual_variance": "high"},
        )
        assert hint is not None
        # Unknown-domain branch query, not the low-R² branch query
        assert "textile NIR calibration" in hint["query"]
        assert "不在已知领域列表内" in hint["reason"]

    def test_none_diagnostics_handled(self):
        """None diagnostics should not raise; low-R² path still triggers."""
        hint = _build_knowledge_hint(
            domain="soil",
            grade="B",
            passed=True,
            attempt=1,
            r2_val=0.5,
            diagnostics=None,
        )
        assert hint is not None
        assert "typical R2 RPD" in hint["query"]


# ---------------------------------------------------------------------------
# nir_reflect: knowledge_hint integration
# ---------------------------------------------------------------------------
class TestNirReflectKnowledgeHint:
    """Tests that nir_reflect surfaces knowledge_hint in its return value."""

    @staticmethod
    def _make_metrics_file(tmp_path: Path, r2_val: float = 0.65, with_diagnostics: bool = True) -> str:
        """Create a temporary metrics.json and return its path."""
        metrics = {
            "method": "pls",
            "n_components": 5,
            "domain": "default",
            "n_samples": 150,
            "preprocessing": "SNV",
            "preprocessing_steps": [{"method": "snv"}],
            "R2_val": r2_val,
            "RPD": 2.1,
            "RMSEP": 0.5,
            "RMSECV": 0.4,
            "test": {"RMSE": 0.5, "R2": 0.60, "RPD": 2.0, "bias": 0.01},
            "val": {"RMSE": 0.4, "R2": r2_val, "RPD": 2.1, "bias": 0.01},
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

    def test_knowledge_hint_present_when_low_r2(self, tmp_path):
        """nir_reflect returns a non-null knowledge_hint when R²_val < 0.7."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, r2_val=0.55, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert "knowledge_hint" in result_dict
        hint = result_dict["knowledge_hint"]
        assert hint is not None
        assert hint["should_search"] is True
        assert "typical R2 RPD" in hint["query"]

    def test_knowledge_hint_none_when_r2_ok(self, tmp_path):
        """nir_reflect returns null knowledge_hint when R² is acceptable."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, r2_val=0.92, with_diagnostics=False)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=1,
            )

        result_dict = json.loads(result)
        assert result_dict["knowledge_hint"] is None

    def test_knowledge_hint_for_unknown_domain(self, tmp_path):
        """nir_reflect returns a hint for an unknown domain even with ok R²."""
        from deerflow.community.nir.tools import nir_reflect_tool

        metrics_path = self._make_metrics_file(tmp_path, r2_val=0.92, with_diagnostics=False)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="textile",
                attempt=1,
            )

        result_dict = json.loads(result)
        hint = result_dict["knowledge_hint"]
        assert hint is not None
        assert "textile" in hint["query"]
        assert "不在已知领域列表内" in hint["reason"]

    def test_knowledge_hint_triggers_when_grade_c_attempt_2(self, tmp_path):
        """nir_reflect surfaces a hint when grade=C and attempt>=2."""
        from deerflow.community.nir.tools import nir_reflect_tool

        # R²=0.65 with upward trend → quality grade should be low.
        metrics_path = self._make_metrics_file(tmp_path, r2_val=0.65, with_diagnostics=True)
        mock_runtime = MagicMock()

        with patch("deerflow.community.nir.reflect._resolve", return_value=metrics_path):
            result = nir_reflect_tool.func(
                runtime=mock_runtime,
                metrics_path="/mnt/user-data/outputs/metrics.json",
                history="[]",
                domain="default",
                attempt=2,
            )

        result_dict = json.loads(result)
        hint = result_dict["knowledge_hint"]
        assert hint is not None
        # attempt=2 + grade low → either airpls branch (upward trend) or low-R² branch
        # The upward trend takes priority inside _build_knowledge_hint.
        assert hint["should_search"] is True
        assert "airpls" in hint["query"] or "typical R2 RPD" in hint["query"] or "improve RPD" in hint["query"]
