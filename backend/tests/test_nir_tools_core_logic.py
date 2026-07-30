"""Core behaviour tests for NIR community tool helpers.

Covers:
- _parse_pipeline_step: parsing method-name strings and dicts with params
- nir_reflect: diagnostics extraction, fallback_suggestion naming
- _build_knowledge_hint: structured knowledge-base retrieval triggers
"""

import json
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from nir_core.models import PreprocessingStep

from deerflow.community.nir.tools import _build_knowledge_hint, _parse_pipeline_step


def test_backend_default_nir_dependency_includes_deep_runtime():
    """Every model advertised by the default Gateway must be runnable there."""
    backend_pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    config = tomllib.loads(backend_pyproject.read_text(encoding="utf-8"))
    dependencies = config["project"]["dependencies"]

    assert "nir-core[deep,mat73]" in dependencies
    assert config["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    } in config["tool"]["uv"]["index"]
    nir_core_pyproject = backend_pyproject.parents[1] / "nir_core" / "pyproject.toml"
    nir_core_config = tomllib.loads(nir_core_pyproject.read_text(encoding="utf-8"))
    assert nir_core_config["tool"]["uv"]["sources"]["torch"] == {"index": "pytorch-cpu"}
    assert config["tool"]["uv"]["index"] == nir_core_config["tool"]["uv"]["index"]


def test_cnn_analyze_checks_runtime_before_resolving_or_loading_data():
    """A missing CNN runtime must fail before file IO and preprocessing."""
    from deerflow.community.nir.tools import nir_analyze_tool

    with (
        patch(
            "nir_core.model.cnn.require_cnn_runtime",
            side_effect=ImportError("torch unavailable"),
        ),
        patch(
            "deerflow.community.nir.modeling._resolve",
            side_effect=AssertionError("data path must not be resolved"),
        ),
    ):
        result = nir_analyze_tool.func(
            runtime=MagicMock(),
            data_path="/mnt/user-data/uploads/data.mat",
            method="cnn",
        )

    payload = json.loads(result)
    assert payload["status"] == "error"
    assert payload["code"] == "nir_model_runtime_unavailable"
    assert payload["details"] == {
        "requested_method": "cnn",
        "required_dependency": "torch",
        "action_required": "rebuild_gateway_with_deep_runtime",
        "substitution_requires_user_approval": True,
    }


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
    assert artifact["monitoring_reference"]["method"] == "pca_t2_q"


def test_partitioned_model_uses_named_external_split(tmp_path: Path):
    """Named partitions keep the external set out of all tuning decisions."""
    import joblib
    import pandas as pd

    from deerflow.community.nir.tools import nir_train_partitioned_model_tool

    rng = np.random.RandomState(24)
    n_wavelengths = 21
    labels = np.array(["Cal"] * 18 + ["Tuning"] * 9 + ["Val Ext"] * 9)
    X = rng.normal(size=(labels.size, n_wavelengths))
    y = 3.0 + X[:, 3] * 1.5 - X[:, 11] * 0.8 + rng.normal(scale=0.04, size=labels.size)
    frame = pd.DataFrame(X, columns=[str(900 + 3 * index) for index in range(n_wavelengths)])
    frame.insert(0, "DM", y)
    frame.insert(0, "Set", labels)
    source = tmp_path / "partitioned.csv"
    frame.to_csv(source, index=False)
    model_file = tmp_path / "partitioned.pkl"
    metrics_file = tmp_path / "partitioned.json"

    paths = {
        "/mnt/user-data/uploads/partitioned.csv": str(source),
        "/mnt/user-data/outputs/partitioned.pkl": str(model_file),
        "/mnt/user-data/outputs/partitioned.json": str(metrics_file),
    }
    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        result = nir_train_partitioned_model_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/partitioned.csv",
            split_col="Set",
            train_label="Cal",
            tuning_label="Tuning",
            test_label="Val Ext",
            y_col=1,
            x_cols="2:",
            pipeline_steps='["snv", {"method": "derivative1", "params": {"window": 5, "order": 2}}, "autoscale"]',
            max_components=3,
            compare_cars=False,
            model_output="/mnt/user-data/outputs/partitioned.pkl",
            metrics_output="/mnt/user-data/outputs/partitioned.json",
        )

    payload = json.loads(result)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    manifest = json.loads(Path(str(model_file) + ".manifest.json").read_text(encoding="utf-8"))
    artifact = joblib.load(model_file)
    assert payload["status"] == "ok"
    assert payload["protocol"] == "named_partition_external_validation"
    assert payload["validation_scope"] == "independent_external_validation"
    assert metrics["protocol"] == payload["protocol"]
    assert metrics["validation_scope"] == payload["validation_scope"]
    assert payload["evidence"]["model_sha256"] == manifest["sha256"]
    assert payload["evidence"]["metrics_sha256"] == manifest["metrics_sha256"]
    assert payload["evidence"]["training_data_sha256"] == metrics["training_data_hash"]
    assert isinstance(payload["passed"], bool)
    assert metrics["partitions"]["train"]["n_samples"] == 18
    assert metrics["partitions"]["tuning"]["n_samples"] == 9
    assert metrics["partitions"]["external_test"]["n_samples"] == 9
    assert artifact["format"] == "nir_model_artifact"
    assert (tmp_path / "partitioned_report.md").is_file()


def test_auto_split_model_persists_deterministic_holdout_protocol(tmp_path: Path):
    """Unpartitioned CSV data receives a reproducible, leakage-safe 70/15/15 split."""
    import joblib
    import pandas as pd

    from deerflow.community.nir.tools import nir_train_auto_split_model_tool

    rng = np.random.RandomState(31)
    n_samples = 80
    n_wavelengths = 19
    X = rng.normal(size=(n_samples, n_wavelengths))
    y = 1.7 + X[:, 4] * 1.2 - X[:, 13] * 0.5 + rng.normal(scale=0.03, size=n_samples)
    frame = pd.DataFrame(X, columns=[str(1000 + 4 * index) for index in range(n_wavelengths)])
    frame.insert(0, "Protein", y)
    source = tmp_path / "unpartitioned.csv"
    frame.to_csv(source, index=False)
    model_file = tmp_path / "auto_split.pkl"
    metrics_file = tmp_path / "auto_split.json"

    paths = {
        "/mnt/user-data/uploads/unpartitioned.csv": str(source),
        "/mnt/user-data/outputs/auto_split.pkl": str(model_file),
        "/mnt/user-data/outputs/auto_split.json": str(metrics_file),
    }
    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        result = nir_train_auto_split_model_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/unpartitioned.csv",
            y_col=0,
            x_cols="1:",
            pipeline_steps='["snv", "autoscale"]',
            max_components=3,
            compare_cars=False,
            model_output="/mnt/user-data/outputs/auto_split.pkl",
            metrics_output="/mnt/user-data/outputs/auto_split.json",
        )

    payload = json.loads(result)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    manifest = json.loads(Path(str(model_file) + ".manifest.json").read_text(encoding="utf-8"))
    artifact = joblib.load(model_file)
    partitions = metrics["partitions"]
    split_indices = [set(partitions[name]["sample_indices"]) for name in ("calibration", "tuning", "holdout_test")]

    assert payload["status"] == "ok"
    assert payload["protocol"] == "deterministic_auto_split_holdout"
    assert payload["target"] == "Protein"
    assert payload["validation_scope"] == "independent_holdout_not_external"
    assert payload["evidence"]["model_sha256"] == manifest["sha256"]
    assert payload["evidence"]["metrics_sha256"] == manifest["metrics_sha256"]
    assert payload["evidence"]["training_data_sha256"] == metrics["training_data_hash"]
    assert [partitions[name]["n_samples"] for name in ("calibration", "tuning", "holdout_test")] == [56, 12, 12]
    assert not (split_indices[0] & split_indices[1] or split_indices[0] & split_indices[2] or split_indices[1] & split_indices[2])
    assert set.union(*split_indices) == set(range(n_samples))
    assert metrics["split"]["strategy"] == "spxy"
    assert metrics["split"]["requested_strategy"] == "auto"
    assert metrics["split"]["random_state"] == 42
    assert metrics["wavelength_selection_decision"]["mode"] == "disabled"
    assert payload["wavelength_selection_decision"]["evaluate_cars"] is False
    assert artifact["format"] == "nir_model_artifact"
    assert (tmp_path / "auto_split_report.md").is_file()


def test_auto_split_default_autonomously_evaluates_cars(tmp_path: Path):
    """A generic request needs no explicit compare_cars flag for wide data."""
    import pandas as pd

    from deerflow.community.nir.tools import nir_train_auto_split_model_tool

    rng = np.random.RandomState(131)
    X = rng.normal(size=(70, 60))
    y = 2.0 + X[:, 3] * 1.4 - X[:, 17] * 0.6 + rng.normal(scale=0.03, size=70)
    frame = pd.DataFrame(X, columns=[str(900 + 3 * index) for index in range(X.shape[1])])
    frame.insert(0, "Protein", y)
    source = tmp_path / "autonomous_selection.csv"
    frame.to_csv(source, index=False)
    model_file = tmp_path / "autonomous_selection.pkl"
    metrics_file = tmp_path / "autonomous_selection.json"
    paths = {
        "/mnt/user-data/uploads/autonomous_selection.csv": str(source),
        "/mnt/user-data/outputs/autonomous_selection.pkl": str(model_file),
        "/mnt/user-data/outputs/autonomous_selection.json": str(metrics_file),
    }

    with (
        patch(
            "deerflow.community.nir.modeling._resolve",
            side_effect=lambda _runtime, path, *, read_only: paths[path],
        ),
        patch(
            "nir_core.model.selection.cars_wavelength_selection",
            return_value=(X[:, [3, 17]], [3, 17]),
        ),
    ):
        result = nir_train_auto_split_model_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/autonomous_selection.csv",
            y_col=0,
            x_cols="1:",
            pipeline_steps='["mean_center"]',
            max_components=2,
            model_output="/mnt/user-data/outputs/autonomous_selection.pkl",
            metrics_output="/mnt/user-data/outputs/autonomous_selection.json",
        )

    payload = json.loads(result)
    assert payload["status"] == "ok"
    assert payload["wavelength_selection_decision"]["mode"] == "auto"
    assert payload["wavelength_selection_decision"]["evaluate_cars"] is True
    assert payload["model_selection_decision"]["mode"] == "auto"
    assert payload["model_selection_decision"]["selected_method"] in {item["method"] for item in payload["model_candidates"]}
    assert {item["method"] for item in payload["candidate_results"]} == {
        "none",
        "cars",
    }


def test_one_shot_analysis_defaults_to_autonomous_wavelength_selection(tmp_path: Path):
    """The generic one-shot path decides on CARS without an explicit instruction."""
    from deerflow.community.nir.tools import nir_analyze_tool

    rng = np.random.RandomState(132)
    X = rng.normal(size=(70, 60))
    y = 1.5 + X[:, 3] * 1.2 - X[:, 17] * 0.7 + rng.normal(scale=0.03, size=70)
    wv = np.linspace(900.0, 1700.0, X.shape[1])
    source = tmp_path / "one_shot.npz"
    output_dir = tmp_path / "one_shot_output"
    output_dir.mkdir()
    np.savez(source, X=X, y=y, wv=wv)

    with (
        patch(
            "deerflow.community.nir.modeling._resolve",
            return_value=str(source),
        ),
        patch(
            "deerflow.community.nir.modeling._resolve_writable_dir",
            return_value=str(output_dir),
        ),
        patch(
            "nir_core.model.selection.cars_wavelength_selection",
            side_effect=lambda spectra, *_args, **_kwargs: (
                spectra[:, [3, 17]],
                [3, 17],
            ),
        ),
    ):
        result = nir_analyze_tool.func(
            runtime=MagicMock(),
            data_path="/mnt/user-data/uploads/one_shot.npz",
            auto_preprocess=False,
        )

    payload = json.loads(result)
    persisted = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["wavelength_selection_decision"]["mode"] == "auto"
    assert payload["wavelength_selection_decision"]["evaluate_cars"] is True
    assert payload["model_selection_decision"]["mode"] == "auto"
    assert payload["model_selection_decision"]["selected_method"] == payload["method"]
    assert {item["method"] for item in payload["wavelength_selection_candidates"]} == {
        "none",
        "cars",
    }
    assert persisted["wavelength_selection_decision"] == payload["wavelength_selection_decision"]
    assert "模型选择模式" in (output_dir / "report.md").read_text(encoding="utf-8")


def test_auto_split_model_prefers_detected_group_field(tmp_path: Path):
    """Auto mode keeps detected batch groups intact across all three partitions."""
    import pandas as pd

    from deerflow.community.nir.tools import nir_train_auto_split_model_tool

    rng = np.random.RandomState(44)
    n_groups = 10
    samples_per_group = 8
    n_samples = n_groups * samples_per_group
    X = rng.normal(size=(n_samples, 15))
    y = 4.0 + X[:, 2] * 0.9 + rng.normal(scale=0.05, size=n_samples)
    frame = pd.DataFrame(X, columns=[str(1100 + 5 * index) for index in range(X.shape[1])])
    frame.insert(0, "Protein", y)
    frame.insert(0, "Batch", np.repeat([f"B{index:02d}" for index in range(n_groups)], samples_per_group))
    source = tmp_path / "grouped.csv"
    frame.to_csv(source, index=False)
    model_file = tmp_path / "grouped.pkl"
    metrics_file = tmp_path / "grouped.json"

    paths = {
        "/mnt/user-data/uploads/grouped.csv": str(source),
        "/mnt/user-data/outputs/grouped.pkl": str(model_file),
        "/mnt/user-data/outputs/grouped.json": str(metrics_file),
    }
    with patch(
        "deerflow.community.nir.modeling._resolve",
        side_effect=lambda _runtime, path, *, read_only: paths[path],
    ):
        result = nir_train_auto_split_model_tool.func(
            runtime=MagicMock(),
            file_path="/mnt/user-data/uploads/grouped.csv",
            y_col=1,
            x_cols="2:",
            split_strategy="auto",
            pipeline_steps='["snv", "autoscale"]',
            max_components=2,
            compare_cars=False,
            model_output="/mnt/user-data/outputs/grouped.pkl",
            metrics_output="/mnt/user-data/outputs/grouped.json",
        )

    payload = json.loads(result)
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    partitions = metrics["partitions"]
    group_sets = [set(partitions[name]["group_values"]) for name in ("calibration", "tuning", "holdout_test")]

    assert payload["status"] == "ok"
    assert metrics["split"]["strategy"] == "group"
    assert metrics["split"]["group_column"] == "Batch"
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])
    assert set.union(*group_sets) == {f"B{index:02d}" for index in range(n_groups)}


def test_autonomous_split_uses_y_stratification_above_spxy_limit():
    """Large ungrouped data avoids the quadratic SPXY distance matrix."""
    import pandas as pd

    from deerflow.community.nir.modeling import _select_autonomous_split

    rng = np.random.RandomState(55)
    X = rng.normal(size=(120, 8))
    y = np.linspace(0.0, 12.0, X.shape[0])
    raw = pd.DataFrame({"Target": y})

    calibration, tuning, holdout, decision = _select_autonomous_split(
        raw,
        X,
        y,
        y_col=0,
        strategy="auto",
        group_col=None,
        tuning_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        spxy_max_samples=50,
    )

    assert decision["strategy"] == "y_stratified"
    assert "SPXY limit" in decision["reason"]
    assert [len(calibration), len(tuning), len(holdout)] == [84, 18, 18]
    assert not (set(calibration) & set(tuning) or set(calibration) & set(holdout) or set(tuning) & set(holdout))


def test_wavelength_selection_policy_autonomously_evaluates_high_dimensional_data():
    from deerflow.community.nir.modeling import _decide_autonomous_wavelength_selection

    decision = _decide_autonomous_wavelength_selection(
        compare_cars=None,
        n_calibration=5000,
        n_wavelengths=306,
        baseline_rmse=0.2,
        y_tuning=np.linspace(0.0, 1.0, 100),
        max_selection_samples=750,
    )

    assert decision["mode"] == "auto"
    assert decision["evaluate_cars"] is True
    assert "many_wavelengths" in decision["signals"]
    assert decision["cars_fit_samples"] == 750
    assert decision["sample_cap_applied"] is True


def test_wavelength_selection_policy_skips_small_feature_sets_and_honors_override():
    from deerflow.community.nir.modeling import _decide_autonomous_wavelength_selection

    automatic = _decide_autonomous_wavelength_selection(
        compare_cars=None,
        n_calibration=100,
        n_wavelengths=20,
        baseline_rmse=0.8,
        y_tuning=np.linspace(0.0, 1.0, 30),
    )
    disabled = _decide_autonomous_wavelength_selection(
        compare_cars=False,
        n_calibration=100,
        n_wavelengths=300,
        baseline_rmse=0.2,
        y_tuning=np.linspace(0.0, 1.0, 30),
    )

    assert automatic["evaluate_cars"] is False
    assert automatic["reason_code"] == "insufficient_wavelengths"
    assert disabled["mode"] == "disabled"
    assert disabled["reason_code"] == "explicitly_disabled"


def test_wavelength_candidate_choice_requires_material_tuning_improvement():
    from deerflow.community.nir.modeling import _choose_wavelength_candidate

    full = {"method": "none", "RMSE_tuning": 1.0, "n_selected": 300, "n_components": 6}
    marginal_cars = {"method": "cars", "RMSE_tuning": 0.998, "n_selected": 40, "n_components": 5}
    useful_cars = {"method": "cars", "RMSE_tuning": 0.97, "n_selected": 45, "n_components": 5}

    chosen, evidence = _choose_wavelength_candidate([full, marginal_cars], min_relative_improvement=0.005)
    assert chosen["method"] == "none"
    assert evidence["reason_code"] == "cars_improvement_below_threshold"

    chosen, evidence = _choose_wavelength_candidate([full, useful_cars], min_relative_improvement=0.005)
    assert chosen["method"] == "cars"
    assert evidence["reason_code"] == "cars_improved_tuning_rmse"


def test_model_selection_policy_expands_candidates_only_when_signals_justify_it():
    from deerflow.community.nir.modeling import _decide_autonomous_model_selection

    weak_high_dimensional = _decide_autonomous_model_selection(
        requested_method="auto",
        n_calibration=120,
        n_features=180,
        baseline_rmse=0.8,
        y_tuning=np.linspace(0.0, 1.0, 30),
    )
    strong_compact = _decide_autonomous_model_selection(
        requested_method="auto",
        n_calibration=200,
        n_features=30,
        baseline_rmse=0.02,
        y_tuning=np.linspace(0.0, 1.0, 40),
    )

    assert weak_high_dimensional["candidate_methods"] == [
        "pls",
        "ridge",
        "svr",
        "et",
    ]
    assert "weak_pls_tuning_fit" in weak_high_dimensional["signals"]
    assert strong_compact["candidate_methods"] == ["pls"]
    assert strong_compact["reason_code"] == "pls_baseline_sufficient"


def test_model_selection_policy_honors_explicit_method_and_runtime_caps():
    from deerflow.community.nir.modeling import _decide_autonomous_model_selection

    forced = _decide_autonomous_model_selection(
        requested_method="svr",
        n_calibration=5000,
        n_features=300,
        baseline_rmse=1.0,
        y_tuning=np.linspace(0.0, 1.0, 100),
    )
    capped = _decide_autonomous_model_selection(
        requested_method="auto",
        n_calibration=5000,
        n_features=300,
        baseline_rmse=1.0,
        y_tuning=np.linspace(0.0, 1.0, 100),
        max_svr_samples=1500,
        max_tree_samples=2500,
    )

    assert forced["mode"] == "forced"
    assert forced["candidate_methods"] == ["svr"]
    assert "svr" not in capped["candidate_methods"]
    assert "et" not in capped["candidate_methods"]
    assert capped["runtime_caps_applied"] == ["svr", "et"]


def test_model_candidate_choice_requires_material_improvement_over_pls():
    from deerflow.community.nir.modeling import _choose_model_candidate

    pls = {"method": "pls", "RMSE_tuning": 1.0}
    marginal_svr = {"method": "svr", "RMSE_tuning": 0.995}
    useful_svr = {"method": "svr", "RMSE_tuning": 0.94}

    chosen, evidence = _choose_model_candidate(
        [pls, marginal_svr],
        min_relative_improvement=0.01,
    )
    assert chosen["method"] == "pls"
    assert evidence["reason_code"] == "alternative_improvement_below_threshold"

    chosen, evidence = _choose_model_candidate(
        [pls, useful_svr],
        min_relative_improvement=0.01,
    )
    assert chosen["method"] == "svr"
    assert evidence["reason_code"] == "alternative_improved_tuning_rmse"


def test_model_family_selection_detects_nonlinear_candidate_improvement():
    from deerflow.community.nir.modeling import _select_model_family

    rng = np.random.RandomState(144)
    X = rng.uniform(-2.0, 2.0, size=(140, 8))
    y = np.sin(2.5 * X[:, 0]) + 0.6 * X[:, 1] ** 2 + rng.normal(scale=0.03, size=140)

    chosen, decision, candidates = _select_model_family(
        "auto",
        X[:105],
        y[:105],
        X[105:],
        y[105:],
        max_components=6,
        cv_folds=3,
        max_svr_samples=1500,
        max_tree_samples=2500,
        min_relative_improvement=0.01,
    )

    assert {item["method"] for item in candidates} == {"pls", "ridge", "svr", "et"}
    assert chosen["method"] in {"svr", "et"}
    assert decision["selected_method"] == chosen["method"]
    assert decision["adoption"]["reason_code"] == "alternative_improved_tuning_rmse"


def test_nir_predict_applies_artifact_wavelength_selection(tmp_path: Path):
    """Prediction accepts a full-width X matrix and slices train-selected columns."""
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir._common import _write_trusted_model_artifact
    from deerflow.community.nir.io_tools import nir_predict_tool

    rng = np.random.RandomState(13)
    X = rng.rand(20, 5)
    selected_indices = [1, 3]
    y = X[:, selected_indices] @ np.array([2.0, -1.0])
    model = LinearRegression().fit(X[:, selected_indices], y)

    model_file = tmp_path / "model.pkl"
    data_file = tmp_path / "predict.npz"
    np.savez(data_file, X=X)
    _write_trusted_model_artifact(
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
        str(model_file),
    )

    virtual_model = "/mnt/user-data/outputs/model.pkl"
    virtual_data = "/mnt/user-data/uploads/predict.npz"
    resolved = {
        virtual_model: str(model_file),
        virtual_data: str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(tmp_path / "prediction-audit.jsonl"),
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
    from nir_core.models import PreprocessingStep
    from nir_core.preprocess.pipeline import PreprocessingPipeline
    from sklearn.linear_model import LinearRegression

    from deerflow.community.nir._common import _write_trusted_model_artifact
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
    _write_trusted_model_artifact(
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
        str(model_file),
    )

    resolved = {
        "/mnt/user-data/outputs/model-v2.pkl": str(model_file),
        "/mnt/user-data/uploads/raw-predict.npz": str(data_file),
        "/mnt/user-data/outputs/prediction-audit.jsonl": str(tmp_path / "prediction-audit.jsonl"),
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
