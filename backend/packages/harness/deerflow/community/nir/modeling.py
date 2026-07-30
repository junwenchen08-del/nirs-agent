"""NIR modeling tools: train / analyze / compare / register.

This module contains the model-building tools that share a common
pattern: load data → optional leakage-safe preprocessing → train PLS/PCR/SVR
→ full evaluation → quality gate → plots + report → knowledge_hint.

Keeping them together (rather than splitting per-tool) avoids duplicating
the shared train+evaluate helper code, while still separating them from
I/O, preprocessing-only, and reflection tools.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.resources import ResourceLimitError
from nir_core.io.sniffers import inspect_file

from deerflow.tools.types import Runtime

from ._common import (
    _bind_model_metrics,
    _err,
    _json_default,
    _load_npz_safely,
    _model_evidence,
    _model_runtime_preflight_error,
    _ok,
    _parse_pipeline_step,
    _resolve,
    _resolve_writable_dir,
    _sha256_file,
    _write_trusted_model_artifact,
)
from ._knowledge_hint import _build_knowledge_hint
from ._report import _build_report
from ._resources import budget_for_runtime, check_spectral_data, resource_error
from ._science_gate import (
    dataset_science_gate,
    reproducibility_evidence,
    science_gate_error,
    split_science_gate,
)
from .artifacts import _build_model_artifact
from .candidate_selection import (
    _choose_model_candidate,  # noqa: F401 - compatibility re-export
    _choose_wavelength_candidate,
    _decide_autonomous_model_selection,  # noqa: F401 - compatibility re-export
    _decide_autonomous_wavelength_selection,
    _model_candidate_summary,
    _wavelength_candidate_summary,
)
from .data_splitting import select_autonomous_split
from .multi_target import (
    _parse_multi_pipeline,  # noqa: F401 - compatibility re-export
    nir_train_multi_model_tool,  # noqa: F401 - compatibility re-export
)
from .registration import nir_register_model_tool  # noqa: F401 - compatibility re-export
from .single_target import (
    _apply_wavelength_selection,
    _autonomous_pls_wavelength_selection,
    _extract_rmsecv,
    _score_pls_wavelength_candidate,
    _select_model_family,
    _target_rank_selection_indices,
    _train_one_model,
    nir_train_model_tool,  # noqa: F401 - compatibility re-export
)

# Fast single-component CSV protocols
# ---------------------------------------------------------------------------


# Data-splitting implementations live in data_splitting.py. Aliases imported
# above preserve the historical private API while keeping this facade small.


def _select_autonomous_split(*args, **kwargs):
    """Compatibility facade for callers that imported the former helper."""
    return select_autonomous_split(*args, **kwargs)


@tool("nir_train_auto_split_model", parse_docstring=True)
def nir_train_auto_split_model_tool(
    runtime: Runtime,
    file_path: str,
    y_col: int,
    x_cols: str,
    wv_row: int | None = 0,
    tuning_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    split_strategy: str = "auto",
    group_col: str | None = None,
    spxy_max_samples: int = 500,
    pipeline_steps: str = '["snv", "autoscale"]',
    method: str = "auto",
    max_components: int = 20,
    compare_cars: bool | None = None,
    cars_params: str = '{"n_mc_samples": 20, "n_folds": 5, "random_state": 42}',
    max_cars_samples: int = 750,
    min_cars_improvement: float = 0.005,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    model_output: str = "/mnt/user-data/outputs/auto_split_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/auto_split_metrics.json",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train an autonomously selected model for a CSV without official partitions.

    The tool autonomously selects a reproducible calibration/tuning/holdout
    split. It prefers eligible batch/domain metadata fields, uses SPXY for
    smaller ungrouped datasets, and falls back to target-value stratification
    above the SPXY size limit. All
    preprocessing and optional CARS selection are fitted on calibration rows
    only. Tuning rows select the wavelength candidate and, by default, a
    bounded model family from PLS/Ridge/SVR/Extra Trees. Holdout rows are
    evaluated once after all choices are fixed. The holdout metrics are not
    presented as external validation.

    Args:
        file_path: Virtual path to the original labelled CSV file.
        y_col: 0-based numeric target-column index in the CSV.
        x_cols: Spectral-column selector passed to nir_core, for example ``"9:"``.
        wv_row: Wavelength/header row index for mixed text/numeric CSV layouts.
        tuning_ratio: Fraction reserved for model selection; default 0.15.
        test_ratio: Fraction reserved for one-time holdout testing; default 0.15.
        random_state: Reproducible split seed.
        split_strategy: ``auto`` (default), ``group``, ``spxy``,
            ``y_stratified``, or ``random``.
        group_col: Optional explicit batch/domain column for group splitting.
        spxy_max_samples: Largest sample count eligible for automatic SPXY;
            larger ungrouped datasets use target-value stratification.
        pipeline_steps: JSON preprocessing pipeline fitted on calibration rows.
        method: ``auto`` (default) for bounded model-family selection, or an
            explicit supported model such as ``pls`` or ``svr``.
        max_components: Maximum PLS latent variables considered on tuning rows.
        compare_cars: Optional override. None (default) lets the runtime decide;
            true forces a CARS comparison and false disables it.
        cars_params: JSON CARS parameters used when CARS is evaluated.
        max_cars_samples: Maximum calibration samples used to fit CARS. Larger
            sets use deterministic target-rank coverage, still calibration-only.
        min_cars_improvement: Minimum relative Tuning RMSE improvement required
            before the selected CARS wavelengths replace the full spectrum.
        max_svr_samples: Largest calibration set eligible for autonomous SVR.
        max_tree_samples: Largest calibration set eligible for autonomous
            Extra Trees comparison.
        min_model_improvement: Minimum relative Tuning RMSE improvement an
            alternative must achieve before replacing PLS.
        model_output: Virtual output path for a deployable model artifact.
        metrics_output: Virtual output path for holdout metrics and split indices.
        domain: Application domain used by the quality gate.

    Returns:
        JSON containing the deterministic split, tuning and holdout metrics,
        selected model details, and persisted artifact/report paths.
    """
    try:
        import json

        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        import pandas as pd
        from nir_core.io.loaders import load_csv
        from nir_core.model.evaluation import compute_metrics
        from nir_core.model.selection import cars_wavelength_selection
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.utils.metrics import evaluate_quality

        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, file_path, read_only=True)
        budget.check_file(real_in, stage="auto_split_file_preflight")
        raw = pd.read_csv(real_in)
        if not (0 <= int(y_col) < raw.shape[1]):
            return _err(f"y_col {y_col} is outside the CSV column range 0..{raw.shape[1] - 1}")
        target_name = str(raw.columns[int(y_col)])
        data = load_csv(real_in, y_col=int(y_col), x_cols=x_cols, wv_row=wv_row)
        check_spectral_data(budget, data, stage="auto_split_matrix_preflight", peak_multiplier=8.0)
        if data.y is None or data.wv is None:
            return _err("Auto-split calibration requires both a target column and wavelength headers.")

        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel()
        wv = np.asarray(data.wv, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y length ({y.shape[0]})")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            return _err("Auto-split calibration requires finite X and y values.")
        dataset_validation = dataset_science_gate(X, y, wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)

        budget.checkpoint("auto_split_selection")
        calibration_indices, tuning_indices, test_indices, split_decision = select_autonomous_split(
            raw,
            X,
            y,
            y_col=int(y_col),
            strategy=split_strategy,
            group_col=group_col,
            tuning_ratio=float(tuning_ratio),
            test_ratio=float(test_ratio),
            random_state=int(random_state),
            spxy_max_samples=int(spxy_max_samples),
        )
        X_cal, y_cal = X[calibration_indices], y[calibration_indices]
        X_tune, y_tune = X[tuning_indices], y[tuning_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        split_validation = split_science_gate(
            calibration=X_cal,
            tuning=X_tune,
            holdout=X_test,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        try:
            raw_steps = json.loads(pipeline_steps)
            if not isinstance(raw_steps, list):
                return _err("pipeline_steps must be a JSON list")
            steps = [_parse_pipeline_step(value) for value in raw_steps]
        except (TypeError, ValueError) as exc:
            return _err(f"Invalid pipeline_steps JSON: {exc}")
        is_valid, reason = validate_pipeline(steps)
        if not is_valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")

        selection_pipe = PreprocessingPipeline(steps).fit(X_cal, wv)
        X_cal_s = selection_pipe.transform(X_cal, wv)
        X_tune_s = selection_pipe.transform(X_tune, wv)

        full_result = _score_pls_wavelength_candidate(
            {"method": "none", "indices": list(range(X.shape[1])), "params": {}},
            X_cal_s,
            y_cal,
            X_tune_s,
            y_tune,
            max_components=int(max_components),
        )
        selection_decision = _decide_autonomous_wavelength_selection(
            compare_cars=compare_cars,
            n_calibration=X_cal_s.shape[0],
            n_wavelengths=X_cal_s.shape[1],
            baseline_rmse=full_result["RMSE_tuning"],
            y_tuning=y_tune,
            max_selection_samples=int(max_cars_samples),
        )
        candidate_results: list[dict] = [full_result]
        if selection_decision["evaluate_cars"]:
            parsed_cars = json.loads(cars_params)
            if not isinstance(parsed_cars, dict):
                return _err("cars_params must be a JSON object")
            selection_rows = _target_rank_selection_indices(y_cal, selection_decision["cars_fit_samples"])
            _, cars_indices = cars_wavelength_selection(
                X_cal_s[selection_rows],
                y_cal[selection_rows],
                n_mc_samples=int(parsed_cars.get("n_mc_samples", 20)),
                n_folds=int(parsed_cars.get("n_folds", 5)),
                random_state=int(parsed_cars.get("random_state", random_state)),
            )
            candidate_results.append(
                _score_pls_wavelength_candidate(
                    {
                        "method": "cars",
                        "indices": cars_indices,
                        "params": parsed_cars,
                        "selection_fit_samples": int(selection_rows.size),
                    },
                    X_cal_s,
                    y_cal,
                    X_tune_s,
                    y_tune,
                    max_components=int(max_components),
                )
            )

        chosen, adoption_decision = _choose_wavelength_candidate(
            candidate_results,
            min_relative_improvement=float(min_cars_improvement),
        )
        selection_decision["adoption"] = adoption_decision
        selected_indices = [int(index) for index in chosen["indices"]]
        X_cal_selected = X_cal_s[:, selected_indices]
        X_tune_selected = X_tune_s[:, selected_indices]
        budget.checkpoint("auto_split_model_selection")
        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            method,
            X_cal_selected,
            y_cal,
            X_tune_selected,
            y_tune,
            max_components=int(max_components),
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )
        tuning_prediction = chosen_model["_tuning_prediction"]

        final_train_indices = np.sort(np.concatenate([calibration_indices, tuning_indices]))
        final_pipe = PreprocessingPipeline(steps).fit(X[final_train_indices], wv)
        X_final = final_pipe.transform(X[final_train_indices], wv)[:, selected_indices]
        X_holdout = final_pipe.transform(X_test, wv)[:, selected_indices]
        final_model, best_n, cv_results, final_predict_fn = _train_one_model(
            chosen_model["method"],
            X_final,
            y[final_train_indices],
            max_components=int(max_components),
            cv_folds=5,
            cv_strategy="auto",
        )
        train_prediction = final_predict_fn(final_model, X_final)
        holdout_prediction = final_predict_fn(final_model, X_holdout)

        selection_meta = {
            "method": chosen["method"],
            "params": chosen["params"],
            "n_original": int(X.shape[1]),
            "n_selected": int(len(selected_indices)),
            "selected_indices": None if chosen["method"] == "none" else selected_indices,
            "selected_wavelengths": None if chosen["method"] == "none" else [float(wv[index]) for index in selected_indices],
        }
        partition_metrics = {
            "calibration": {"n_samples": int(calibration_indices.size), "sample_indices": calibration_indices.tolist()},
            "tuning": {"n_samples": int(tuning_indices.size), "sample_indices": tuning_indices.tolist()},
            "holdout_test": {"n_samples": int(test_indices.size), "sample_indices": test_indices.tolist()},
        }
        group_assignments = split_decision.get("group_assignments")
        if group_assignments:
            for partition_name, group_values in group_assignments.items():
                partition_metrics[partition_name]["group_values"] = group_values

        training_data_hash = _sha256_file(real_in)
        metrics = {
            "training_data_hash": training_data_hash,
            "scientific_validation": {
                "schema_version": 1,
                "passed": True,
                "dataset": dataset_validation,
                "partition_separation": split_validation,
            },
            "reproducibility": reproducibility_evidence(
                random_state=int(random_state),
                protocol="deterministic_auto_split_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "split_strategy": split_decision["strategy"],
                    "tuning_ratio": float(tuning_ratio),
                    "test_ratio": float(test_ratio),
                    "method": method,
                    "max_components": int(max_components),
                    "compare_cars": compare_cars,
                    "min_cars_improvement": float(min_cars_improvement),
                    "min_model_improvement": float(min_model_improvement),
                },
            ),
            "protocol": "deterministic_auto_split_holdout",
            "validation_scope": "independent_holdout_not_external",
            "target": target_name,
            "method": chosen_model["method"],
            "n_components": None if best_n is None else int(best_n),
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(X.shape[1]),
            "n_wavelengths_model": int(len(selected_indices)),
            "preprocessing": final_pipe.description(),
            "preprocessing_steps": raw_steps,
            "wavelength_selection": selection_meta,
            "wavelength_selection_decision": selection_decision,
            "model_selection_decision": model_selection_decision,
            "split": {
                **split_decision,
                "requested_ratios": {
                    "calibration": float(1.0 - tuning_ratio - test_ratio),
                    "tuning": float(tuning_ratio),
                    "holdout_test": float(test_ratio),
                },
            },
            "partitions": partition_metrics,
            "candidate_results": candidate_results,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y[final_train_indices], train_prediction),
            "val": compute_metrics(y_tune, tuning_prediction),
            "test": compute_metrics(y_test, holdout_prediction),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        metrics["RMSECV"] = metrics["val"]["RMSE"]
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(final_train_indices.size))
        metrics["quality"] = quality

        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        _write_trusted_model_artifact(
            _build_model_artifact(
                final_model,
                method=chosen_model["method"],
                preprocessing_pipeline=final_pipe,
                preprocessing_desc=final_pipe.description(),
                wavelength_selection=selection_meta,
                X_reference=X_final,
            ),
            real_model,
        )
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(
            real_model,
            real_metrics,
            training_data_hash=training_data_hash,
        )

        report_path = os.path.join(os.path.dirname(real_model), "auto_split_report.md")
        report = [
            "# Auto-split NIR holdout report",
            "",
            f"- Target: `{target_name}`",
            f"- Split strategy: `{split_decision['strategy']}` (requested `{split_decision['requested_strategy']}`)",
            f"- Split reason: {split_decision['reason']}",
            f"- Random seed: {int(random_state)}",
            f"- Samples: {calibration_indices.size} calibration / {tuning_indices.size} tuning / {test_indices.size} holdout",
            "- Validation scope: independent holdout generated from this dataset; not external validation",
            f"- Pipeline: {final_pipe.description()}",
            f"- Wavelength-selection policy: {selection_decision['mode']} — {selection_decision['reason']}",
            f"- Selection: {chosen['method']} ({len(selected_indices)} of {X.shape[1]} wavelengths)",
            f"- Model-selection policy: {model_selection_decision['mode']} - {model_selection_decision['reason']}",
            f"- Selected model: {chosen_model['method']}",
            f"- Latent variables: {best_n if best_n is not None else 'not applicable'}",
            f"- Tuning RMSE: {metrics['val']['RMSE']:.5f}",
            f"- Holdout RMSE: {metrics['test']['RMSE']:.5f}",
            f"- Holdout R2: {metrics['test']['R2']:.5f}",
            f"- Holdout RPD: {metrics['test']['RPD']:.5f}",
            f"- Holdout bias: {metrics['test']['bias']:.5f}",
        ]
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")

        out_virtual = os.path.dirname(model_output)
        candidate_summary = [
            {
                "method": item["method"],
                "n_selected": item["n_selected"],
                "n_components": item["n_components"],
                "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            }
            for item in candidate_results
        ]
        return _ok(
            {
                "status": "ok",
                "protocol": "deterministic_auto_split_holdout",
                "validation_scope": "independent_holdout_not_external",
                "target": target_name,
                "split": metrics["split"],
                "partitions": {name: {"n_samples": value["n_samples"]} for name, value in metrics["partitions"].items()},
                "preprocessing": final_pipe.description(),
                "wavelength_selection": selection_meta,
                "wavelength_selection_decision": selection_decision,
                "model_selection_decision": model_selection_decision,
                "method": chosen_model["method"],
                "n_components": None if best_n is None else int(best_n),
                "RMSE_tuning": round(float(metrics["val"]["RMSE"]), 5),
                "holdout": {key: round(float(metrics["test"][key]), 5) for key in ("RMSE", "R2", "RPD", "bias")},
                "candidate_results": candidate_summary,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "evidence": _model_evidence(
                    real_model,
                    real_metrics,
                    training_data_hash=training_data_hash,
                    protocol="deterministic_auto_split_holdout",
                    validation_scope="independent_holdout_not_external",
                ),
                "report": out_virtual + "/auto_split_report.md",
                "resource_budget": budget.evidence(),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_train_partitioned_model", parse_docstring=True)
def nir_train_partitioned_model_tool(
    runtime: Runtime,
    file_path: str,
    split_col: str,
    train_label: str,
    tuning_label: str,
    test_label: str,
    y_col: int,
    x_cols: str,
    wv_row: int | None = 0,
    pipeline_steps: str = '["snv", {"method": "derivative1", "params": {"window": 15, "order": 2}}, "autoscale"]',
    method: str = "auto",
    domain: str = "default",
    max_components: int = 20,
    compare_cars: bool | None = None,
    cars_params: str = '{"n_mc_samples": 50, "n_folds": 5, "random_state": 42}',
    max_cars_samples: int = 750,
    min_cars_improvement: float = 0.005,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    model_output: str = "/mnt/user-data/outputs/partitioned_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/partitioned_metrics.json",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train an autonomously selected model on named data partitions.

    This is intended for datasets whose CSV records carry an official split
    column, rather than a random split.  It enforces the following protocol:
    preprocessing and CARS selection are fitted on ``train_label`` only;
    ``tuning_label`` selects wavelengths and a bounded model family;
    ``test_label`` is used once for final external metrics. This prevents a
    held-out season, batch, or instrument set from leaking into any selection
    decision.

    Args:
        file_path: Virtual path to the original labelled CSV file.
        split_col: Header name containing named partitions, for example ``Set``.
        train_label: Value in split_col used for calibration, for example ``Cal``.
        tuning_label: Value used for model selection, for example ``Tuning``.
        test_label: Value reserved for external testing, for example ``Val Ext``.
        y_col: 0-based numeric target-column index in the CSV.
        x_cols: Spectral-column selector passed to nir_core, for example ``"9:"``.
        wv_row: Wavelength/header row index for mixed text/numeric CSV layouts.
        domain: Application domain used for quality-gate thresholds.
        pipeline_steps: JSON preprocessing pipeline, fitted on Cal only during selection.
        method: ``auto`` (default) or an explicit supported model family.
        max_components: Maximum PLS latent variables considered on Tuning.
        compare_cars: Optional override. None (default) autonomously decides,
            true forces CARS comparison, and false disables it.
        cars_params: JSON CARS parameters used when CARS is evaluated.
        max_cars_samples: Maximum calibration samples used to fit CARS.
        min_cars_improvement: Minimum relative Tuning RMSE improvement required
            before adopting the CARS subset.
        max_svr_samples: Largest Cal set eligible for autonomous SVR.
        max_tree_samples: Largest Cal set eligible for autonomous Extra Trees.
        min_model_improvement: Minimum relative Tuning RMSE improvement required
            before an alternative model replaces PLS.
        model_output: Virtual output path for a deployable model artifact.
        metrics_output: Virtual output path for external-validation metrics.

    Returns:
        JSON containing tuning and external metrics, selected wavelengths, and
        paths to the persisted model, metrics, and Markdown report.
    """
    try:
        import json

        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        import pandas as pd
        from nir_core.io.loaders import load_csv
        from nir_core.model.evaluation import compute_metrics
        from nir_core.model.selection import cars_wavelength_selection
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.utils.metrics import evaluate_quality

        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, file_path, read_only=True)
        budget.check_file(real_in, stage="partitioned_file_preflight")
        raw = pd.read_csv(real_in)
        if split_col not in raw.columns:
            return _err(f"split_col {split_col!r} is not a CSV header; available columns include {list(raw.columns[:12])}")

        data = load_csv(real_in, y_col=y_col, x_cols=x_cols, wv_row=wv_row)
        check_spectral_data(
            budget,
            data,
            stage="partitioned_matrix_preflight",
            peak_multiplier=8.0,
        )
        labels = raw[split_col].astype(str).to_numpy()
        if labels.shape[0] != data.X.shape[0]:
            return _err(f"Split column has {labels.shape[0]} rows but loaded spectra have {data.X.shape[0]}. Use a one-row CSV header and a wv_row that is not a sample row.")
        if data.y is None or data.wv is None:
            return _err("Partitioned calibration requires both y_col and a wavelength row.")

        train_mask = labels == str(train_label)
        tune_mask = labels == str(tuning_label)
        test_mask = labels == str(test_label)
        if not (train_mask.any() and tune_mask.any() and test_mask.any()):
            counts = {value: int((labels == value).sum()) for value in (str(train_label), str(tuning_label), str(test_label))}
            return _err(f"Each partition must contain samples; observed counts: {counts}")
        if np.any((train_mask.astype(int) + tune_mask.astype(int) + test_mask.astype(int)) > 1):
            return _err("train_label, tuning_label, and test_label must be distinct.")

        try:
            raw_steps = json.loads(pipeline_steps)
            steps = [_parse_pipeline_step(value) for value in raw_steps]
        except (TypeError, ValueError) as exc:
            return _err(f"Invalid pipeline_steps JSON: {exc}")
        is_valid, reason = validate_pipeline(steps)
        if not is_valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")

        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel()
        wv = np.asarray(data.wv, dtype=float).ravel()
        dataset_validation = dataset_science_gate(X, y, wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)
        X_cal, y_cal = X[train_mask], y[train_mask]
        X_tune, y_tune = X[tune_mask], y[tune_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        split_validation = split_science_gate(
            calibration=X_cal,
            tuning=X_tune,
            holdout=X_test,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        # Fit all stateful transforms on Cal only for every tuning decision.
        selection_pipe = PreprocessingPipeline(steps).fit(X_cal, wv)
        X_cal_s = selection_pipe.transform(X_cal, wv)
        X_tune_s = selection_pipe.transform(X_tune, wv)

        full_result = _score_pls_wavelength_candidate(
            {"method": "none", "indices": list(range(X.shape[1])), "params": {}},
            X_cal_s,
            y_cal,
            X_tune_s,
            y_tune,
            max_components=int(max_components),
        )
        selection_decision = _decide_autonomous_wavelength_selection(
            compare_cars=compare_cars,
            n_calibration=X_cal_s.shape[0],
            n_wavelengths=X_cal_s.shape[1],
            baseline_rmse=full_result["RMSE_tuning"],
            y_tuning=y_tune,
            max_selection_samples=int(max_cars_samples),
        )
        candidate_results: list[dict] = [full_result]
        if selection_decision["evaluate_cars"]:
            try:
                parsed_cars = json.loads(cars_params)
                if not isinstance(parsed_cars, dict):
                    return _err("cars_params must be a JSON object")
                selection_rows = _target_rank_selection_indices(y_cal, selection_decision["cars_fit_samples"])
                _, cars_indices = cars_wavelength_selection(
                    X_cal_s[selection_rows],
                    y_cal[selection_rows],
                    n_mc_samples=int(parsed_cars.get("n_mc_samples", 50)),
                    n_folds=int(parsed_cars.get("n_folds", 5)),
                    random_state=int(parsed_cars.get("random_state", 42)),
                )
                candidate_results.append(
                    _score_pls_wavelength_candidate(
                        {
                            "method": "cars",
                            "indices": cars_indices,
                            "params": parsed_cars,
                            "selection_fit_samples": int(selection_rows.size),
                        },
                        X_cal_s,
                        y_cal,
                        X_tune_s,
                        y_tune,
                        max_components=int(max_components),
                    )
                )
            except (TypeError, ValueError) as exc:
                return _err(f"CARS selection failed: {exc}")

        chosen, adoption_decision = _choose_wavelength_candidate(
            candidate_results,
            min_relative_improvement=float(min_cars_improvement),
        )
        selection_decision["adoption"] = adoption_decision
        selected_indices = [int(index) for index in chosen["indices"]]
        budget.checkpoint("partitioned_model_selection")
        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            method,
            X_cal_s[:, selected_indices],
            y_cal,
            X_tune_s[:, selected_indices],
            y_tune,
            max_components=int(max_components),
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )

        # Final fitting may incorporate Tuning data, but never external test data.
        final_train_mask = train_mask | tune_mask
        final_pipe = PreprocessingPipeline(steps).fit(X[final_train_mask], wv)
        X_final = final_pipe.transform(X[final_train_mask], wv)[:, selected_indices]
        X_external = final_pipe.transform(X_test, wv)[:, selected_indices]
        final_model, best_n, cv_results, final_predict_fn = _train_one_model(
            chosen_model["method"],
            X_final,
            y[final_train_mask],
            max_components=int(max_components),
            cv_folds=5,
            cv_strategy="auto",
        )
        y_pred_train = final_predict_fn(final_model, X_final)
        y_pred_external = final_predict_fn(final_model, X_external)

        selection_meta = {
            "method": chosen["method"],
            "params": chosen["params"],
            "n_original": int(X.shape[1]),
            "n_selected": int(len(selected_indices)),
            "selected_indices": None if chosen["method"] == "none" else selected_indices,
            "selected_wavelengths": None if chosen["method"] == "none" else [float(wv[index]) for index in selected_indices],
        }
        training_data_hash = _sha256_file(real_in)
        metrics = {
            "training_data_hash": training_data_hash,
            "scientific_validation": {
                "schema_version": 1,
                "passed": True,
                "dataset": dataset_validation,
                "partition_separation": split_validation,
            },
            "reproducibility": reproducibility_evidence(
                random_state=42,
                protocol="official_partition_external_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "split_column": split_col,
                    "train_label": train_label,
                    "tuning_label": tuning_label,
                    "test_label": test_label,
                    "method": method,
                    "max_components": int(max_components),
                    "compare_cars": compare_cars,
                    "min_cars_improvement": float(min_cars_improvement),
                    "min_model_improvement": float(min_model_improvement),
                },
            ),
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "method": chosen_model["method"],
            "n_components": None if best_n is None else int(best_n),
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(X.shape[1]),
            "n_wavelengths_model": int(len(selected_indices)),
            "preprocessing": final_pipe.description(),
            "preprocessing_steps": raw_steps,
            "wavelength_selection": selection_meta,
            "wavelength_selection_decision": selection_decision,
            "model_selection_decision": model_selection_decision,
            "partitions": {
                "column": split_col,
                "train": {"label": train_label, "n_samples": int(train_mask.sum())},
                "tuning": {"label": tuning_label, "n_samples": int(tune_mask.sum())},
                "external_test": {"label": test_label, "n_samples": int(test_mask.sum())},
            },
            "candidate_results": candidate_results,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y[final_train_mask], y_pred_train),
            "val": {"RMSE": float(chosen_model["RMSE_tuning"])},
            "test": compute_metrics(y_test, y_pred_external),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = float(1.0 - (chosen_model["RMSE_tuning"] ** 2) / np.var(y_tune)) if np.var(y_tune) > 0 else float("nan")
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEC"] = metrics["train"]["RMSE"]
        metrics["RMSECV"] = float(chosen_model["RMSE_tuning"])
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(final_train_mask.sum()))
        metrics["quality"] = quality

        real_model = _resolve(runtime, model_output, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        _write_trusted_model_artifact(
            _build_model_artifact(
                final_model,
                method=chosen_model["method"],
                preprocessing_pipeline=final_pipe,
                preprocessing_desc=final_pipe.description(),
                wavelength_selection=selection_meta,
                X_reference=X_final,
            ),
            real_model,
        )
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        with open(real_metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(
            real_model,
            real_metrics,
            training_data_hash=training_data_hash,
        )

        report_path = os.path.join(os.path.dirname(real_model), "partitioned_report.md")
        report = [
            "# Partitioned NIR external-validation report",
            "",
            f"- Split: `{split_col}` = `{train_label}` / `{tuning_label}` / `{test_label}`",
            f"- Samples: {int(train_mask.sum())} / {int(tune_mask.sum())} / {int(test_mask.sum())}",
            f"- Pipeline: {final_pipe.description()}",
            f"- Wavelength-selection policy: {selection_decision['mode']} — {selection_decision['reason']}",
            f"- Selection: {chosen['method']} ({len(selected_indices)} of {X.shape[1]} wavelengths)",
            f"- Model-selection policy: {model_selection_decision['mode']} - {model_selection_decision['reason']}",
            f"- Selected model: {chosen_model['method']}",
            f"- Latent variables: {best_n if best_n is not None else 'not applicable'}",
            f"- Tuning RMSE: {chosen_model['RMSE_tuning']:.5f}",
            f"- External RMSE: {metrics['test']['RMSE']:.5f}",
            f"- External R2: {metrics['test']['R2']:.5f}",
            f"- External RPD: {metrics['test']['RPD']:.5f}",
        ]
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")

        out_virtual = os.path.dirname(model_output)
        candidate_summary = [
            {
                "method": item["method"],
                "n_selected": item["n_selected"],
                "n_components": item["n_components"],
                "RMSE_tuning": round(float(item["RMSE_tuning"]), 5),
            }
            for item in candidate_results
        ]
        return _ok(
            {
                "status": "ok",
                "protocol": "named_partition_external_validation",
                "validation_scope": "independent_external_validation",
                "preprocessing": final_pipe.description(),
                "wavelength_selection": selection_meta,
                "wavelength_selection_decision": selection_decision,
                "model_selection_decision": model_selection_decision,
                "method": chosen_model["method"],
                "n_components": None if best_n is None else int(best_n),
                "RMSE_tuning": round(float(chosen_model["RMSE_tuning"]), 5),
                "external": {key: round(float(metrics["test"][key]), 5) for key in ("RMSE", "R2", "RPD", "bias")},
                "candidate_results": candidate_summary,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "evidence": _model_evidence(
                    real_model,
                    real_metrics,
                    training_data_hash=training_data_hash,
                    protocol="named_partition_external_validation",
                    validation_scope="independent_external_validation",
                ),
                "report": out_virtual + "/partitioned_report.md",
                "resource_budget": budget.evidence(),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_analyze
# ---------------------------------------------------------------------------


@tool("nir_analyze", parse_docstring=True)
def nir_analyze_tool(
    runtime: Runtime,
    data_path: str,
    subset: str | None = None,
    auto_preprocess: bool = True,
    method: str = "auto",
    wavelength_selection: str = "auto",
    wavelength_selection_params: str | None = None,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    output_dir: str = "/mnt/user-data/outputs/nir_analysis",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """One-shot end-to-end NIR analysis pipeline (deterministic, single run).

    Runs: load → three-way split → (optional) nested-CV preprocessing
    selection → model training → full evaluation → quality assessment →
    Markdown report. This tool does NOT run the reflection loop; it returns
    a single best-effort result. For quality-driven retries, use the
    step-by-step workflow (nir_preprocess + nir_train_model) driven by the
    nir-coordinator Skill.

    Args:
        data_path: Virtual path to the raw spectral file (.mat/.csv/.txt) or
            an already-standardised .npz.
        subset: Optional sub-dataset name inside a MATLAB struct. Leave unset
            for flat files. For multiple subsets prefer
            ``nir_analyze_collection`` so the agent receives one compact result.
        auto_preprocess: If True, run nested-CV preprocessing selection over
            the default candidate pipelines and use the winner. If False,
            use raw spectra directly.
        method: ``auto`` (default) autonomously compares a bounded subset of
            PLS/Ridge/SVR/Extra Trees, or explicitly use ``pls`` / ``pcr`` /
            ``svr`` / ``rf`` / ``et`` / ``gbm`` / ``ridge`` / ``lasso`` /
            ``elasticnet`` / ``knn`` / ``mlp`` / ``cnn``.
        wavelength_selection: Train-only wavelength selection. ``"auto"``
            (default) compares full-spectrum PLS with CARS only when data and
            tuning signals justify the cost. Explicit alternatives are
            ``"none"``, ``"cars"``, ``"spa"``, or ``"manual"``.
        wavelength_selection_params: Optional JSON object with method params.
            Auto mode accepts CARS params plus ``max_selection_samples`` and
            ``min_relative_improvement``.
        max_svr_samples: Largest training set eligible for autonomous SVR.
        max_tree_samples: Largest training set eligible for autonomous Extra
            Trees comparison.
        min_model_improvement: Minimum relative validation RMSE improvement
            required before an alternative model replaces PLS.
        output_dir: Virtual directory for outputs (report, model, metrics).
        domain: Application domain for quality-gate thresholds.

    Returns:
        JSON with the full metrics, quality assessment, paths to the
        generated report and model files, and knowledge_hint.

        ★ knowledge_hint: When non-null (unknown domain or R²_val < 0.7),
        the LLM SHOULD call ``nir_search_knowledge(query=hint['query'])`` to
        retrieve relevant paper sections before reporting to the user. If the
        result is unsatisfactory, switch to the step-by-step workflow
        (nir_train_model + nir_reflect) for a reflection loop.
    """
    try:
        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        from nir_core.io.loaders import auto_detect_and_load, load_mat
        from nir_core.io.sniffers import detect_format
        from nir_core.model.evaluation import (
            compute_metrics,
            nested_cv_preprocessing,
            split_dataset,
        )
        from nir_core.utils.metrics import evaluate_quality

        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, data_path, read_only=True)
        budget.check_file(real_in, stage="analyze_file_preflight")
        real_outdir = _resolve_writable_dir(runtime, output_dir)

        # Load (npz or raw file).
        if data_path.endswith(".npz"):
            data_dict = _load_npz_safely(real_in)
            from nir_core.models import SpectralData

            data = SpectralData(
                X=np.asarray(data_dict["X"], dtype=float),
                y=(np.asarray(data_dict["y"], dtype=float).ravel() if data_dict.get("y") is not None and data_dict["y"].size else None),
                wv=(np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None),
            )
        elif subset is not None and detect_format(real_in) == "mat":
            data = load_mat(real_in, subset=subset)
        else:
            data = auto_detect_and_load(real_in)
        check_spectral_data(budget, data, stage="analyze_matrix_preflight", peak_multiplier=8.0)

        if data.y is None:
            return _err("No reference values (y) found; cannot train a model.")

        X, y = np.asarray(data.X, dtype=float), np.asarray(data.y, dtype=float).ravel()
        data_wv = np.asarray(data.wv, dtype=float).ravel() if data.wv is not None else None
        dataset_validation = dataset_science_gate(X, y, data_wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=0.20,
            val_ratio=0.10,
            random_state=42,
        )
        split_validation = split_science_gate(
            calibration=X_tr,
            tuning=X_val,
            holdout=X_te,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        # Preprocessing selection (leakage-safe fit/transform).
        best_pipe = None
        if auto_preprocess:
            best_pipe, _ = nested_cv_preprocessing(
                X_tr,
                y_tr,
                X_val,
                y_val,
                inner_folds=3,
                max_components=10,
                random_state=42,
                wv=data.wv,
            )
            best_pipe = best_pipe.__class__(best_pipe.steps).fit(X_tr, data.wv)
            X_tr = best_pipe.transform(X_tr, data.wv)
            X_val = best_pipe.transform(X_val, data.wv)
            X_te = best_pipe.transform(X_te, data.wv)

        preprocessing_desc = best_pipe.description() if best_pipe else "none"
        requested_model = (method or "auto").strip().lower()
        requested_selection = (wavelength_selection or "auto").strip().lower()
        wavelength_selection_candidates: list[dict] = []
        if requested_selection == "auto" and requested_model in {"auto", "pls"}:
            (
                X_tr,
                X_val,
                X_te,
                _model_wv,
                wavelength_selection_meta,
                wavelength_selection_decision,
                wavelength_selection_candidates,
            ) = _autonomous_pls_wavelength_selection(
                X_tr,
                y_tr,
                X_val,
                y_val,
                X_te,
                wv=data.wv,
                wavelength_selection_params=wavelength_selection_params,
                max_components=10,
                cv_folds=5,
            )
        else:
            explicit_selection = "none" if requested_selection == "auto" else requested_selection
            X_tr, X_val, X_te, _model_wv, wavelength_selection_meta = _apply_wavelength_selection(
                X_tr,
                X_val,
                X_te,
                y_tr,
                wv=data.wv,
                wavelength_selection=explicit_selection,
                wavelength_selection_params=wavelength_selection_params,
                cv_folds=5,
            )
            if requested_selection == "auto":
                wavelength_selection_decision = {
                    "mode": "auto",
                    "evaluate_cars": False,
                    "reason_code": "non_pls_method_uses_full_spectrum",
                    "reason": "Autonomous CARS comparison is currently limited to PLS; the requested model uses the full spectrum.",
                    "signals": [],
                    "selected_method": "none",
                    "adoption": {"reason_code": "full_spectrum_for_non_pls_method"},
                }
            else:
                wavelength_selection_decision = {
                    "mode": "disabled" if explicit_selection == "none" else "forced",
                    "evaluate_cars": explicit_selection == "cars",
                    "reason_code": "explicit_selection_method",
                    "reason": f"The caller explicitly selected wavelength_selection={explicit_selection!r}.",
                    "signals": ["explicit_request"],
                    "selected_method": explicit_selection,
                    "adoption": {"reason_code": "explicit_selection_honored"},
                }

        budget.checkpoint("analyze_model_selection")
        chosen_model, model_selection_decision, model_candidate_results = _select_model_family(
            requested_model,
            X_tr,
            y_tr,
            X_val,
            y_val,
            max_components=10,
            cv_folds=5,
            max_svr_samples=int(max_svr_samples),
            max_tree_samples=int(max_tree_samples),
            min_relative_improvement=float(min_model_improvement),
        )
        selected_method = chosen_model["method"]
        model = chosen_model["_model"]
        best_n = chosen_model["n_components"]
        cv_results = chosen_model["cv_results"]
        predict_fn = chosen_model["_predict_fn"]
        y_pred_te = predict_fn(model, X_te)
        y_pred_val = chosen_model["_tuning_prediction"]
        y_pred_tr = predict_fn(model, X_tr)

        training_data_hash = _sha256_file(real_in)
        metrics = {
            "training_data_hash": training_data_hash,
            "scientific_validation": {
                "schema_version": 1,
                "passed": True,
                "dataset": dataset_validation,
                "partition_separation": split_validation,
            },
            "reproducibility": reproducibility_evidence(
                random_state=42,
                protocol="automated_analysis_three_way_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "test_ratio": 0.20,
                    "val_ratio": 0.10,
                    "method": method,
                    "auto_preprocess": bool(auto_preprocess),
                    "wavelength_selection": wavelength_selection,
                    "min_model_improvement": float(min_model_improvement),
                },
            ),
            "protocol": "automated_analysis_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "method": selected_method,
            "n_components": best_n,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "n_wavelengths_original": int(wavelength_selection_meta["n_original"]),
            "n_wavelengths_model": int(wavelength_selection_meta["n_selected"]),
            "auto_preprocess": auto_preprocess,
            "preprocessing": preprocessing_desc,
            "wavelength_selection": wavelength_selection_meta,
            "wavelength_selection_decision": wavelength_selection_decision,
            "wavelength_selection_candidates": wavelength_selection_candidates,
            "model_selection_decision": model_selection_decision,
            "model_candidates": _model_candidate_summary(model_candidate_results),
            "train": compute_metrics(y_tr, y_pred_tr),
            "val": compute_metrics(y_val, y_pred_val),
            "test": compute_metrics(y_te, y_pred_te),
            "cv_results": cv_results,
        }
        metrics["R2_val"] = metrics["val"]["R2"]
        metrics["RPD"] = metrics["test"]["RPD"]
        metrics["RMSEP"] = metrics["test"]["RMSE"]
        quality = evaluate_quality(metrics, domain=domain, n_samples=int(X.shape[0]))
        metrics["quality"] = quality

        try:
            from nir_core.diagnostics import compute_residual_diagnostics

            metrics["diagnostics"] = compute_residual_diagnostics(y_val, y_pred_val)
        except Exception as exc:
            warnings.warn(f"Residual diagnostics failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

        # Save model + metrics + plots + report.
        model_path = os.path.join(real_outdir, "model.pkl")
        metrics_path = os.path.join(real_outdir, "metrics.json")
        raw_spectra_path = os.path.join(real_outdir, "raw_spectra.png")
        pred_path = os.path.join(real_outdir, "predicted_vs_reference.png")
        resid_path = os.path.join(real_outdir, "residuals.png")
        cv_path = os.path.join(real_outdir, "cv_curve.png")
        report_path = os.path.join(real_outdir, "report.md")

        _write_trusted_model_artifact(
            _build_model_artifact(
                model,
                method=selected_method,
                preprocessing_pipeline=best_pipe,
                preprocessing_desc=preprocessing_desc,
                wavelength_selection=wavelength_selection_meta,
                X_reference=X_tr,
            ),
            model_path,
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(
            model_path,
            metrics_path,
            training_data_hash=training_data_hash,
        )

        import base64

        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra

        raw_b64 = plot_raw_spectra(data, n_highlight=5)
        pred_b64 = plot_predicted_vs_reference(y_te, y_pred_te, title="Predicted vs Reference (test set)")
        resid_b64 = plot_residuals(y_te, y_pred_te)

        # CV curve only for methods that have cv_results with n_components.
        cv_b64 = ""
        if isinstance(cv_results, dict) and cv_results.get("n_components"):
            cv_b64 = plot_cv_curve(
                cv_results["n_components"],
                cv_results["mean_rmse_cv"],
                best_n,
                cv_results.get("std_rmse_cv"),
            )

        for path, b64 in [
            (raw_spectra_path, raw_b64),
            (pred_path, pred_b64),
            (resid_path, resid_b64),
            (cv_path, cv_b64),
        ]:
            if b64:
                with open(path, "wb") as f:
                    f.write(base64.b64decode(b64))

        # Markdown report (with embedded images).
        report = _build_report(
            data,
            metrics,
            quality,
            best_pipe,
            raw_spectra_b64=raw_b64,
            predicted_vs_reference_b64=pred_b64,
            residuals_b64=resid_b64,
            cv_curve_b64=cv_b64,
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # ★ Knowledge-base hint (same trigger policy as nir_train_model).
        knowledge_hint = _build_knowledge_hint(
            domain=domain,
            grade=quality.get("grade"),
            passed=quality.get("passed"),
            r2_val=metrics.get("R2_val"),
            diagnostics=metrics.get("diagnostics"),
        )

        return _ok(
            {
                "status": "ok",
                "protocol": "automated_analysis_three_way_holdout",
                "validation_scope": "independent_holdout_not_external",
                "method": selected_method,
                "n_components": best_n,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": wavelength_selection_meta,
                "wavelength_selection_decision": wavelength_selection_decision,
                "wavelength_selection_candidates": _wavelength_candidate_summary(wavelength_selection_candidates),
                "model_selection_decision": model_selection_decision,
                "model_candidates": _model_candidate_summary(model_candidate_results),
                "R2_val": round(metrics["R2_val"], 4),
                "RPD": round(metrics["RPD"], 4),
                "RMSEP": round(metrics["RMSEP"], 4),
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "thresholds_used": quality["thresholds_used"],
                "output_dir": output_dir,
                "report": output_dir + "/report.md",
                "model": output_dir + "/model.pkl",
                "metrics": output_dir + "/metrics.json",
                "evidence": _model_evidence(
                    model_path,
                    metrics_path,
                    training_data_hash=training_data_hash,
                    protocol="automated_analysis_three_way_holdout",
                    validation_scope="independent_holdout_not_external",
                ),
                "resource_budget": budget.evidence(),
                "knowledge_hint": knowledge_hint,
                "plots": {
                    "raw_spectra": output_dir + "/raw_spectra.png",
                    "predicted_vs_reference": output_dir + "/predicted_vs_reference.png",
                    "residuals": output_dir + "/residuals.png",
                    "cv_curve": output_dir + "/cv_curve.png" if cv_b64 else None,
                },
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_analyze_collection
# ---------------------------------------------------------------------------


def _collection_subset_names(real_path: str, subsets: str) -> list[str]:
    """Resolve explicit or auto-detected MAT subset names with stable ordering."""
    requested = subsets.strip()
    if requested.lower() == "auto":
        info = json.loads(inspect_file(real_path))
        values = info.get("available_subsets")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError("MAT collection analysis requires at least two available_subsets")
    else:
        parsed = json.loads(requested)
        if not isinstance(parsed, list):
            raise ValueError('subsets must be "auto" or a JSON array of subset names')
        values = parsed

    names: list[str] = []
    for value in values:
        name = str(value).strip()
        if not name or name in names:
            continue
        names.append(name)
    if len(names) < 2:
        raise ValueError("At least two unique MAT subset names are required")
    if len(names) > 20:
        raise ValueError("At most 20 MAT subsets can be analyzed in one collection run")
    return names


def _safe_subset_dir(name: str, index: int) -> str:
    """Return a path-safe, deterministic directory name for a MAT subset."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or f"subset_{index:02d}"


def _analyze_collection_item(**kwargs) -> str:
    """Invoke the single-dataset implementation without another agent turn."""
    analyze = nir_analyze_tool.func
    if analyze is None:  # pragma: no cover - defensive LangChain compatibility
        raise RuntimeError("nir_analyze implementation is unavailable")
    return analyze(**kwargs)


def _compact_collection_result(subset: str, payload: dict) -> dict:
    """Keep only decision-grade metrics and artifact paths in model context."""
    keys = (
        "method",
        "n_components",
        "preprocessing",
        "wavelength_selection",
        "wavelength_selection_decision",
        "wavelength_selection_candidates",
        "model_selection_decision",
        "model_candidates",
        "protocol",
        "validation_scope",
        "evidence",
        "R2_val",
        "RPD",
        "RMSEP",
        "grade",
        "passed",
        "action",
        "report",
        "model",
        "metrics",
        "plots",
    )
    return {"subset": subset, **{key: payload.get(key) for key in keys if payload.get(key) is not None}}


def _collection_summary_markdown(results: list[dict], failures: list[dict]) -> str:
    """Build a small human-readable collection summary without plot payloads."""
    lines = [
        "# NIR MAT 多子数据集汇总",
        "",
        "> 完整报告、指标、模型和图表保存在各子目录；本文件仅保留汇总指标。",
        "",
        "| 子数据集 | 方法 | 成分数 | R²(验证) | RPD | RMSEP | 等级 | 通过 |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in results:
        subset = str(item["subset"]).replace("|", "\\|")
        lines.append(
            f"| {subset} | {item.get('method', '-')} | {item.get('n_components', '-')} | {item.get('R2_val', '-')} | {item.get('RPD', '-')} | {item.get('RMSEP', '-')} | {item.get('grade', '-')} | {'是' if item.get('passed') else '否'} |"
        )
    if failures:
        lines.extend(["", "## 未完成的子数据集"])
        for item in failures:
            lines.append(f"- `{item['subset']}`: {item['error']}")
    lines.append("")
    return "\n".join(lines)


@tool("nir_analyze_collection", parse_docstring=True)
def nir_analyze_collection_tool(
    runtime: Runtime,
    data_path: str,
    subsets: str = "auto",
    auto_preprocess: bool = True,
    method: str = "auto",
    wavelength_selection: str = "auto",
    wavelength_selection_params: str | None = None,
    max_svr_samples: int = 1500,
    max_tree_samples: int = 2500,
    min_model_improvement: float = 0.01,
    output_dir: str = "/mnt/user-data/outputs/nir_collection",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Analyze every independent sub-dataset in a MATLAB collection in one tool call.

    The computation remains sequential for predictable memory use, while all
    intermediate orchestration stays inside the tool. The agent receives only
    compact per-subset metrics and artifact paths, never full report bodies.

    Args:
        data_path: Virtual path to a MATLAB ``.mat`` file containing nested
            sub-datasets.
        subsets: ``"auto"`` to use all inspected ``available_subsets``, or a
            JSON array of explicit subset names.
        auto_preprocess: Whether each subset runs nested-CV preprocessing
            selection.
        method: Modeling mode passed to ``nir_analyze``; defaults to autonomous
            model-family selection.
        wavelength_selection: Train-only selection mode passed to
            ``nir_analyze``; defaults to autonomous full-spectrum/CARS
            evaluation for PLS.
        wavelength_selection_params: Optional JSON object for selection params.
        max_svr_samples: Per-subset autonomous SVR sample limit.
        max_tree_samples: Per-subset autonomous Extra Trees sample limit.
        min_model_improvement: Relative tuning improvement required to replace
            PLS with another model family.
        output_dir: Virtual base directory. Each subset receives its own child
            directory, plus one compact ``collection_summary.md``.
        domain: Application domain used for quality thresholds.

    Returns:
        Compact JSON with per-subset metrics, a primary result, all artifact
        paths, and the collection summary path.
    """
    try:
        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        real_input = _resolve(runtime, data_path, read_only=True)
        subset_names = _collection_subset_names(real_input, subsets)
        real_output_dir = _resolve_writable_dir(runtime, output_dir)
        os.makedirs(real_output_dir, exist_ok=True)

        results: list[dict] = []
        failures: list[dict] = []
        for index, subset_name in enumerate(subset_names, start=1):
            child_dir = f"{output_dir.rstrip('/')}/{_safe_subset_dir(subset_name, index)}"
            raw = _analyze_collection_item(
                runtime=runtime,
                data_path=data_path,
                subset=subset_name,
                auto_preprocess=auto_preprocess,
                method=method,
                wavelength_selection=wavelength_selection,
                wavelength_selection_params=wavelength_selection_params,
                max_svr_samples=max_svr_samples,
                max_tree_samples=max_tree_samples,
                min_model_improvement=min_model_improvement,
                output_dir=child_dir,
                domain=domain,
                tool_call_id="",
            )
            payload = json.loads(raw)
            if payload.get("status") == "ok":
                results.append(_compact_collection_result(subset_name, payload))
            else:
                failures.append({"subset": subset_name, "error": str(payload.get("error", "analysis failed"))[:500]})

        if not results:
            return _err(f"All MAT sub-dataset analyses failed: {failures}")

        primary = max(results, key=lambda item: float(item.get("R2_val", float("-inf"))))
        summary_virtual = f"{output_dir.rstrip('/')}/collection_summary.md"
        summary_real = os.path.join(real_output_dir, "collection_summary.md")
        with open(summary_real, "w", encoding="utf-8") as handle:
            handle.write(_collection_summary_markdown(results, failures))

        artifacts: list[str] = [summary_virtual]
        for item in results:
            for key in ("report", "metrics", "model"):
                value = item.get(key)
                if isinstance(value, str):
                    artifacts.append(value)
            plots = item.get("plots")
            if isinstance(plots, dict):
                artifacts.extend(value for value in plots.values() if isinstance(value, str))

        return _ok(
            {
                "status": "ok",
                "protocol": "mat_collection_sequential_compact",
                "validation_scope": "independent_holdout_not_external",
                "subset_count": len(results),
                "requested_subset_count": len(subset_names),
                "failed_subsets": failures,
                "passed": not failures and all(bool(item.get("passed")) for item in results),
                "grade": primary.get("grade"),
                "primary_subset": primary["subset"],
                "model_path": primary.get("model"),
                "metrics_path": primary.get("metrics"),
                "evidence": primary.get("evidence"),
                "report": summary_virtual,
                "results": results,
                "artifacts": list(dict.fromkeys(artifacts)),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# nir_compare
# ---------------------------------------------------------------------------


@tool("nir_compare", parse_docstring=True)
def nir_compare_tool(
    runtime: Runtime,
    data_path: str,
    pipelines: str,
    method: str = "pls",
    domain: str = "default",
    output_dir: str = "/mnt/user-data/outputs/nir_comparison",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Compare multiple preprocessing pipelines and pick the best model.

    Runs each pipeline sequentially (deterministic, no LLM in the loop):
    load then apply pipeline then train model then evaluate.  After all
    pipelines are evaluated, generates an HTML comparison gallery and a
    Markdown summary, and returns the best pipeline.

    Args:
        data_path: Virtual path to the raw spectral file (.mat/.csv/.txt) or
            an already-standardised .npz.
        pipelines: JSON string: list of pipeline definitions. Each pipeline
            is a list of steps; each step can be a method name string or a
            dict with a method name and optional hyper-parameters dict.
            Use "[]" for raw spectra (no preprocessing).
        method: Modelling method: ``pls`` / ``pcr`` / ``svr`` / ``rf`` /
            ``et`` / ``gbm`` / ``ridge`` / ``lasso`` / ``elasticnet`` /
            ``knn`` / ``mlp`` / ``cnn``.
        domain: Application domain for quality-gate thresholds.
        output_dir: Virtual directory for comparison outputs.

    Returns:
        JSON with the best pipeline, its metrics, and paths to the gallery
        HTML and summary Markdown.
    """
    try:
        import json as _json

        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        from nir_core.io.loaders import auto_detect_and_load
        from nir_core.model.evaluation import compute_metrics, split_dataset
        from nir_core.plotting.gallery import generate_comparison_gallery
        from nir_core.utils.metrics import evaluate_quality

        try:
            from nir_core.models import ModelResult
            from nir_core.preprocess.pipeline import PreprocessingPipeline
        except ImportError:
            return _err("nir_core V3 features (PreprocessingPipeline) are not available in the current sandbox. Please rebuild the Docker image to refresh nir_core.")

        # Parse pipelines.
        try:
            pipe_list = _json.loads(pipelines) if isinstance(pipelines, str) else pipelines
        except (ValueError, TypeError):
            return _err(f"Invalid pipelines JSON: {pipelines!r}")

        if not pipe_list:
            pipe_list = [[]]  # raw spectra

        # Load data.
        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, data_path, read_only=True)
        budget.check_file(real_in, stage="compare_file_preflight")
        data = auto_detect_and_load(real_in)
        check_spectral_data(budget, data, stage="compare_matrix_preflight", peak_multiplier=8.0)
        X = np.asarray(data.X, dtype=float)
        y = np.asarray(data.y, dtype=float).ravel() if data.y is not None else None
        if y is None:
            return _err("Data has no reference values (y); cannot train models.")
        data_wv = np.asarray(data.wv, dtype=float).ravel() if data.wv is not None else None
        dataset_validation = dataset_science_gate(X, y, data_wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)

        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_dataset(
            X,
            y,
            test_ratio=0.20,
            val_ratio=0.10,
            random_state=42,
        )
        split_validation = split_science_gate(
            calibration=X_tr,
            tuning=X_val,
            holdout=X_te,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)
        training_data_hash = _sha256_file(real_in)
        comparison_reproducibility = reproducibility_evidence(
            random_state=42,
            protocol="preprocessing_comparison_three_way_holdout",
            input_sha256=training_data_hash,
            parameters={
                "test_ratio": 0.20,
                "val_ratio": 0.10,
                "method": method,
                "pipeline_count": len(pipe_list),
            },
        )

        results = []
        summaries = []
        best_idx = 0
        best_rpd = -1.0

        for i, methods in enumerate(pipe_list):
            budget.checkpoint(f"compare_pipeline_{i + 1}")
            # Build and apply pipeline.
            steps = [_parse_pipeline_step(m) for m in methods]
            pipe = PreprocessingPipeline(steps=steps) if steps else None

            try:
                if pipe is not None:
                    pipe.fit(X_tr, data.wv)
                    X_tr_p = pipe.transform(X_tr, data.wv)
                    X_val_p = pipe.transform(X_val, data.wv)
                    X_te_p = pipe.transform(X_te, data.wv)
                    pp_desc = pipe.description()
                else:
                    X_tr_p, X_val_p, X_te_p = X_tr, X_val, X_te
                    pp_desc = "无（原始光谱）"
            except Exception as pp_exc:
                summaries.append(
                    {
                        "pipeline": methods,
                        "description": " → ".join(methods) if methods else "raw",
                        "error": f"{type(pp_exc).__name__}: {pp_exc}",
                    }
                )
                continue

            # Train.
            model, best_n, cv_results, predict_fn = _train_one_model(method, X_tr_p, y_tr, max_components=20, cv_folds=10, cv_strategy="auto")
            y_pred_tr = predict_fn(model, X_tr_p)
            y_pred_val = predict_fn(model, X_val_p)
            y_pred_te = predict_fn(model, X_te_p)

            m = {
                "method": method,
                "n_components": best_n,
                "domain": domain,
                "n_samples": int(X.shape[0]),
                "training_data_hash": training_data_hash,
                "scientific_validation": {
                    "schema_version": 1,
                    "passed": True,
                    "dataset": dataset_validation,
                    "partition_separation": split_validation,
                },
                "reproducibility": comparison_reproducibility,
                "train": compute_metrics(y_tr, y_pred_tr),
                "val": compute_metrics(y_val, y_pred_val),
                "test": compute_metrics(y_te, y_pred_te),
                "preprocessing": pp_desc,
                "pipeline": methods,
            }
            m["R2_val"] = m["val"]["R2"]
            m["RPD"] = m["test"]["RPD"]
            m["RMSEP"] = m["test"]["RMSE"]
            m["RMSEC"] = m["train"]["RMSE"]
            m["RMSECV"] = _extract_rmsecv(cv_results, best_n)
            m["quality"] = evaluate_quality(m, domain=domain, n_samples=int(X.shape[0]))
            try:
                from nir_core.diagnostics import compute_residual_diagnostics

                m["diagnostics"] = compute_residual_diagnostics(y_val, y_pred_val)
            except Exception as exc:
                warnings.warn(f"Residual diagnostics failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

            # Store y_ref/y_pred for gallery plot.
            m_with_arrays = dict(m)
            m_with_arrays["y_ref"] = y_te
            m_with_arrays["y_pred"] = y_pred_te
            results.append(
                ModelResult(
                    method=method,
                    n_components=best_n,
                    metrics=m_with_arrays,
                    preprocessing_steps=steps,
                )
            )

            summary_idx = len(summaries)
            summary = {
                "pipeline": methods,
                "description": pp_desc,
                "n_components": best_n,
                "R2_val": round(m["R2_val"], 4),
                "RPD": round(m["RPD"], 4),
                "RMSEP": round(m["RMSEP"], 4),
                "RMSECV": round(m["RMSECV"], 4) if m["RMSECV"] else None,
                "grade": m["quality"]["grade"],
                "passed": m["quality"]["passed"],
                "diagnostics": m.get("diagnostics"),
            }
            summaries.append(summary)

            if m["RPD"] > best_rpd:
                best_rpd = m["RPD"]
                best_idx = summary_idx

        if not summaries:
            return _err("All pipelines failed; no models to compare.")

        # Generate gallery HTML.
        real_outdir = _resolve_writable_dir(runtime, output_dir)
        html = generate_comparison_gallery(results)
        gallery_path = os.path.join(real_outdir, "comparison_gallery.html")
        with open(gallery_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Markdown summary.
        best = summaries[best_idx]
        md_lines = [
            "# 预处理流水线对比报告",
            "",
            f"共比较 {len(summaries)} 种预处理组合，建模方法: {method}",
            "",
            "## 对比结果",
            "",
            "| # | 预处理 | 成分数 | R²(验证) | RPD | RMSEP | 等级 | 通过 |",
            "|---|--------|--------|----------|-----|-------|------|------|",
        ]
        for i, s in enumerate(summaries):
            mark = " ⭐" if i == best_idx else ""
            md_lines.append(f"| {i + 1} | {s['description']}{mark} | {s.get('n_components', '-')} | {s.get('R2_val', '-')} | {s.get('RPD', '-')} | {s.get('RMSEP', '-')} | {s.get('grade', '-')} | {'是' if s.get('passed') else '否'} |")
        md_lines.extend(
            [
                "",
                "## 最佳组合",
                f"- 预处理: {best['description']}",
                f"- 成分数: {best.get('n_components')}",
                f"- R²(验证): {best.get('R2_val')}",
                f"- RPD: {best.get('RPD')}",
                f"- RMSEP: {best.get('RMSEP')}",
                f"- 等级: {best.get('grade')}",
                "",
            ]
        )
        summary_md = "\n".join(md_lines)
        md_path = os.path.join(real_outdir, "comparison_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(summary_md)

        # Save full metrics JSON.
        all_metrics_path = os.path.join(real_outdir, "all_metrics.json")
        with open(all_metrics_path, "w", encoding="utf-8") as f:
            _json.dump(summaries, f, ensure_ascii=False, indent=2, default=_json_default)

        knowledge_hint = _build_knowledge_hint(
            domain=domain,
            grade=best.get("grade"),
            passed=best.get("passed"),
            r2_val=best.get("R2_val"),
            diagnostics=best.get("diagnostics"),
        )

        return _ok(
            {
                "status": "ok",
                "n_compared": len(summaries),
                "best_index": best_idx,
                "best": best,
                "all_results": summaries,
                "gallery": output_dir + "/comparison_gallery.html",
                "summary": output_dir + "/comparison_summary.md",
                "all_metrics": output_dir + "/all_metrics.json",
                "scientific_validation": {
                    "schema_version": 1,
                    "passed": True,
                    "dataset": dataset_validation,
                    "partition_separation": split_validation,
                },
                "reproducibility": comparison_reproducibility,
                "knowledge_hint": knowledge_hint,
                "resource_budget": budget.evidence(),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
