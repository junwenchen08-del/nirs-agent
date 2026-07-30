"""Multi-target preprocessing metadata and report helpers."""

from __future__ import annotations

import os
import sys
import warnings
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.resources import ResourceLimitError

from deerflow.tools.types import Runtime

from ._common import (
    _bind_model_metrics,
    _err,
    _json_default,
    _load_npz_safely,
    _model_runtime_preflight_error,
    _ok,
    _parse_pipeline_step,
    _resolve,
    _sha256_file,
    _write_trusted_model_artifact,
)
from ._knowledge_hint import _build_knowledge_hint
from ._resources import budget_for_runtime, resource_error
from ._science_gate import (
    dataset_science_gate,
    reproducibility_evidence,
    science_gate_error,
    split_science_gate,
)
from .single_target import (
    _apply_wavelength_selection,
    _extract_rmsecv,
    _train_one_model,
)


def _resolve_multi_path(runtime: Runtime, path: str, *, read_only: bool) -> str:
    """Honor the historical modeling-facade resolver for compatible callers."""
    facade = sys.modules.get("deerflow.community.nir.modeling")
    resolver = getattr(facade, "_resolve", _resolve) if facade is not None else _resolve
    return resolver(runtime, path, read_only=read_only)


def _parse_multi_component_names(data_dict: dict, component_names: str | None, n_targets: int) -> list[str]:
    """Resolve target names from an override, NPZ metadata, or stable defaults."""
    import json

    if component_names is not None:
        parsed = json.loads(component_names) if isinstance(component_names, str) else component_names
        if not isinstance(parsed, list):
            raise ValueError("component_names must be a JSON list")
        names = [str(name).strip() for name in parsed]
    elif data_dict.get("y_names") is not None and np.asarray(data_dict["y_names"]).size:
        names = [str(name).strip() for name in np.asarray(data_dict["y_names"]).ravel()]
    else:
        names = [f"y{i}" for i in range(n_targets)]
    if len(names) != n_targets:
        raise ValueError(f"component_names length ({len(names)}) != y target count ({n_targets})")
    if any(not name for name in names):
        raise ValueError("component_names must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("component_names must be unique")
    return names


def _parse_multi_pipeline(pipeline_steps: str | None) -> tuple[list, list]:
    """Parse and validate a multi-component preprocessing specification."""
    import json

    if pipeline_steps is None:
        return [], []
    try:
        raw_steps = json.loads(pipeline_steps) if isinstance(pipeline_steps, str) else pipeline_steps
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pipeline_steps JSON: {pipeline_steps!r}") from exc
    if not isinstance(raw_steps, list):
        raise ValueError("pipeline_steps must be a JSON list")
    steps = [_parse_pipeline_step(item) for item in raw_steps]
    from nir_core.preprocess.pipeline import validate_pipeline

    is_valid, reason = validate_pipeline(steps)
    if not is_valid:
        raise ValueError(f"Invalid pipeline: {reason}")
    return raw_steps, steps


def _fit_multi_pipeline(steps: list, X_train: np.ndarray, wv: np.ndarray | None):
    """Fit one configured pipeline on the training spectra only."""
    if not steps:
        return None, "none"
    from nir_core.preprocess.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline(steps=steps)
    pipeline.fit(X_train, wv)
    return pipeline, pipeline.description()


def _multi_preprocessing_artifact(pipeline, description: str, *, shared: bool) -> dict:
    return {
        "shared": shared,
        "description": description,
        "pipeline": pipeline,
        "apply_on_predict": pipeline is not None,
    }


def _write_multi_report(out_dir: str, metrics: dict) -> str:
    """Write a compact comparison report for all target models."""
    lines = [
        "# Multi-component NIR calibration report",
        "",
        f"Method: `{metrics['method']}`  ",
        f"Targets: {metrics['n_targets']}  ",
        f"Shared preprocessing: {metrics['shared_preprocessing']}",
        "",
        "| Component | R2 val | RPD | RMSEP | Grade | Passed |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for item in metrics["per_component"]:
        lines.append(f"| {item['name']} | {item['R2_val']:.4f} | {item['RPD']:.4f} | {item['RMSEP']:.4f} | {item['quality']['grade']} | {item['quality']['passed']} |")
    overall = metrics["overall"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"Mean validation R2: {overall['mean_R2_val']:.4f}  ",
            f"Mean RPD: {overall['mean_RPD']:.4f}  ",
            f"Passed: {overall['n_passed']}/{overall['n_total']}",
            "",
            "Each component's diagnostic plots are stored in its numbered subdirectory.",
        ]
    )
    report_path = os.path.join(out_dir, "multi_report.md")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


@tool("nir_train_multi_model", parse_docstring=True)
def nir_train_multi_model_tool(
    runtime: Runtime,
    input_path: str,
    method: str = "pls",
    pipeline_steps: str | None = None,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    max_components: int = 20,
    cv_folds: int = 10,
    cv_strategy: str = "auto",
    wavelength_selection: str = "none",
    wavelength_selection_params: str | None = None,
    shared_preprocessing: bool = True,
    component_names: str | None = None,
    model_output: str = "/mnt/user-data/outputs/multi_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/multi_metrics.json",
    output_dir: str = "/mnt/user-data/outputs/multi_outputs",
    domain: str = "default",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train independent models for multiple reference components in one run.

    All components share one train/validation/test split. Preprocessing is
    fitted on training spectra only and can be shared or fitted separately;
    wavelength selection and model training are always independent per
    component. The saved version-3 artifact can be passed directly to
    ``nir_predict`` for N-by-K predictions.

    Args:
        input_path: Virtual path to an NPZ containing X and two-dimensional y.
        method: Modeling method accepted by nir_train_model.
        pipeline_steps: Optional JSON list of leakage-safe preprocessing steps.
        test_ratio: Fraction reserved for the independent test set.
        val_ratio: Fraction reserved for validation.
        max_components: Upper bound for PLS/PCR component search.
        cv_folds: Number of cross-validation folds.
        cv_strategy: Cross-validation strategy: auto, loocv, or fixed.
        wavelength_selection: Per-component selection: none, cars, spa, or manual.
        wavelength_selection_params: Optional JSON object for wavelength selection.
        shared_preprocessing: Fit one preprocessing pipeline for all components when true.
        component_names: Optional JSON list overriding NPZ y_names.
        model_output: Virtual output path for the version-3 model artifact.
        metrics_output: Virtual output path for multi-component metrics JSON.
        output_dir: Virtual directory for the report and component plots.
        domain: Application domain used by each component quality gate.

    Returns:
        JSON containing per-component metrics, overall pass rate, model,
        metrics and report paths, and retrieval hints for weak components.
    """
    try:
        import base64
        import json

        if runtime_error := _model_runtime_preflight_error(method):
            return runtime_error

        from nir_core.diagnostics import compute_residual_diagnostics
        from nir_core.model.evaluation import compute_metrics, split_dataset
        from nir_core.models import SpectralData
        from nir_core.plotting.model_diag import (
            plot_cv_curve,
            plot_multi_predicted_vs_reference,
            plot_multi_residuals,
            plot_predicted_vs_reference,
            plot_residuals,
        )
        from nir_core.plotting.spectra import plot_raw_spectra
        from nir_core.utils.metrics import evaluate_quality

        budget = budget_for_runtime(runtime)
        real_in = _resolve_multi_path(runtime, input_path, read_only=True)
        budget.check_file(real_in, stage="multi_model_file_preflight")
        data_dict = _load_npz_safely(real_in)
        X = np.asarray(data_dict["X"], dtype=float)
        y = np.asarray(data_dict["y"], dtype=float)
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None
        if y.ndim != 2 or y.shape[1] < 2:
            return _err(f"nir_train_multi_model requires y with shape (n_samples, K>=2), got {y.shape}")
        if X.shape[0] != y.shape[0]:
            return _err(f"X rows ({X.shape[0]}) != y rows ({y.shape[0]})")
        budget.check_matrix_shape(
            X.shape,
            dtype=X.dtype,
            target_count=int(y.shape[1]),
            peak_multiplier=8.0,
            stage="multi_model_matrix_preflight",
        )
        dataset_validation = dataset_science_gate(X, y, wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)

        names = _parse_multi_component_names(data_dict, component_names, y.shape[1])
        raw_steps, steps = _parse_multi_pipeline(pipeline_steps)
        (X_tr_raw, y_tr), (X_val_raw, y_val), (X_te_raw, y_te) = split_dataset(X, y, test_ratio=test_ratio, val_ratio=val_ratio, random_state=42)
        split_validation = split_science_gate(
            calibration=X_tr_raw,
            tuning=X_val_raw,
            holdout=X_te_raw,
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        shared_pipe = None
        shared_desc = "none"
        if shared_preprocessing:
            shared_pipe, shared_desc = _fit_multi_pipeline(steps, X_tr_raw, wv)
            X_tr_shared = shared_pipe.transform(X_tr_raw, wv) if shared_pipe is not None else X_tr_raw
            X_val_shared = shared_pipe.transform(X_val_raw, wv) if shared_pipe is not None else X_val_raw
            X_te_shared = shared_pipe.transform(X_te_raw, wv) if shared_pipe is not None else X_te_raw

        real_model = _resolve_multi_path(runtime, model_output, read_only=False)
        real_metrics = _resolve_multi_path(runtime, metrics_output, read_only=False)
        real_output_dir = _resolve_multi_path(runtime, output_dir, read_only=False)
        os.makedirs(os.path.dirname(real_model), exist_ok=True)
        os.makedirs(os.path.dirname(real_metrics), exist_ok=True)
        os.makedirs(real_output_dir, exist_ok=True)

        models: list = []
        selections: list[dict] = []
        preprocessing_artifacts: list[dict] = []
        monitoring_references: list[dict | None] = []
        per_component: list[dict] = []
        knowledge_hints: list[dict] = []
        test_predictions: list[np.ndarray] = []

        try:
            from nir_core.model.pls import compute_vip, get_regression_coefficients, get_regression_intercept
        except ImportError:
            compute_vip = get_regression_coefficients = get_regression_intercept = None

        raw_plot = plot_raw_spectra(SpectralData(X=X, y=y, y_names=names, wv=wv), n_highlight=5)
        with open(os.path.join(real_output_dir, "raw_spectra.png"), "wb") as file:
            file.write(base64.b64decode(raw_plot))

        for index, name in enumerate(names):
            budget.checkpoint(f"multi_model_component_{index + 1}")
            if shared_preprocessing:
                pipeline = shared_pipe
                preprocessing_desc = shared_desc
                X_tr_base, X_val_base, X_te_base = X_tr_shared, X_val_shared, X_te_shared
                preprocessing_artifacts.append(_multi_preprocessing_artifact(shared_pipe, shared_desc, shared=True))
            else:
                pipeline, preprocessing_desc = _fit_multi_pipeline(steps, X_tr_raw, wv)
                X_tr_base = pipeline.transform(X_tr_raw, wv) if pipeline is not None else X_tr_raw
                X_val_base = pipeline.transform(X_val_raw, wv) if pipeline is not None else X_val_raw
                X_te_base = pipeline.transform(X_te_raw, wv) if pipeline is not None else X_te_raw
                preprocessing_artifacts.append(_multi_preprocessing_artifact(pipeline, preprocessing_desc, shared=False))

            X_tr, X_val, X_te, model_wv, selection = _apply_wavelength_selection(
                X_tr_base,
                X_val_base,
                X_te_base,
                y_tr[:, index],
                wv=wv,
                wavelength_selection=wavelength_selection,
                wavelength_selection_params=wavelength_selection_params,
                cv_folds=cv_folds,
            )
            model, best_n, cv_results, predict_fn = _train_one_model(
                method,
                X_tr,
                y_tr[:, index],
                max_components=max_components,
                cv_folds=cv_folds,
                cv_strategy=cv_strategy,
            )
            pred_tr = np.asarray(predict_fn(model, X_tr), dtype=float).ravel()
            pred_val = np.asarray(predict_fn(model, X_val), dtype=float).ravel()
            pred_te = np.asarray(predict_fn(model, X_te), dtype=float).ravel()
            if X_tr.shape[0] >= 3:
                from nir_core.utils.drift import fit_monitoring_reference

                monitoring_references.append(fit_monitoring_reference(X_tr))
            else:
                monitoring_references.append(None)

            item = {
                "name": name,
                "method": method,
                "n_components": best_n,
                "train": compute_metrics(y_tr[:, index], pred_tr),
                "val": compute_metrics(y_val[:, index], pred_val),
                "test": compute_metrics(y_te[:, index], pred_te),
                "cv_results": cv_results,
                "preprocessing": preprocessing_desc,
                "wavelength_selection": selection,
            }
            item["R2_val"] = item["val"]["R2"]
            item["RPD"] = item["test"]["RPD"]
            item["RMSEC"] = item["train"]["RMSE"]
            item["RMSECV"] = _extract_rmsecv(cv_results, best_n)
            item["RMSEP"] = item["test"]["RMSE"]
            item["diagnostics"] = compute_residual_diagnostics(y_val[:, index], pred_val)

            if method == "pls" and compute_vip is not None:
                try:
                    vip_scores = compute_vip(model, X_tr, y_tr[:, index])
                    top_local = [int(i) for i in np.argsort(vip_scores)[-10:][::-1]]
                    selected_indices = selection.get("selected_indices") or list(range(len(vip_scores)))
                    item["vip_summary"] = {
                        "max": float(np.max(vip_scores)),
                        "mean": float(np.mean(vip_scores)),
                        "n_above_1": int(np.sum(vip_scores > 1.0)),
                        "top_10_indices": [int(selected_indices[i]) for i in top_local],
                    }
                    coefficients = get_regression_coefficients(model)
                    item["coef_summary"] = {
                        "max_abs": float(np.max(np.abs(coefficients))),
                        "mean_abs": float(np.mean(np.abs(coefficients))),
                        "intercept": get_regression_intercept(model),
                    }
                except Exception as exc:
                    warnings.warn(f"Interpretability failed for {name}: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

            quality = evaluate_quality(item, domain=domain, n_samples=int(X.shape[0]))
            item["quality"] = quality
            hint = _build_knowledge_hint(
                domain=domain,
                grade=quality.get("grade"),
                passed=quality.get("passed"),
                r2_val=item["R2_val"],
                diagnostics=item["diagnostics"],
            )
            if hint is not None:
                knowledge_hints.append({"name": name, **hint})

            component_dir = os.path.join(real_output_dir, f"component_{index + 1:02d}")
            os.makedirs(component_dir, exist_ok=True)
            plot_payloads = {
                "predicted_vs_reference.png": plot_predicted_vs_reference(y_te[:, index], pred_te, title=name),
                "residuals.png": plot_residuals(y_te[:, index], pred_te),
            }
            if isinstance(cv_results, dict) and cv_results.get("n_components"):
                plot_payloads["cv_curve.png"] = plot_cv_curve(cv_results["n_components"], cv_results["mean_rmse_cv"], best_n, cv_results.get("std_rmse_cv"))
            for filename, encoded in plot_payloads.items():
                with open(os.path.join(component_dir, filename), "wb") as file:
                    file.write(base64.b64decode(encoded))
            item["plots"] = {key.removesuffix(".png"): f"{output_dir}/component_{index + 1:02d}/{key}" for key in plot_payloads}

            models.append(model)
            selections.append(selection)
            per_component.append(item)
            test_predictions.append(pred_te)

        prediction_matrix = np.column_stack(test_predictions)
        combined_plots = {
            "multi_predicted_vs_reference.png": plot_multi_predicted_vs_reference(y_te, prediction_matrix, names),
            "multi_residuals.png": plot_multi_residuals(y_te, prediction_matrix, names),
        }
        for filename, encoded in combined_plots.items():
            with open(os.path.join(real_output_dir, filename), "wb") as file:
                file.write(base64.b64decode(encoded))

        preprocessing_artifact = preprocessing_artifacts
        overall = {
            "mean_R2_val": float(np.mean([item["R2_val"] for item in per_component])),
            "mean_RPD": float(np.mean([item["RPD"] for item in per_component])),
            "n_passed": int(sum(bool(item["quality"]["passed"]) for item in per_component)),
            "n_total": len(per_component),
        }
        overall["passed"] = overall["n_passed"] == overall["n_total"]
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
                protocol="multi_target_random_three_way_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "test_ratio": float(test_ratio),
                    "val_ratio": float(val_ratio),
                    "cv_folds": int(cv_folds),
                    "cv_strategy": cv_strategy,
                    "method": method,
                    "max_components": int(max_components),
                    "wavelength_selection": wavelength_selection,
                    "shared_preprocessing": bool(shared_preprocessing),
                },
            ),
            "method": method,
            "n_targets": len(names),
            "component_names": names,
            "domain": domain,
            "n_samples": int(X.shape[0]),
            "shared_preprocessing": shared_preprocessing,
            "preprocessing": shared_desc if shared_preprocessing else [item["preprocessing"] for item in per_component],
            "preprocessing_steps": raw_steps,
            "per_component": per_component,
            "overall": overall,
            "plots": {
                "predicted_vs_reference": f"{output_dir}/multi_predicted_vs_reference.png",
                "residuals": f"{output_dir}/multi_residuals.png",
                "raw_spectra": f"{output_dir}/raw_spectra.png",
            },
        }

        artifact = {
            "format": "nir_model_artifact",
            "version": 3,
            "multi_output": True,
            "n_targets": len(names),
            "component_names": names,
            "models": models,
            "method": method,
            "preprocessing": preprocessing_artifact,
            "wavelength_selection": selections,
            "monitoring_reference": monitoring_references,
        }
        _write_trusted_model_artifact(artifact, real_model)
        with open(real_metrics, "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(
            real_model,
            real_metrics,
            training_data_hash=training_data_hash,
        )
        _write_multi_report(real_output_dir, metrics)

        concise_components = [
            {
                "name": item["name"],
                "R2_val": round(item["R2_val"], 4),
                "RPD": round(item["RPD"], 4),
                "RMSEP": round(item["RMSEP"], 4),
                "grade": item["quality"]["grade"],
                "passed": item["quality"]["passed"],
            }
            for item in per_component
        ]
        return _ok(
            {
                "status": "ok",
                "method": method,
                "n_targets": len(names),
                "component_names": names,
                "shared_preprocessing": shared_preprocessing,
                "per_component": concise_components,
                "overall": overall,
                "grade": "good" if overall["passed"] else "fair",
                "passed": overall["passed"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "report": f"{output_dir}/multi_report.md",
                "plots": metrics["plots"],
                "knowledge_hint": knowledge_hints or None,
                "resource_budget": budget.evidence(),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except MemoryError:
        return _err("内存不足: 多成分训练数据过大。请减少成分、波长或样本数量后重试。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
