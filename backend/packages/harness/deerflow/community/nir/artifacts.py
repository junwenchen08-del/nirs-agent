"""Model artifact assembly, diagnostics plots, and report persistence."""

from __future__ import annotations

import os
import warnings

import numpy as np

from ._report import _build_report


def _build_model_artifact(
    model,
    *,
    method: str,
    preprocessing_pipeline,
    preprocessing_desc: str,
    wavelength_selection: dict,
    X_reference: np.ndarray | None = None,
):
    """Return a deployable model artifact with a compact monitoring reference."""
    monitoring_reference = None
    if X_reference is not None:
        reference_array = np.asarray(X_reference)
        if reference_array.ndim == 2 and reference_array.shape[0] >= 3:
            from nir_core.utils.drift import fit_monitoring_reference

            monitoring_reference = fit_monitoring_reference(reference_array)
    if preprocessing_pipeline is None and wavelength_selection.get("method") == "none" and monitoring_reference is None:
        return model
    return {
        "format": "nir_model_artifact",
        "version": 3 if monitoring_reference is not None else 2,
        "model": model,
        "method": method,
        "preprocessing": {
            "description": preprocessing_desc,
            "pipeline": preprocessing_pipeline,
            "apply_on_predict": preprocessing_pipeline is not None,
        },
        "wavelength_selection": wavelength_selection,
        "monitoring_reference": monitoring_reference,
    }


def _write_plots_and_report(
    *,
    out_dir: str,
    spec_data,
    metrics: dict,
    quality: dict,
    best_pipe,
    y_te,
    y_pred_te,
    cv_results,
    best_n,
    method: str,
    model,
    X_tr,
    y_tr,
    wv,
) -> tuple[str, str, str, str, str, str]:
    """Generate plots + Markdown report under ``out_dir``.

    Returns ``(raw_b64, pred_b64, resid_b64, cv_b64, vip_b64, coef_b64)``
    so the caller can decide which paths to surface in the JSON payload.
    """
    import base64

    from nir_core.plotting.model_diag import (
        plot_cv_curve,
        plot_predicted_vs_reference,
        plot_residuals,
    )
    from nir_core.plotting.spectra import plot_raw_spectra

    pred_b64 = plot_predicted_vs_reference(y_te, y_pred_te)
    resid_b64 = plot_residuals(y_te, y_pred_te)
    raw_b64 = plot_raw_spectra(spec_data, n_highlight=5)

    cv_b64 = ""
    if isinstance(cv_results, dict) and cv_results.get("n_components"):
        cv_b64 = plot_cv_curve(
            cv_results["n_components"],
            cv_results["mean_rmse_cv"],
            best_n,
            cv_results.get("std_rmse_cv"),
        )

    for fname, b64 in [
        ("predicted_vs_reference.png", pred_b64),
        ("residuals.png", resid_b64),
        ("raw_spectra.png", raw_b64),
        ("cv_curve.png", cv_b64),
    ]:
        if b64:
            with open(os.path.join(out_dir, fname), "wb") as f:
                f.write(base64.b64decode(b64))

    # ★ v3: VIP and regression coefficient plots (PLS only).
    vip_b64 = ""
    coef_b64 = ""
    if method == "pls":
        try:
            from nir_core.model.pls import compute_vip, get_regression_coefficients
            from nir_core.plotting.model_diag import plot_regression_coefficients, plot_vip

            vip_scores = compute_vip(model, X_tr, y_tr)
            coef = get_regression_coefficients(model)
            vip_b64 = plot_vip(vip_scores, wv=wv)
            coef_b64 = plot_regression_coefficients(coef, wv=wv)
            for fname, b64 in [("vip_scores.png", vip_b64), ("regression_coefficients.png", coef_b64)]:
                if b64:
                    with open(os.path.join(out_dir, fname), "wb") as f:
                        f.write(base64.b64decode(b64))
        except Exception as exc:
            warnings.warn(f"VIP plot generation failed: {type(exc).__name__}: {exc}", RuntimeWarning, stacklevel=2)

    report = _build_report(
        spec_data,
        metrics,
        quality,
        best_pipe,
        raw_spectra_b64=raw_b64,
        predicted_vs_reference_b64=pred_b64,
        residuals_b64=resid_b64,
        cv_curve_b64=cv_b64,
    )
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return raw_b64, pred_b64, resid_b64, cv_b64, vip_b64, coef_b64
