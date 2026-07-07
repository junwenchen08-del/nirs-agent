"""Tests for nir_core.plotting.gallery."""

from __future__ import annotations

import numpy as np

from nir_core.models import ModelResult
from nir_core.plotting.gallery import generate_comparison_gallery


def test_gallery_empty_results_returns_html() -> None:
    """Empty results list should return a valid HTML document."""
    html = generate_comparison_gallery([])
    assert isinstance(html, str)
    assert "<html" in html.lower()
    assert "</html>" in html.lower()


def test_gallery_with_model_results_contains_table_and_metrics() -> None:
    """Gallery with ModelResult list should contain a table and metric values."""
    results = [
        ModelResult(
            method="PLS",
            n_components=5,
            metrics={"RMSEP": 0.1234, "R2": 0.9512, "RPD": 4.56},
        ),
        ModelResult(
            method="SVM",
            n_components=None,
            metrics={"RMSEP": 0.2345, "R2": 0.9012, "RPD": 3.21},
        ),
    ]
    html = generate_comparison_gallery(results)
    assert "<html" in html.lower()
    assert "<table>" in html.lower()
    assert "PLS" in html
    assert "SVM" in html
    assert "0.1234" in html
    assert "0.9512" in html
    assert "4.5600" in html
    # Section for each model.
    assert html.count("<section>") == 2


def test_gallery_with_prediction_data_embeds_image() -> None:
    """When y_ref/y_pred are in metrics, gallery should embed a base64 PNG."""
    rng = np.random.default_rng(0)
    y_ref = np.linspace(0, 10, 40)
    y_pred = y_ref + rng.normal(0.0, 0.2, size=y_ref.size)
    results = [
        ModelResult(
            method="PLS",
            n_components=3,
            metrics={
                "RMSEP": 0.2,
                "R2": 0.95,
                "RPD": 4.5,
                "y_ref": y_ref,
                "y_pred": y_pred,
            },
        ),
    ]
    html = generate_comparison_gallery(results)
    assert "data:image/png;base64," in html


def test_gallery_handles_missing_metrics_gracefully() -> None:
    """Gallery should not crash when metrics dict is missing keys."""
    results = [
        ModelResult(method="PCR", n_components=4, metrics={}),
    ]
    html = generate_comparison_gallery(results)
    assert "<html" in html.lower()
    assert "PCR" in html
