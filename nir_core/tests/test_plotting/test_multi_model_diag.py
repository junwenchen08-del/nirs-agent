"""Multi-component diagnostic plotting tests."""

from __future__ import annotations

import base64

import numpy as np

from nir_core.plotting.model_diag import (
    plot_multi_predicted_vs_reference,
    plot_multi_residuals,
)


def test_multi_component_plots_return_png_payloads() -> None:
    y_ref = np.column_stack(
        (
            np.arange(12, dtype=float),
            np.arange(12, dtype=float) * 2,
            np.arange(12, dtype=float) * -1,
        )
    )
    y_pred = y_ref + np.array([0.1, -0.2, 0.3])

    for payload in (
        plot_multi_predicted_vs_reference(
            y_ref, y_pred, ["protein", "moisture", "oil"]
        ),
        plot_multi_residuals(y_ref, y_pred, ["protein", "moisture", "oil"]),
    ):
        assert base64.b64decode(payload).startswith(b"\x89PNG")
