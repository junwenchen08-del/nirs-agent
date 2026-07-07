"""Plotting utilities for nir_core.

Provides visualization helpers for spectral data, model diagnostics, and
comparison galleries. All plot functions return base64-encoded PNG strings
(or HTML strings for galleries) so they can be embedded in reports or
served over HTTP without writing to disk.

Submodules:
- spectra: Raw and preprocessed spectral plots.
- model_diag: Model diagnostic plots (predicted vs reference, residuals, drift).
- gallery: Multi-model comparison HTML gallery.
"""

from __future__ import annotations
