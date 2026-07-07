"""Calibration model module for nir_core.

Contains:
- PLS / PCR / SVR trainers and predictors.
- Ensemble aggregation utilities.
- Dataset splitting, cross-validation, and nested-CV preprocessing
  selection (evaluation layer).
- Wavelength selection algorithms (CARS, SPA).
"""

from __future__ import annotations

__all__: list[str] = []
