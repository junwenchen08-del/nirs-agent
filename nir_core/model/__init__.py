"""Calibration model module for nir_core.

Contains:
- Classic chemometric trainers: PLS / PCR / SVR.
- Machine learning trainers: Random Forest / Extra Trees / Gradient
  Boosting / Ridge / Lasso / ElasticNet / KNN.
- Deep learning trainers: MLP (sklearn) / 1D-CNN (PyTorch, optional).
- Ensemble aggregation utilities.
- Dataset splitting, cross-validation, and nested-CV preprocessing
  selection (evaluation layer).
- Wavelength selection algorithms (CARS, SPA).
"""

from __future__ import annotations

__all__: list[str] = []
