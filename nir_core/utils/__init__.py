"""nir_core.utils: Utility functions for metrics, validation, drift, and registry.

This package contains pure helper functions and the ModelRegistry class
used by the NIR agent's calibration workflow.

Modules:
- metrics: Calibration quality metrics (RMSE, R2, RPD, bias, slope, MAE) and
  ``evaluate_quality`` for grade-based quality assessment.
- validation: Outlier detection, train/test leakage checks, retry decision,
  and next-pipeline selection.
- drift: Mahalanobis-based drift detection and a composite drift index.
- registry: JSON-backed ``ModelRegistry`` for versioned model artifacts.
"""
