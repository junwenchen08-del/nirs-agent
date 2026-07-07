"""Preprocessing subpackage for nir_core.

Contains deterministic spectral preprocessing algorithms:

- :mod:`nir_core.preprocess.scatter`  -- SNV, MSC (scatter correction).
- :mod:`nir_core.preprocess.smoothing` -- Savitzky-Golay smoothing & derivatives.
- :mod:`nir_core.preprocess.baseline` -- airPLS, asLS, detrend baseline removal.
- :mod:`nir_core.preprocess.scaling`  -- mean centering, autoscale, normalize.
- :mod:`nir_core.preprocess.pipeline` -- pipeline orchestration & candidates.

All functions are pure: they accept a ``numpy.ndarray`` and return a new
``numpy.ndarray`` of the same shape, leaving the input untouched.
"""

from __future__ import annotations
