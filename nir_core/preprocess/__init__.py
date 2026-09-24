"""Preprocessing subpackage for nir_core.

Contains deterministic spectral preprocessing algorithms:

- :mod:`nir_core.preprocess.scatter`  -- SNV, robust SNV, MSC, EMSC.
- :mod:`nir_core.preprocess.despike` -- isolated detector-spike removal.
- :mod:`nir_core.preprocess.smoothing` -- SG and Norris-Williams derivatives.
- :mod:`nir_core.preprocess.alignment` -- wavelength validation/resampling.
- :mod:`nir_core.preprocess.baseline` -- airPLS, asLS, detrend baseline removal.
- :mod:`nir_core.preprocess.scaling`  -- mean centering, autoscale, normalize.
- :mod:`nir_core.preprocess.pipeline` -- pipeline orchestration & candidates.
- :mod:`nir_core.preprocess.registry` -- authoritative method/parameter catalog.
- :mod:`nir_core.preprocess.recommendation` -- bounded calibration-only candidates.

Ordinary preprocessing functions are pure and shape-preserving. Axis
resampling also leaves its input untouched, but may change the wavelength
count and must therefore be paired with the returned target wavelength axis.
"""

from __future__ import annotations
