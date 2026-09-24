"""nir_core: Deterministic NIR spectroscopy algorithms.

Design principles:
- Zero framework dependency (only numpy/scipy/sklearn/matplotlib/pydantic).
- Pure-function first: numpy in -> numpy out, no side effects.
- Pydantic data contracts for all I/O.
- Fully type-annotated.
- Independently testable without DeerFlow.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-export core data models for convenience.
from nir_core.config import (
    NirConfig,
    QualityThresholds,
    get_nir_config,
    set_nir_config,
)
from nir_core.models import (
    ModelResult,
    PreprocessingResult,
    PreprocessingStep,
    ReflectionRecord,
    SpectralData,
)

__all__ = [
    "ModelResult",
    "NirConfig",
    "PreprocessingResult",
    "PreprocessingStep",
    "QualityThresholds",
    "ReflectionRecord",
    "SpectralData",
    "__version__",
    "get_nir_config",
    "set_nir_config",
]
