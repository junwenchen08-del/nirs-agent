# nir_core

Deterministic near-infrared (NIR) spectroscopy algorithms for the NIR Agent.

## Design Principles

- **Zero framework dependency**: only numpy, scipy, scikit-learn, matplotlib, pydantic.
- **Pure-function first**: numpy in, numpy out, no side effects.
- **Pydantic data contracts**: all I/O uses strongly-typed models.
- **Fully type-annotated** and independently testable without DeerFlow.
- **Installable**: `pip install -e .` into a sandbox venv.

## Installation

```bash
# Development install
pip install -e .

# With optional extras
pip install -e ".[mat73,dev]"
```

## Package Layout

```
nir_core/
├── __init__.py          # Re-exports core models and config
├── models.py            # SpectralData, PreprocessingStep, ModelResult, ...
├── config.py            # NirConfig, QualityThresholds (package-managed config)
├── io/                  # Data I/O: loaders, sniffers, validators, writers
├── preprocess/          # SNV, MSC, SG, airPLS, ASLS, scaling, pipeline
├── model/               # PLS, PCR, SVR, ensemble, evaluation, CARS/SPA
├── utils/               # metrics, validation, drift, registry
├── plotting/            # spectra, model diagnostics, gallery
└── tests/               # Unit + integration tests
```

## Quick Start

```python
from nir_core import SpectralData, get_nir_config
from nir_core.tests.generators import generate_synthetic_spectra

data = generate_synthetic_spectra()
print(data.summary())

cfg = get_nir_config()
print(cfg.quality.get_thresholds("soil", n_samples=50))
```

## Configuration

NIR config is managed inside the package (`nir_core/config.py`), NOT in
DeerFlow's `config.yaml` (whose `AppConfig` Pydantic model rejects undefined
fields). Override defaults by loading a JSON file:

```python
from nir_core.config import NirConfig, set_nir_config
set_nir_config(NirConfig.load("nir_config.json"))
```

Quality thresholds are tiered by application domain (food_protein, food_moisture,
soil, feed, pharma, default) and auto-relaxed for small samples (<100).
