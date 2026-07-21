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

For a fitted PLS model, export a prediction equation in the original,
uncentered spectral space with both coefficients and intercept:

```python
from nir_core.model.pls import get_regression_coefficients, get_regression_intercept

coefficients = get_regression_coefficients(model)
intercept = get_regression_intercept(model)
y_pred = X @ coefficients + intercept
```

CSV knowledge documents are converted to bounded Markdown tables using the
Python standard library; pandas and tabulate are not required for CSV import.

The default tree-model searches are bounded for interactive agent runs and
small NIR calibration sets: eight GBM candidates and four Random Forest /
Extra Trees candidates. They avoid very deep or oversized forests that are
both slow and prone to overfitting.

## Testing

```bash
# Fast development gate: skips compute-intensive training and feature selection.
pytest -m "not slow"

# Slow training, feature-selection, and persistence regressions only.
pytest -m slow

# Complete suite (fast + slow).
pytest
```

Expensive model modules share one deterministic fitted result across assertions.
Reproducibility tests still perform an independent second fit, so fixture reuse
does not hide random-seed regressions.

## Configuration

NIR config is managed inside the package (`nir_core/config.py`), NOT in
DeerFlow's `config.yaml` (whose `AppConfig` Pydantic model rejects undefined
fields). Override defaults by loading a JSON file:

```python
from nir_core.config import NirConfig, set_nir_config
set_nir_config(NirConfig.load("nir_config.json"))
```

Quality thresholds are tiered by application domain (food_protein, food_moisture,
soil, feed, pharma, default). Production evaluation never relaxes thresholds
merely because the sample count is small. Exploratory callers must explicitly
set `allow_small_sample_relaxation=True`; significant bias blocks passage even
when R² and RPD meet their nominal thresholds.

Model-space monitoring references use compact PCA Hotelling T² and Q-residual
limits fitted on training spectra only. User-facing NPZ readers keep pickle
disabled, and deployable model artifacts are loaded only after output-path and
SHA-256 manifest verification (with optional required HMAC signing).
