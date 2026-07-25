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

Public MAT/CSV loaders enforce a configurable `ResourceBudget` before parsing
and after materializing the spectral matrix. Defaults are 512 MiB per file,
50 million matrix elements, 100,000 samples, 50,000 wavelengths, 256 targets,
4 GiB estimated peak memory, and 900 seconds. Override them with the matching
`NIR_MAX_*` environment variables documented in the root README.

Embedding changes are evaluation-gated. `knowledge.evaluation` compares
separately indexed retrievers using labeled Chinese/multilingual cases and
reports Recall@K, MRR, nDCG@K, no-hit accuracy, latency, and language slices.
Start from `knowledge/retrieval_eval_cases.example.json`; the current four-paper
BGE-M3 corpus has a versioned 28-case set at
`knowledge/retrieval_eval_cases.bge-m3.v1.json` and its measured baseline at
`knowledge/retrieval_eval_baseline.bge-m3.v1.md`. The post-retrieval policy
comparison is recorded at
`knowledge/retrieval_eval_baseline.bge-m3.policy-v1.md`. Do not compare two
embedding models against one shared vector index.

`knowledge.retrieval_policy` over-fetches vector candidates, caps chunks per
document, and applies calibrated abstention before returning evidence. The
default BGE-M3 policy accepts a strong score directly, accepts a mid-range
score only when it clearly leads the next document, and otherwise returns no
evidence. All policy values are configurable with
`NIR_KNOWLEDGE_RETRIEVAL_*` variables and do not require re-embedding.

`utils.scientific_validation` is the shared release contract for calibration
data. It validates matrix/target alignment, finite values, target variation,
strict wavelength ordering, conflicting exact duplicates, and exact
cross-partition leakage. It emits a deterministic data fingerprint and a
reproducibility manifest containing the seed, protocol, parameters, and core
library versions. DeerFlow training tools persist these reports and model
registration rejects artifacts without passing evidence.

`knowledge.evidence.assess_evidence` packages retrieval output for cited
answers or decisions. Answer mode permits one reliable cited document;
decision mode requires at least two independent quality A-C documents and
explicit comparison of possible conflicts. Publication is rejected until
title, authors, year, and source metadata are complete.
DOIs use a soft gate: supplied values are normalized and syntax-validated,
missing values are extracted from parsed text when possible, and publication
readiness emits `doi_missing` without blocking a genuinely DOI-less source.
Evidence assessment reports per-document DOI completeness without changing
the independent-document decision threshold.

Local embedding deployments use `NIR_KNOWLEDGE_EMBEDDING_MODEL`,
`NIR_KNOWLEDGE_EMBEDDING_DIM`, `NIR_KNOWLEDGE_CHROMA_PATH`,
`NIR_KNOWLEDGE_CATALOG_PATH`, `NIR_KNOWLEDGE_COLLECTION_NAME`, and
`NIR_KNOWLEDGE_INDEX_VERSION`. The repo-root `.env` is loaded when
`python-dotenv` is installed. Model or dimension changes require a fresh,
isolated index.

The DeerFlow integration keeps orchestration in `community.nir.modeling` while
data splitting, candidate policy, single-target fitting, multi-target helpers,
artifact persistence, and registration live in six focused modules. Existing
tool names remain stable through the modeling facade.

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
