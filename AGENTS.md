# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, Codex, and others) when working with code in this repository. It is the source of truth; the sibling `CLAUDE.md` imports it via `@AGENTS.md`.

It is the **monorepo orientation layer**: it maps the whole repo and points to the
module guides that own the depth. For anything inside a module, read that module's
guide rather than expecting full detail here:

- **[backend/AGENTS.md](backend/AGENTS.md)** — backend depth: harness/app split, agent &
  middleware chain, sandbox, MCP, skills, memory, IM channels, persistence/migrations,
  config system, test layout.
- **[frontend/AGENTS.md](frontend/AGENTS.md)** — frontend depth: Next.js App Router layout,
  thread/streaming data flow, code style, commands.

## What is DeerFlow

DeerFlow is a LangGraph-based AI super-agent system with a full-stack architecture. The
backend runs a "super agent" with sandboxed execution, persistent memory, subagent
delegation, and extensible tools (built-in, MCP, community), all per-thread isolated. The
frontend is a Next.js chat UI. External IM platforms (Feishu, Slack, Telegram, Discord,
DingTalk) bridge into the same agent through the Gateway.

## Service Topology

A single `make dev` / Docker stack runs four cooperating services:

| Service         | Port   | Role                                                                 |
| --------------- | ------ | ------------------------------------------------------------------- |
| **Nginx**       | `2026` | Unified reverse-proxy entry point — open this in the browser        |
| **Gateway API** | `8001` | FastAPI REST API + embedded LangGraph-compatible agent runtime      |
| **Frontend**    | `3000` | Next.js web interface                                               |
| **Provisioner** | `8002` | Optional — only when sandbox is configured for provisioner/K8s mode |

Nginx is the single public entry: it serves the frontend and proxies `/api/langgraph/*`
to the Gateway's LangGraph runtime, rewriting it to Gateway's native `/api/*` routes; all
other `/api/*` go straight to the Gateway REST routers. See
[backend/AGENTS.md](backend/AGENTS.md) for the runtime and router detail.

## Repository Map

```
deer-flow/
├── Makefile                        # Root orchestration: drives the full stack (dev/start/stop, docker, setup)
├── config.example.yaml             # Template → copy to config.yaml (gitignored) at repo root
├── extensions_config.example.json  # Template → copy to extensions_config.json (gitignored): MCP servers + skills
├── backend/                        # Python backend — see backend/AGENTS.md
│   ├── Makefile                    # Per-module backend commands (dev, gateway, test, lint, migrate-rev)
│   ├── packages/harness/           # deerflow-harness package (import: deerflow.*) — agent framework
│   └── app/                        # FastAPI Gateway + IM channels (import: app.*)
├── frontend/                       # Next.js frontend (pnpm) — see frontend/AGENTS.md
├── docker/                         # docker-compose files, nginx config, provisioner
├── skills/                         # Agent skills: public/ (committed), custom/ (gitignored)
├── contracts/                      # Cross-component JSON contracts (e.g. subagent status)
├── scripts/                        # Root orchestration scripts invoked by the Makefile (check, configure, doctor, support_bundle, serve, docker, deploy, setup_wizard)
├── tests/                          # Root-level tests (currently tests/skills/ — public skill tests)
└── docs/                           # Cross-cutting docs, plans, and design notes
```

Runtime config lives at the **repo root**: copy `config.example.yaml` → `config.yaml`
(main app config) and `extensions_config.example.json` → `extensions_config.json` (MCP
servers + skills). Both real files are gitignored and may be edited at runtime via the
Gateway API. Config schema and resolution order are documented in
[backend/AGENTS.md](backend/AGENTS.md).

## Commands: Root vs. Module

**Root `make` targets drive the whole stack** (run from the repo root):

```bash
make setup       # Interactive setup wizard (recommended for new users)
make doctor      # Check configuration and system requirements
make support-bundle  # Generate redacted troubleshooting summary, AI issue draft, and optional zip
make config      # Generate local config files from the examples
make check       # Check that required tools are installed
make install     # Install all dependencies (frontend + backend + pre-commit hooks)
make dev         # Start all services with hot-reload (Gateway + Frontend + Nginx)
make start       # Start all services in production mode (local, optimized)
make stop        # Stop all running services
make up / down   # Build/stop the production Docker stack (browser at localhost:2026)
make docker-start / docker-stop / docker-logs   # Docker development environment
```

Run `make help` for the full list.

**Per-module commands drive a single module** (run inside that module):

```bash
# Backend (see backend/AGENTS.md for the full set)
cd backend && make dev        # Gateway API with reload (port 8001)
cd backend && make test       # Backend test suite
cd backend && make test-nir   # All backend NIR regressions
cd backend && make test-nir-e2e  # Focused deployable NIR lifecycle
cd backend && make lint       # ruff check
cd backend && make format     # ruff format

# Frontend (see frontend/AGENTS.md for the full set)
cd frontend && pnpm dev       # Dev server with Turbopack (port 3000)
cd frontend && pnpm check     # Lint + type check (run before committing)
cd frontend && pnpm test      # Unit tests

# Deterministic NIR algorithm package
cd nir_core && pytest -m "not slow"  # Fast development gate
cd nir_core && pytest -m slow        # Compute-intensive model/selection regressions
cd nir_core && pytest                # Complete fast + slow suite
```

CI mirrors these gates in `.github/workflows/nir-core-tests.yml`: production
Ruff checks plus separate fast and slow jobs. NIR NPZ boundaries never enable
pickle. Prediction accepts only NIR-generated model artifacts under
`/mnt/user-data/outputs` after SHA-256 manifest verification; deployments may
require HMAC signing with `NIR_ARTIFACT_SIGNING_KEY` and
`NIR_REQUIRE_SIGNED_ARTIFACTS=1`. Version-3 single-target and multi-target
artifacts persist a training-only PCA T²/Q monitoring reference so prediction
reports actual applicability-domain drift rather than self-reference distance.
Every `nir_predict` attempt appends a privacy-minimized event to the locked,
fsynced SHA-256 chain at
`/mnt/user-data/outputs/prediction-audit.jsonl`; raw spectra and row-level
predictions are excluded, and audit persistence failures fail closed.
Available training-reference drift results also update an atomic model-hash
scoped state file. The default hysteresis policy starts an alert after three
consecutive batches at or above 0.50 drift and records recovery after two
consecutive batches at or below 0.10. Alert/recovery transitions are appended
once to a separate tamper-evident chain and surfaced in `nir_predict` results;
monitor failures remain explicit without invalidating completed inference.
Model registration uses locked atomic JSON replacement and records separate
training-data and artifact SHA-256 hashes. Production quality gates do not
relax for small samples, and significant bias prevents passage.
`backend/tests/test_nir_end_to_end_regression.py` is the deployable-lifecycle
regression anchor: real CSV normalization, leakage-safe inline preprocessing,
training, approved registration, verified reload, prediction, training-domain
drift detection, continuous-alert transition, prediction-audit chaining, and
tamper rejection run in one test with only virtual-path resolution mocked.
Pushes to `Duan` run the focused NIR release gate in
`.github/workflows/nir-release-gate.yml`; Gitee-native repositories use the
matching `.workflow/NIRPipeline.yml` after Gitee Go is enabled for the repo.

The NIR evaluation control room lives at `/workspace/evaluations`. Its Gateway
batch endpoint owner-checks explicit thread/scenario mappings and reuses the
same deterministic evaluator as `backend/make eval-nir`; browser-local history
contains only the latest 12 aggregate summaries.

NIR file ingestion is confidence-gated. `nir_core.io.schema` profiles CSV/TXT
encoding, delimiter, decimal mark, field roles, and orientation, and recursively
enumerates numeric MAT v5/v7.3 leaves under dotted paths. High-confidence
schemas load automatically; equal matrix candidates or ambiguous numeric CSV
columns are surfaced as `needs_user_mapping` rather than silently guessed.

Primary single-target NIR runtime paths autonomously decide whether wavelength selection
is worthwhile. They establish a full-spectrum tuning baseline, conditionally
compare CARS using calibration data only, cap selector fitting on large datasets,
and retain CARS only for a material tuning improvement; the persisted decision
evidence never consults the final test or external holdout.

Those primary single-target paths also default to bounded model-family
selection: PLS is the baseline, while Ridge, SVR, and Extra Trees are added only
when dimensionality, tuning quality, sample count, and runtime caps justify
them. An alternative must materially improve tuning RMSE; the final holdout is
never used to choose the model family.
Supervised qualitative NIR analysis uses `nir_train_classifier` under the
`classification` workflow task. It supports string labels, stratified or
group-isolated calibration/tuning/holdout partitions, bounded PLS-DA,
logistic, and calibrated-SVM selection, and a version-4 prediction artifact
with class scores, confidence/margin review policy, and training-domain
drift reference. The trainer accepts both ordinary samples-in-rows labelled
CSV files and common public spectroscopy samples-in-columns CSV files via
`label_row`/`sample_cols`; `nir_inspect` surfaces bounded
`classification_label_rows` hints so agents do not read raw spectral tables
into context or write conversion scripts. Classification final responses are grounded against balanced
accuracy, macro-F1, MCC, confusion-matrix, and per-class evidence; the built-in
protocol is an internal holdout and must not be presented as external validation.
Named Cal/Tuning/Test regression partitions are supported under both assurance
levels: `nir_train_partitioned_model(validation_scope="independent_external_validation")`
for provenance-backed external validation, or
`validation_scope="independent_holdout_not_external"` when the same fixed split
must be preserved without an external-validation claim.
The default backend dependency is `nir-core[deep,mat73,mcp]`, with `torch` pinned
to PyTorch's explicit CPU wheel index so a CPU Gateway does not pull CUDA
packages. Every advertised 1D-CNN path therefore has PyTorch in the Gateway
runtime. Explicit CNN requests run a dependency preflight before file IO or
preprocessing. After any explicitly selected model fails,
`NIRWorkflowMiddleware` blocks a different model family until the latest user
message explicitly approves that replacement.

NIR preprocessing exposes 21 ordinary pipeline methods. The added explicit-only
methods are robust SNV, EMSC, isolated-spike removal, and first/second
Norris-Williams derivatives; they do not enter the default candidate pipelines.
EMSC is fitted on the training boundary and its reference spectrum is persisted
with the existing preprocessing artifact for prediction replay. Wavelength-axis
resampling is deliberately separate as `nir_align_wavelengths`: it validates
strictly monotonic axes, updates `X` and `wv` together, records both axis hashes,
and rejects extrapolation by default. This is axis normalization, not calibration
transfer.
`nir_core.preprocess.registry` is the single source for public method IDs,
parameter schemas, auto-selection levels, and provider metadata. New pipelines
prefer pinned Chemotools 0.4.4 implementations for every numerically verified
overlap, including RNV, EMSC, AirPLS/ArPLS, Whittaker, median filtering,
Norris-Williams, detrending, rubber-band correction, and supported scaling.
Native robust SNV, isolated despiking, and max-norm behavior remain project
fallbacks; `NIR_PREPROCESSING_PROVIDER_POLICY=native` rolls back methods with a
native binding. Provider bindings survive serialization, while legacy artifacts
without bindings are pinned to native behavior. Bounded automatic candidates are built
from the calibration partition only, always include raw spectra, honor
explicit-only methods, and use RMSECV with a 1% simplicity rule. The catalog
list/detail and bounded-recommendation tools are read-only agent interfaces.
Every new preprocessing or modeling execution must first attempt the live
Chemotools MCP catalog through `chemotools_list_capabilities`. The workflow
middleware persists the attempt and rejects plan finalization plus
`nir_preprocess`/modeling tools when it is missing; a recorded MCP failure
permits the governed native fallback.

Chemotools 0.4.4 is also exposed as a first-party stdio MCP component at
`nir_core.chemotools_mcp.chemotools_server`; the example extensions config enables it by
default. Its allowlisted catalog is generated from the pinned package's public
module exports and covers preprocessing, adaptation, augmentation, feature
selection, regression, outliers, physics, plotting, inspectors, and datasets.
It exposes grouped catalog/validation/fit/apply/function/render tools rather
than arbitrary Python reflection. Stdio cwd scoping confines IO to the thread
workspace, NumPy loads disable pickle, and server-created joblib artifacts are
SHA-256 checked before reload. The NIR workflow allows catalog reads during
planning but restricts state-changing MCP execution to execution/evaluation;
governed `nir_train_*` paths remain authoritative for split-safe model choice,
quality gates, registration, and production prediction.

Calibration transfer is a separate capability, never an ordinary preprocessing
step. `nir_core.calibration_transfer` wraps Chemotools DS/PDS/SST and enforces
paired sample identities, target-to-source direction, instrument/axis hashes,
and no extrapolation. Gateway fit/apply/evaluate tools persist signed-or-hashed
transfer artifacts under `/mnt/user-data/outputs`. Spectral-only improvement is
`spectrally_validated`; an internal holdout plus a trusted model remains
`model_validated_internal`. Production approval additionally requires a
non-overlapping independent paired validation set, a bound trusted reference
model, and improved validation RMSEP.

The NIR retrieval knowledge base keeps vectors in ChromaDB and document
governance in a lightweight SQLite catalog. Governed ingestion derives stable
document IDs from DOI, normalized bibliography, or source SHA-256; content
hashes make imports idempotent and stable-source updates replace stale chunks
while incrementing the document version. `cjk-section-v2` applies a CJK-aware
budget and records section path, PDF pages, neighboring chunk IDs, chunker
version, and index version. Search is publication-gated (`published` only), and
evidence returned to the workflow carries stable chunk IDs plus an
`untrusted_evidence` marker. CLI and Gateway uploads default to `draft`; use the
explicit status transition before documents become agent-searchable. Local
embedding deployments are configured with `NIR_KNOWLEDGE_*` environment
variables. Model or dimension changes must use an isolated ChromaDB path and
index version; the current local candidate is BGE-M3 at 1024 dimensions.
Knowledge uploads use a 600-second timeout in both Gateway and Nginx because
CPU parsing and embedding of large PDFs can exceed the default proxy timeout.
The knowledge settings UI can publish documents and edit descriptive metadata;
metadata changes are synchronized to both the SQLite catalog and existing
ChromaDB chunks without changing document identity or content version.
The BGE-M3 retrieval benchmark cases are versioned in
`nir_core/knowledge/retrieval_eval_cases.bge-m3.v1.json`; keep negative and
cross-document cases when extending it.
Production search over-fetches candidates, caps chunks per document, and
applies the calibrated strong/weak score plus cross-document-margin policy in
`nir_core.knowledge.retrieval_policy`. The `/search` response exposes the
decision diagnostics. `NIR_KNOWLEDGE_RETRIEVAL_*` tuning does not require an
index rebuild, but every change must be rerun against the versioned benchmark.
Optional second-stage reranking lives in `nir_core.knowledge.reranker` and is
configured by `NIR_KNOWLEDGE_RERANK_*`. It may reorder the dense candidate pool
but must preserve dense cosine scores as the calibrated answerability signal,
expose both dense and reranker scores plus the active ranking strategy, and
fail open to dense ordering when model loading or inference fails. Reranker
changes do not require an index rebuild, but must be evaluated against the
same versioned cases. The current calibration uses 12 document-diversified
rerank candidates, a 512-token cross-encoder limit, and 0.63/0.60 dense
strong/weak thresholds.
The agent-side HTTP fallback uses
`NIR_KNOWLEDGE_SEARCH_TIMEOUT_SECONDS` (60 seconds by default); keep it aligned
with or above measured CPU reranker latency so a failed modeling attempt can
advance through the knowledge-retry stage.
`nir_core.utils.scientific_validation` is mandatory in every primary training
entry point: it blocks invalid wavelength axes, conflicting duplicate spectra,
and exact cross-partition leakage, then records a replay manifest. Model
manifests bind the exact metrics and training-data hashes; registration rejects
missing, failed, mismatched, or quality-failing evidence. Knowledge search also
returns an `evidence_assessment`: `answer` mode needs cited evidence, while
`decision` mode requires at least two independent quality A-C documents.
Publishing requires title, authors, year, and source metadata, and agents must
abstain when the requested evidence mode is not allowed.
DOI handling is a soft gate: normalize and validate supplied values, extract a
missing DOI from parsed text, reject malformed values, and expose
`doi_missing` plus evidence-level DOI completeness without blocking legitimate
DOI-less publications.
Legacy knowledge metadata migrations use hash-pinned JSON manifests and the
`audit-metadata` / `migrate-metadata` CLI commands. A real migration requires a
new backup directory, temporarily withdraws published chunks, updates the
catalog and Chroma metadata without re-embedding, and republishes only records
that satisfy the current publication gate.

Rule of thumb: **root `make` = the full application**; **`backend/Makefile` and `frontend/`
(`pnpm`) = per-module work.**

## Where to Go Next

- Backend work → **[backend/AGENTS.md](backend/AGENTS.md)**
- Frontend work → **[frontend/AGENTS.md](frontend/AGENTS.md)**
- Setup & install → **[README.md](README.md)**, **[CONTRIBUTING.md](CONTRIBUTING.md)**
- Project overview & usage → **[README.md](README.md)**
- Security policy → **[SECURITY.md](SECURITY.md)**
- Changes → **[CHANGELOG.md](CHANGELOG.md)**

## Cross-Cutting Conventions

These apply repo-wide; module guides own the module-specific detail.

- **Documentation update policy** — keep docs in sync with code: update `README.md` for
  user-facing changes and the relevant `AGENTS.md` for development/architecture changes in
  the same change set.
- **Test-driven development** — features and bug fixes ship with tests. Backend tests live
  in `backend/tests/` (TDD is mandatory there; see [backend/AGENTS.md](backend/AGENTS.md));
  frontend tests live in `frontend/tests/`.
- **Format before pushing** — run `make format` (backend) / `pnpm check` (frontend). Backend
  CI enforces `ruff format --check`, so formatting must be clean before a push.
