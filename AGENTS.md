# AGENTS.md

Public product identity is ChemAgent. All visible surfaces, bilingual docs,
notifications, default agent prose and channel replies use that name. Use the
frontend branding constants for verified project links; local docs are the help
entry. Preserve legacy import, protocol, storage and environment identities for
compatibility. See docs/chemagent-branding.md.

Autonomous modeling has a shared round timeline/recorded-relation graph with
lazy detail and linked Cal/Cal CV/Val diagnostics. Display evidence is captured
from original search predictions and hash-bound to the exact execution; reads
never retrain or reopen Test. Historical gaps remain explicit. See
docs/nir-round-process.md and the backend/frontend module guides.

NIR automatic publication appends trusted model/HTML/ZIP references via the
existing artifact reducer and labels the deterministic message with its session.
Chat restores old published deliveries through the existing owner-scoped GET;
it never forges present_files tool calls, changes prior checkpoints, trains or
evaluates Test to display download cards. See the Docker download fix report.

Malformed autonomous tool JSON is never executed or silently repaired. The
existing finite continuation guard supplies parser errors, preserves provider
finish/invalid-call metadata, and durably closes an incomplete search on failure.
Do not select a model or label this closure LLM STOP/budget RETAIN. Evidence
summaries deduplicate experiment IDs and identify Cal CV/Val scopes. Tests:
backend/tests/test_nir_malformed_decision_recovery.py.

The unified NIR supplement defaults new single-target web regression execution
requests to a server-owned autonomous/sealed workflow after a model-issued Lead
start. Conceptual QA and historical modes stay intact. The composer has no NIR
mode/budget control. Unified v2 policies use explicit null cumulative token caps,
retain true usage and numeric/time/call/repair protections, and never rewrite v1
policy records. See docs/chemagent-unified-autonomous-mode-report.md.
After persisted legal STOP/RETAIN, automatic_delivery.py composes the existing
freeze/finalize/publication services without a provider call or model selection.
Unknown Test/publication leases fail closed; Gateway-confirmed orphan recovery
may resume completed freeze or report-only work. Approval remains a human gate.

This file provides guidance to AI coding agents (Claude Code, Codex, and others) when working with code in this repository. It is the source of truth; the sibling `CLAUDE.md` imports it via `@AGENTS.md`.

Unified regression accepts owned numeric MAT via the existing loader and sealed
search service. Ambiguous matrix/target mappings require explicit correction.
Only completed non-modeling inspection/knowledge workflows can hand off in the
same thread; their immutable snapshots use existing ProcessStore receipts.
Existing modeling/search bindings and budgets cannot restart. MAT inspection
must preserve the orientation already resolved from target/axis evidence.

It is the **monorepo orientation layer**: it maps the whole repo and points to the
module guides that own the depth. For anything inside a module, read that module's
guide rather than expecting full detail here:

- **[backend/AGENTS.md](backend/AGENTS.md)** — backend depth: harness/app split, agent &
  middleware chain, sandbox, MCP, skills, memory, IM channels, persistence/migrations,
  config system, test layout.
- **[frontend/AGENTS.md](frontend/AGENTS.md)** — frontend depth: Next.js App Router layout,
  thread/streaming data flow, code style, commands.

## What is DeerFlow

The opt-in phase-1 `sealed_holdout` protocol adds a safe single-target regression
search/freeze/final-Test path without replacing legacy entry points. Search models
see only Cal/Val; private session state shares the owner/thread ProcessStore v5.
PLS/Ridge/SVR/ET are adapted. Phase 2 adds opt-in `decision_strategy=llm_autonomous`
with a Cal-only DecisionContext, immutable model-issued DecisionProposal and one
initial execution. It must use sealed_holdout; direct search, legacy training,
freeze/finalize are denied in that strategy. Fixed
parameters and explicit Cartesian grids reuse safe_search, without a hidden baseline.
Lead Agent model middleware captures exact tool-call provenance before execution;
cross-task memory is suppressed. Phase 3 adds autonomous_rounds=true: each persisted
experiment feeds a fresh bounded context; the LLM compares and selects TUNE,
CHANGE_PREPROCESS, SWITCH_MODEL, EXPLORE or STOP. Execution reads only accepted
decision IDs. Fit/experiment/action budgets accumulate; hard exhaustion and LLM
STOP are distinct. STOP explicitly retains a successful experiment and waits for
final confirmation with Test sealed. Default false preserves phase-2 one execution.
See backend/AGENTS.md and docs/chemagent-phase3-implementation-report.md. Legacy
per-call holdout outputs are not a sealed search protocol.

Phase 4 is separately opt-in with autonomous_delivery=true (also requires
autonomous_rounds/llm_autonomous/sealed_holdout). decision_budget.py resolves finite
development/autonomous profiles and task_budget against NIR_AUTONOMOUS server
ceilings. Existing operation/file/memory limits stay active; single grids remain
40 candidates/200 fits. Durable call admission and actual usage share the existing
session JSON ledger.
Explicit delivery requests cannot downgrade to legacy; pre-prepare calls lease
the existing receipt store and transfer their counters. Provider input retains
the prepare evidence and current workflow state; obsolete schemas are pruned
after STOP without altering persistent decisions. Normal STOP is LLM-controlled;
budget-terminal RETAIN uses reserved resources, stop_source=budget, and no new
fits or automatic best model.
Terminal errors preserve an existing budget stop source and are recorded separately.
They also preserve completed Test/cache for publication recovery.
The original freeze/finalize tools bind the accepted retained Cal artifact and
single Test lease, then decision_delivery.py reuses trusted artifacts, reports,
ZIP delivery, prediction and registration gates. Approval remains a real user
action. Uncertain leases fail closed; caught export failures retry publication
only. Defaults preserve phases 1–3, and ProcessStore stays v5. Wall checks are
cooperative, token admission estimated, currency limits unavailable. See the
phase-4 implementation report for evidence and limits.

Phase-4 acceptance repairs use decision_schema.py for current-phase provider views
and object proposals (JSON strings stay compatible). Only process-step tools are
exposed in delivery mode; model/action/STOP choice remains with the LLM. Unchanged
contexts are cached against persisted history and runtime/knowledge versions.
Search admission transitions rebuild the final request before reserving its call;
complete converted tool schemas are included in byte-based token estimates.
Provider failure class/status is persisted separately from budget stop provenance.

Phase 5 adds the explicit desktop web entry and read-only autonomous process
projection to existing chat/process components. Gateway binds policy and budget;
ToolOutputBudgetMiddleware preserves atomic sealed protocol JSON. Charts and
downloads use persisted evidence and the original finalized Test cache. See
docs/chemagent-phase5-implementation-report.md for actual browser/provider results.

Phase 6 reuses those paths for Corn/m5 Oil functional validation. The offline
adapter in backend/scripts/prepare_nir_corn.py validates explicit source labels,
axis and duplicates and writes a fixed split/manifest without overwriting files.
Cal-only spectra are captured from the actual fitted preprocessor after execution;
owner-scoped GET reads the bounded hash-checked cache, never data arrays or a model.
decision_provider.py projects duplicate schemas, CV membership and legacy Skill
instructions out of autonomous provider messages while keeping immutable evidence,
constraints and actual model-call provenance. Custom web budgets use server-bounded
autonomous defaults, not development LLM allowance. Real-provider verification
remains per-run authorized; phase-6 real multi-experiment/active-STOP delivery is
not established. See docs/chemagent-phase6-implementation-report.md.

Governed NIR also skips the generic summary LLM for trusted web intake and active
llm_autonomous state: that before-model helper otherwise runs outside the NIR
admission ledger. Existing deterministic provider compaction and stored context
remain authoritative; ordinary/legacy summarization stays enabled. Provider cost
must distinguish Lead receipts from RunJournal totals including middleware calls.

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
The `nir_*` tools use the dedicated `nir` tool group. Host Bash gating identifies
the actual `deerflow.sandbox.tools:bash_tool` provider, so disabling host Bash
does not hide deterministic NIR tools in older configurations.

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

The workspace method library lives at `/workspace/methods`, reusing the managed
method catalog for read-only category browsing and method details. Settings
continues to edit that same catalog. Custom-agent management and the NIR
evaluation dashboard, with their dedicated frontend domains, have been removed.
Historical agent-chat URLs redirect to standard chats; owned thread metadata
preserves execution bindings. Gateway execution, scientific metrics and the
owner-checked deterministic evaluation/trace interfaces remain available.

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

The default `/workspace/chats/{thread_id}` page conditionally exposes a compact
NIR workflow Sheet. Live `thread.values.nir_workflow` is authoritative; when an
older thread's restored stream omits that field, the page uses the existing
owner-checked `/api/threads/{thread_id}/state` endpoint as a fallback. Same-project
snapshots merge by revision and a live different-project state always wins. The
frontend's `core/nir` selector is the privacy and compatibility boundary: it
accepts old optional fields, allowlists and bounds metrics/strings/pipeline
steps, shows only a shortened dataset hash, and excludes artifact paths and raw
evidence. The panel is a current-thread summary, not persistent cross-thread
history, and is intentionally absent from ordinary chats and the custom-agent
chat route.

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

The cross-session NIR asset library remains disabled by default.
`deerflow.community.nir.datasets.DatasetService` explicitly
saves owner-matched thread CSV/TXT/MAT uploads as immutable, hash-verified assets
with versioned interpretation Profiles.
HTTP uploads automatically save successful CSV/TXT/MAT when the library is
enabled and `nir_library.auto_save_uploads` (default true) is on; the upload
response distinguishes new saves, reuse, and failures. Manual Agent saves still
require explicit intent. Source files persist independently of model quality.
Batch D adds owner-checked, idempotent Dataset Attach into a target thread's uploads, `nir_dataset_*` Agent tools,
`nir_dataset_uses`, and workflow lineage binding. The Agent must run a fresh
`nir_inspect`; prior conclusions are never inherited, and concurrent lineage
conflicts block the workflow. `nir_inspect` never saves automatically. Batch F
adds `nir_model_versions`, verified approved-and-registered Model Promotion,
owner-scoped list/get/attach APIs and Agent tools, full model/metrics/training
binding verification, and prediction-compatible copies under a new thread's
`outputs/models/`. Approved agent registration automatically saves the exact
version when `nir_library.auto_save_registered_models` (default true) is enabled,
unless the user refuses storage. Its bounded outcome is persisted separately
from scientific registration. Explicit retries also accept completed approved
workflows after verifying their actual registry and current artifact evidence.
Exact `validation_scope` values
are preserved, and old direct-upload training may keep null Dataset/Profile
lineage. Batch G adds feature-gated `/workspace/nir/datasets` and
`/workspace/nir/models` pages plus controlled physical reclamation. Dataset or
model deletion requires archive first and an exact-ID confirmation; active and
archived model references block Dataset deletion, deleted rows remain as SQL
tombstones, and copies already attached to threads are deliberately retained.
The storage UI distinguishes persistent Dataset/Model bytes, thread-copy bytes,
the new-write admission threshold, and free disk; it never calls that threshold
a strict whole-system quota. See `backend/AGENTS.md` and `frontend/AGENTS.md` for
the exact service and browser boundaries.

The default desktop chat exposes `/models` alongside `/datasets` when the asset
library is enabled. The model picker mounts an exact verified version without
starting a run, retains any newly added data files, and passes the selection into
the next manually submitted message for the existing prediction workflow.
Selection is not inference evidence. See `docs/nir-model-command.md`.

Workspace exports with every sample unassigned use the existing automatic
regression/classification split, including historical exports without the
parsing-only marker. Any explicit role, including exclusion, selects fixed
membership; partial roles and missing partitions fail without random repair.
Saved bands, targets, grouping protections and source bytes remain authoritative.
Safe workspace intake failures return a Chinese reason; autonomous intake blocks
without a session and exposes status only, preventing unchanged-input retries.

NIR final responses default to plain Chinese outcome/metrics/next-step summaries.
Grounding replaces a matching search record with authoritative finalized evidence
without mutating history or duplicating experiments. Future validation advice is
distinct from a completed-validation claim. Final metrics remain scope-separated,
and result/path checks include joblib. Internal guard reasons stay in metadata.
Library saved/reused, registration-only and failed-save outcomes remain distinct.

An autonomous NIR user question is a durable pause, not a premature-final retry.
`user_wait.py` recognizes explicit choices and tracks the original user boundary;
only a later genuine HumanMessage releases it. Hidden continuation, summary and
memory messages cannot do so. Questions suppress sibling tool calls, and waiting
denies further execution/finalization/registration without selecting a model.
The existing clarification middleware ends explicit ask calls; waiting state is
checkpointed and persisted in the current unified binding without a schema change.
Answer requests ending in a colon/newline or asking for a numbered reply also
pause; the displayed question retains its choices. Normal no-argument context
reads after different persisted experiments are distinct for loop detection;
repeated reads without new experiment evidence still hit unchanged limits.
Typed runtime stops take priority over automatic continuation, close incomplete
search durably without a model choice or Test, and render a Chinese outcome.
Unfinished provider drafts remain checkpointed but hidden from chat, including
values snapshots ending in a hidden continuation message. Regression coverage:
test_nir_user_wait.py, test_nir_runtime_stop.py, test_nir_response_grounding.py.

The model library now uses ready cards, a Chinese-friendly model report page and
server-verified input linkage. New sealed ingestion records source-file SHA/size;
promotion recovers absent library IDs from the exact owner/thread source and saves
them. Historical GET recovery verifies finalized session/freeze/search binding and
bounded original-upload identity, never private search/Test caches or an estimator.
Full historical partition binding must match; filenames alone cannot prove a link.
Derived attachments retain the pinned Profile and verify its materialized bytes in
temporary storage. GET does not rewrite model packages, DB lineage or approvals.
Saved summary/curve metadata fills absent feature counts.
The report's model projection must expose the same verified Dataset/Profile IDs
as its dataset object, including read-only historical recovery. Stored lineage
and packages stay unchanged. HTTP 200 alone cannot validate client compatibility;
check the real authenticated page, dataset preview and saved charts as well.
Tests:
backend/tests/test_nir_model_dataset_link.py. See docs/nir-model-library.md.

The model library uses ready cards, a Chinese-friendly model report page and
one-click exact-version mounting into a new chat. It has no personal-storage,
archive or deletion UI. Reports reuse bounded saved results and pinned Dataset
Profiles; missing historical charts are never synthesized. ModelService verifies
the stored package for every report read. See `docs/nir-model-library.md` and the
module guides for the immutable report snapshot and browser regression anchors.

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

- **Docker updates** — production source is image-baked. Rebuild/recreate the
  affected services with the existing Compose project and overlays; preserve
  config and DEER_FLOW_HOME mounts and retain rollback image tags. `.tmp/` is
  excluded from build context. Store deployment backups outside the repository
  build context; restarting an old container does not update source.

- **Documentation update policy** — keep docs in sync with code: update `README.md` for
  user-facing changes and the relevant `AGENTS.md` for development/architecture changes in
  the same change set.
- **Test-driven development** — features and bug fixes ship with tests. Backend tests live
  in `backend/tests/` (TDD is mandatory there; see [backend/AGENTS.md](backend/AGENTS.md));
  frontend tests live in `frontend/tests/`.
- **Format before pushing** — run `make format` (backend) / `pnpm check` (frontend). Backend
  CI enforces `ruff format --check`, so formatting must be clean before a push.
