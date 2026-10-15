# AGENTS.md

Model details distinguish unverified dataset linkage from old model age. A
server-recovered dataset link uses the existing pinned-profile preview and full
dataset navigation; no extra manual selection is required. Raw-file linkage with
no saved parsing version clearly labels the current overview. Missing dimensions
stay absent unless supplied by verified backend metadata. The model-library
browser regression checks this missing-link wording and read-only preview behavior.

messages/nir-delivery-files.tsx adds existing ArtifactFileList cards under
server-generated NIR delivery messages. It uses owner-scoped process hooks and
requires a finalized, successfully published exact-session summary. Older message
links recover only a session reference, never prove publication. Refresh only
reads immutable publication metadata. Download URLs keep existing authentication;
ordinary/human text cannot discover files. Tests cover projection and actual cards.

The unified supplement removes autonomous-mode.tsx and the default composer's
NIR mode/budget props. modelingChatContext drops obsolete NIR control fields from
submission without mutating browser settings; model/chat preferences remain.
Default natural-language regression entry is server-owned. Historical threads
retain their actual modes. Existing process hooks/read-only Cal and Test caches,
charts, timeline and downloads are reused. autonomous-process.tsx displays bounded
automatic_delivery facts from the API, including audit-required/cancelled states;
chat prose is never completion evidence. Publication polling does not need a final
LLM message. Existing budget adapter functions remain internal compatibility
helpers, not UI controls. Tests: core/nir/unified-entry.test.ts and autonomous.test.ts.

This file provides guidance to AI coding agents (Claude Code, Codex, and others) when working with the DeerFlow frontend. It is the source of truth; the sibling `CLAUDE.md` imports it via `@AGENTS.md`.

## Project Overview

DeerFlow Frontend is a Next.js 16 web interface for an AI agent system. It communicates with a LangGraph-based backend to provide thread-based AI conversations with streaming responses, artifacts, and a skills/tools system.

**Stack**: Next.js 16, React 19, TypeScript 5.8, Tailwind CSS 4, pnpm 10.26.2. Requires Node.js 22+ and pnpm 10.26.2+.

### Core dependencies

- **LangGraph SDK** (`@langchain/langgraph-sdk` ^1.5.3) — Agent orchestration and streaming
- **LangChain Core** (`@langchain/core` ^1.1.15) — Fundamental AI building blocks
- **TanStack Query** (`@tanstack/react-query` ^5.90.17) — Server state management
- **UI**: Shadcn UI, MagicUI, React Bits, and Vercel AI SDK elements (generated from registries — see Code Style)

## Commands

| Command          | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| `pnpm dev`       | Dev server with Turbopack (http://localhost:3000) |
| `pnpm build`     | Production build                                  |
| `pnpm check`     | Lint + type check (run before committing)         |
| `pnpm lint`      | ESLint only                                       |
| `pnpm lint:fix`  | ESLint with auto-fix                              |
| `pnpm format`    | Prettier check (`pnpm format:write` to apply)     |
| `pnpm test`      | Run unit tests with Rstest                        |
| `pnpm test:e2e`  | Run E2E tests with Playwright (Chromium)          |
| `pnpm typecheck` | TypeScript type check (`tsc --noEmit`)            |
| `pnpm start`     | Start production server                           |

Unit tests live under `tests/unit/` and mirror the `src/` layout (e.g., `tests/unit/core/api/stream-mode.test.ts` tests `src/core/api/stream-mode.ts`). Powered by Rstest; import source modules via the `@/` path alias.

E2E tests live under `tests/e2e/` and use Playwright with Chromium. They mock all backend APIs via `page.route()` network interception and test real page interactions (navigation, chat input, streaming responses). Config: `playwright.config.ts`.

## Architecture

```
Frontend (Next.js) ──▶ LangGraph SDK ──▶ LangGraph Backend (lead_agent)
                                              ├── Sub-Agents
                                              └── Tools & Skills
```

The frontend is a stateful chat application. Users create **threads** (conversations), send messages, set thread-scoped `/goal` completion conditions, and receive streamed AI responses. The backend orchestrates agents that can produce **artifacts** (files/code), **todos**, and goal state updates.

### Source Layout (`src/`)

- **`app/`** — Next.js App Router. Routes include `/` (landing), `/workspace/chats/[thread_id]` (chat), `/workspace/methods` (method library), `/blog/…`, the `(auth)/{login,setup,auth/callback}` flow, `/[lang]/docs/…`, and `/api/…` route handlers (e.g. `/api/memory`).
- **`components/`** — React components:
  - `ui/` — Shadcn UI primitives (auto-generated, ESLint-ignored)
  - `ai-elements/` — Vercel AI SDK elements (auto-generated, ESLint-ignored)
  - `workspace/` — Chat page components (messages, artifacts, settings)
  - `landing/` — Landing page sections
  - `docs/` — Docs / MDX rendering components
- **`core/`** — Business logic, the heart of the app. Domains include `threads/` (creation, streaming, state), `api/` (LangGraph client singleton), `auth/` (authentication), `artifacts/`, `channels/` (IM connections), `i18n/` (en-US, zh-CN), `nir/` (bounded NIR workflow projection), `nir-library/` (owner-scoped Dataset/Model API contracts), `settings/`, `memory/`, `skills/`, `messages/`, `mcp/`, `models/`, `suggestions/`, `tasks/`, `todos/`, `tools/`, `config/`, `notification/`, `blog/`, plus rendering helpers (`rehype/`, `streamdown/`) and `utils/`.
- **`hooks/`** — Shared React hooks
- **`lib/`** — Utilities (`cn()` from clsx + tailwind-merge)
- **`content/`** — MDX content (blog posts, docs) rendered by the app
- **`styles/`** — Global CSS with Tailwind v4 `@import` syntax and CSS variables for theming
- **`typings/`** — Ambient TypeScript declarations
- Root files: `env.js` (env validation), `mdx-components.ts` (MDX component map)

### Data Flow

The desktop process explorer's preprocessing step renders bounded official
method evidence through `core/nir/method-knowledge.ts` and the matching
`components/workspace/nir/method-knowledge.tsx` card. Missing historical evidence
stays absent; guided, fallback, and unrecorded states are distinct. Only HTTPS
`chemotools.org/methods/` source links become clickable; arbitrary fields and
raw spectra are not projected. References are proposals, never adoption claims.
Tests: `tests/unit/core/nir/method-knowledge.test.ts` and the evidence-backed
`tests/e2e/nir-process-explorer.spec.ts`. No mobile-specific UI is added.

1. User input → thread hooks (`core/threads/hooks.ts`) → LangGraph SDK streaming
2. Stream events update thread state (messages, artifacts, todos, goal)
3. Stop actions call the LangGraph SDK stream stop path; `core/threads/hooks.ts` invalidates current-thread, token-usage, and sidebar/search caches immediately and schedules one follow-up refetch because SDK stop may finish via abort + fire-and-forget cancel before backend title finalization commits
4. TanStack Query manages server state; localStorage stores user settings
5. Components subscribe to thread state and render updates

`core/messages/utils.ts::getMessageGroups` is intentionally tolerant of
out-of-order LangGraph tool results. Orphan `tool` messages are attached to the
most recent visible group, and a leading orphan opens an
`assistant:processing` group so replayed or early tool events remain visible
without surfacing console errors to users.

The method library at `/workspace/methods` reads the existing
`/api/method-knowledge/cards` endpoint through `core/knowledge/method-api.ts`.
`core/knowledge/method-library.ts` groups scientific purposes separately from
execution interfaces: MCP smoothing/scatter references are preprocessing,
regression references are modeling, and transfer/axis alignment remain Other.
Native legacy IDs get a bounded category fallback. Browsing is read-only; the
settings manager keeps editing the same persistent catalog. Method API failures
are retryable, catalog reads accept cancellation, and details expose recorded
applicability, limitations, parameter definitions and sources.
Custom-agent management/gallery/chat pages and the NIR evaluation dashboard,
their exclusive components, API clients, history helpers and translations are
removed. Next.js redirects preserve old links. `pathOfThread` always returns the
standard chat route; `contextOfThread` restores agent bindings only from owned
thread metadata and strips stale browser agent preferences. Submission waits for
metadata, and metadata errors offer an explicit retry. Gateway agent execution
and scientific evaluation interfaces remain shared backend functionality.
Tests: `method-library.test.ts`, `threads/utils.test.ts`,
`method-library.spec.ts` and `sidebar.spec.ts`.

The default chat page projects live `thread.values.nir_workflow` through
`core/nir/selectors.ts` and `core/nir/process-selectors.ts` into bounded view models and conditionally renders the
`components/workspace/nir/` Sheet. When an older restored stream omits that
field, `core/nir/hooks.ts` reads the owner-checked thread state endpoint; live
state remains authoritative, and only same-project snapshots merge by revision.
The projection allowlists metrics, caps strings and list sizes, shortens the
dataset hash, and never passes artifact paths or the complete evidence object
into the component. The desktop ChemAgent explanation panel has process,
selection-evidence, and final-result tabs. Its process selector projects up to
ten historical attempts, reflection/retry records, bounded audit facts,
comparable metric trends, and candidate evidence from the checkpoint. Charts
use only SVG and Tailwind. A trend compares attempts only when metric,
protocol, and validation scope match. Selection scores come from training-only
tuning/CV, never the final holdout. Missing evidence stays explicit, and
registration is read from a recorded workflow event rather than inferred from
`completed`. Ordinary chats render no trigger.
`core/nir/presentation.ts` maps recorded method IDs and reason codes to short
localized explanations. The selection tab leads with the chosen options and
training-only comparison, while raw codes and candidate IDs stay in collapsed
technical details. Unknown reasons remain explicitly unexplained.
The durable process explorer also uses `core/nir/process-display.ts` and
`process-facts.tsx` for Chinese metric/parameter labels, bounded number display
and collapsed original records. Candidate configuration is linked only by an
exact ID from the same step's recommendation; conflicting or absent records
remain unnamed. Never infer methods from hash strings or change recorded
scores/selection. Candidate row identity includes position to distinguish
duplicate IDs; viewing and modeling adoption have separate badges. Workflow
stage/tool names are localized while original codes remain expandable.
Tests: `process-display.test.ts` and evidence-backed desktop browser QA in
`nir-process-explorer.spec.ts` (including compact desktop windows).
Historical custom-agent threads now open in the standard chat page.

The phase-5 desktop autonomous entry was opt-in; the unified supplement above
replaces that control for new default-chat regression tasks.
`components/workspace/nir/autonomous-mode.tsx` submits the structured
`context.nir_autonomous` selection through the existing thread hooks; it does
not append policy flags to the user prompt. Gateway owns the four autonomous
flags and validates the four exposed custom budget fields. The mode control
locks after workflow start. Existing workflows retain their prior
entry behavior. `core/nir/autonomous.ts` is the bounded adapter for the
owner-checked `/nir-process/autonomous` API, not a new workflow state store.
`autonomous-process.tsx` is reused by the chat Sheet and existing process route.
Decision summaries page ten records; expanded details load proposals and trial
feedback on demand. Accepted/pending, executed, failed, rejected, STOP and RETAIN
remain distinct; alternatives never become trained models. Reasons, comparisons
and retention are recorded evidence, never inferred from metric changes.
Val RMSE/R2 trends use successful same-session experiments. The best Val RMSE
marker is informational and never changes the LLM-retained model. Test metrics
and lazily fetched cached points require finalized state and final_test scope;
old missing charts are not recomputed. Phase-5 Test views do not load raw spectra.
Downloads use `urlOfArtifact` and the existing owner-checked artifact service.
Approval/registration are read from actual workflow evidence or explicitly shown
as unrecorded; completion/export is never approval. Existing process hooks keep
owner/thread/resource cache keys, ETags, aborted stale reads, hidden-tab pause
and bounded 3–30 second retry/poll intervals. Anchors:
`tests/unit/core/nir/autonomous.test.ts` and the phase-5 implementation report;
unit projection tests alone do not establish browser/provider acceptance.

ProcessWorkspace owns shared autonomous query cleanup. Unmounting its legacy
child must clear only nir-process-legacy queries; deleting the owner's shared
nir-process cache there erases the parent's session discovery during the
legacy-to-autonomous transition. This was reproduced on the real process route
and verified after browser reload with unchanged persisted session/decision,
fit counts, final cache and Lead receipts. See the phase-5 report for evidence.

Phase 6 adds cal-spectra.tsx within that autonomous chart view. It lazily reads the
owner-scoped immutable Cal cache via existing process hooks and renders two
ProcessChart SVGs with actual ordered methods/parameters, sample and channel counts,
and explicit wavelength units. The adapter bounds/validates identity, finite rows,
axis order and before/after alignment (12 rows, 512 channels maximum); old missing
evidence stays unavailable. View/refresh/zoom never trains or recomputes preprocessing.
Cal CV is labeled OOF; parameter trial RMSECV is labeled mean-fold. Custom budget
help reflects the server-bounded autonomous LLM allowance and separate terminal
reserve. Source anchors: tests/unit/core/nir/cal-spectra.test.ts and the phase-6
implementation report; real multi-experiment/active-STOP delivery remains unverified.

The optional cross-session library lives at `/workspace/nir/datasets` and
`/workspace/nir/models`. The sidebar reads `/api/features` and exposes these
routes only when `nir_library.enabled` is true. `core/nir-library/api.ts` owns
the centralized authenticated/CSRF-protected requests; page components must not
fetch or cache raw spectra, model bytes, host paths, or another user's records.
The dataset library's Upload dataset dialog chooses a local CSV/TXT/MAT file,
defaults the editable name to its filename and sends multipart FormData through
the central authenticated fetch wrapper (no manual Content-Type boundary).
It needs no source thread/path. Failure retains the selection for retry; success
refreshes all datasets so filters cannot hide the new asset. Existing card attach
and `/datasets` composer selection reuse the uploaded asset without starting a
run. Browser anchor: `dataset-library.spec.ts`; API unit test: `api.test.ts`.
The visual parsing panel (`dataset-parsing-settings.tsx`) lazily loads bounded
structure choices when opened, debounces/aborts draft preview requests, validates
dataset/source identity and permits save only for the current successful preview.
MAT group/block/target/axis and CSV/TXT header/orientation/column controls are
localized. JSON and version history remain under Advanced settings. Save failures
retain the draft; success refreshes library previews. Composer attachment pins
the latest confirmed `library_parse` or workspace profile. Anchors:
`dataset-parsing.spec.ts` and the centralized API unit tests.
The Dataset page shows count/size statistics and quantitative, classification,
and unconfirmed filters. It reads owner-scoped `/datasets/{id}/preview` aggregates
with three concurrent requests, validates dataset/hash identity and offers retry.
Its card/dialog presents sample counts, mean spectral ranges and target
distributions, never full individual spectra. Confirmed task profiles take
precedence; ambiguous layouts stay unconfirmed. Rename, explicit save, Profile
versions/confirmation and durable use history remain. The archive button is
removed; existing archived records retain controlled deletion.
Dataset attach opens `/workspace/chats/new?dataset=...`. The composer gets the
owned record, reuses thread preparation and verified attachment, validates its
upload path and displays a mounted file before manual submission. It never
starts a synthetic run. Failure offers retry, requests abort on navigation,
and a successful attachment removes its URL marker.
The Model page shows ready model cards without storage, archive or deletion UI.
Card links open `/workspace/nir/models/[model_id]/[version]`; `model-detail.tsx`
reads the owner-scoped verified report and reuses DatasetPreview with the pinned
training Profile (including superseded confirmed Profiles). Old null Profiles
are explicitly labeled as current-library previews. Dataset/hash/Profile mismatches
never render. Report and preview reads abort on navigation and retry separately.
`model-report-charts.tsx` uses bounded native SVG with Chinese axes, per-scope R²
and RMSE, red identity lines and ordered before/after training spectra. Missing
old chart evidence stays absent. Metrics never mix tuning R² with Test RMSEP.
Model attach opens `/workspace/chats/new?model=...&version=...`; the composer
reads the exact owned version, prepares its thread and reuses verified attachment
without submitting a run. Retry/abort and mounted selection follow `/models`.
Anchors: `model-report.test.ts`, `model-library.spec.ts`, `model-command.spec.ts`.
Dataset deletion dialogs require the exact server identifier and explain that
thread-attached copies and backups remain. Storage copy distinguishes persistent
Dataset/Model bytes, thread-copy bytes, the new-write admission threshold, and
free disk; never label the admission threshold as a strict total quota. Display
`validation_scope` as validation provenance only, never as production approval.
Deleted source thread IDs are plain text, not broken links.

Knowledge settings submit multi-file PDF selections as sequential one-file
requests. BGE-M3 CPU ingestion can take several minutes per document, so files
must not share one batch-level proxy timeout even though the UI presents them
as one selection. The document table exposes publication state and metadata
editing; edits update descriptive catalog/chunk fields through Gateway without
re-uploading or re-embedding the source document. The search test consumes
Gateway retrieval diagnostics and distinguishes a calibrated abstention from
an empty result, showing the rejection reason and top similarity score.

The knowledge page has separate paper and method tabs. Method management
(`method-knowledge-settings-page.tsx`, `core/knowledge/method-api.ts`) uses
`/api/method-knowledge` directly, remaining available with paper RAG offline.
The server's `can_edit` controls management buttons; writes include the visible
catalog revision and surface HTTP 409 instead of retrying stale changes.
Candidate fields use runtime schemas and omit blank values for defaults; the
backend validates cross-parameter bounds. Draft saves, publication, retirement,
confirmed default restoration, source links and audit history are desktop web
flows. New cards only bind installed methods. Browser tests exercise a real
synthetic-only API with no user catalog mutations.
The 94-card catalog covers 70 official navigation documents and 84 installed
Chemotools MCP exports, reusing existing algorithm cards. Type/category/name
filters keep utilities, constants and example datasets distinct. MCP references
show live sanitized parameter schemas (non-finite defaults as explicit text),
required inputs, operations and separate website parameters read-only; website
presence never grants execution. New MCP references do not enter automatic
preprocessing grids. Browser QA exercises real catalog API coverage and MCP
parameter rendering without changing production knowledge state.
The catalog includes preprocessing and regression-model types with a
list filter, accurate provider labels, applicability/limitations/parameter
guidance details and editors. Model parameter combinations are marked as
planning references that do not edit automatic training grids. When all
installed algorithms already have cards, adding another is disabled.
Document listings also consume `publication_readiness`; the publish action is
disabled when required citation metadata (title, authors, year, or source) is
missing, while the server remains the authoritative enforcement point.
DOI warnings do not disable publication; the publish action exposes the
server-provided readiness message so users can verify genuinely DOI-less
papers before continuing.

`/goal` is a built-in composer command, not a skill activation. `src/components/workspace/input-box.tsx` intercepts `/goal`, `/goal clear`, and `/goal <condition>` before normal chat submission, calling Gateway `GET/PUT/DELETE /api/threads/{thread_id}/goal`. Setting `/goal <condition>` also submits the condition text as the next user task so the agent starts running immediately; status and clear do not start a run. Goal requests are tied to the current `threadId` with an `AbortController`, so switching threads or unmounting the composer aborts in-flight goal requests and stale responses cannot update the new thread's goal state. The chat pages render `GoalStatus` above the composer from `AgentThreadState.goal`, with local optimistic state until the next stream `values` update arrives. `/datasets` is another built-in composer command, enabled only in the default chat when the library feature flag is on: with no attachments pending, it lists owner-scoped saved datasets through `core/nir-library/api.ts`. Choosing a ready dataset ensures an owned thread exists, then calls the Dataset attach API without starting a run or sending a synthetic message. The composer displays a thread-scoped mounted selection; only the next manual submission includes its validated upload path in additional_kwargs.files, without re-uploading it. Mount requests abort on thread switch, unmount, or picker close. Clearing the selection does not delete the server-side thread copy. Mounted submissions return the send promise so a failure retains the selection. The picker never reads raw spectra into browser state.

### Key Patterns

- **Saved-model command (desktop web)**: the default chat enables `/models`
  separately from `/datasets`. `nir/model-command-dialog.tsx` reads owner-scoped
  model versions; archived rows cannot be attached. `attachModelForComposer`
  prepares an owned thread and invokes the existing verified model attachment
  API without starting a run. `core/nir-library/mounted-model.ts` validates the
  exact version/path before appending localized selection context to the next
  manually sent human message. This is user selection, not execution evidence.
  The composer retains new data files when the command opens, supports clearing
  and replacement, aborts pending mounts on close/thread change, and retains
  selections on submission failure. Model artifacts are never uploaded as input
  datasets. Browser anchor: `model-command.spec.ts`.

- **NIR deliveries (desktop web)**: `core/artifacts/delivery.ts` recognizes NIR
  report/model/metrics sets and folds supporting cards without deleting files.
  Ordinary artifact lists remain unchanged. New tools primarily present HTML and
  ZIP. Relative Markdown figures resolve within the current artifact directory;
  reject parent traversal. Actual offline HTML browser QA uses
  `NIR_DELIVERY_QA_HTML` with `tests/e2e/nir-delivery.spec.ts`.

- **NIR process explorer**: `/workspace/nir/process/[thread_id]` renders the
  explorer; the chat Sheet links to it with `threadId`. APIs live in
  `core/nir/process-{api,hooks,contract}.ts`, rendering in
  `components/workspace/nir/process-{workspace,chart}.tsx`. Cache keys contain
  owner, thread and full resource/version; abort obsolete requests and remove
  owner cache on exit. Summary polling pauses in hidden tabs and uses ETags and
  revision checks. Plot data is immutable and lazily fetched; zoom never changes
  model evidence. Shared sample IDs link prediction/residual selections. Only
  recorded ordered methods label candidates; hashes remain in explicit details.
  matching server comparison groups form tuning trends; minima are limited to
  the current page. Legacy summaries cannot rank attempts.
  `NEXT_PUBLIC_NIR_PROCESS_ENABLED=false` hides the entry at build time. Browser
  tests consume actual backend training exports; optional scale QA measures
  browser rendering separately from backend latency. See the implementation
  document and `tests/e2e/nir-process-explorer.spec.ts`.

- **Server Components by default**, `"use client"` only for interactive components
- **Thread hooks** (`useThreadStream`, `useSubmitThread`, `useThreads`) are the primary API interface
- **LangGraph client** is a singleton obtained via `getAPIClient()` in `core/api/`
- **Environment validation** uses `@t3-oss/env-nextjs` with Zod schemas (`src/env.js`). Skip with `SKIP_ENV_VALIDATION=1`
- **Subtask step history** (`core/tasks/`) — the subtask card shows a subagent's full step timeline (#3779): its assistant reasoning turns interleaved with the tools it ran. `Subtask.steps[]` is accumulated live from `task_running` events (appended via `mergeSteps`, not overwritten) and backfilled on expand for historical runs by `fetchSubtaskSteps`, which pages the events endpoint scoped to one task (GET `/runs/{runId}/events?event_types=subagent.step&task_id=…&after_seq=…`) until a short page, so the run-wide limit can't truncate the timeline. `core/tasks/steps.ts` is the pure model: `messageToStep` (live), `eventsToSteps` (reload), `mergeSteps` (dedup by `message_index`), and `stepsForDisplay` (what the card renders — keeps tool steps + AI steps with text, drops the trailing final-answer AI step when completed since it's shown as `result`). `core/tasks/subtask-update.ts::computeNextSubtask` is the pure per-subtask state transition (merge step deltas, keep terminal status stable); `core/tasks/context.tsx`'s `useUpdateSubtask` applies it against a `tasksRef` mirroring the latest state (not a closure snapshot), so a late-resolving `fetchSubtaskSteps` backfill merges into current state instead of clobbering SSE steps or sibling subtasks that arrived meanwhile. The owning `run_id` is carried onto history content messages in `buildVisibleHistoryMessages` so the card can resolve the events endpoint.

### Interaction Ownership

- `src/app/workspace/chats/[thread_id]/page.tsx` owns composer busy-state wiring.
- `src/app/workspace/chats/[thread_id]/page.tsx` owns active-goal display and metadata restoration for the unified chat route.
- `src/core/threads/hooks.ts` owns pre-submit upload state and thread submission.
  Upload responses carry optional `dataset_library` saved/reused/failed outcomes.
  `core/uploads/dataset-library.ts` counts distinct saved assets; the composer
  displays localized success or failure feedback before starting the run. An
  archive failure never claims restoration and failed library writes leave the
  current upload usable. `/datasets` continues to fetch actual owner-scoped
  library rows when opened. Browser anchor: `dataset-command.spec.ts`.

## Code Style

- **Imports**: Enforced ordering (builtin → external → internal → parent → sibling), alphabetized, newlines between groups. Use inline type imports: `import { type Foo }`.
- **Unused variables**: Prefix with `_`.
- **Class names**: Use `cn()` from `@/lib/utils` for conditional Tailwind classes.
- **Path alias**: `@/*` maps to `src/*`.
- **Components**: `ui/` and `ai-elements/` are generated from registries (Shadcn, MagicUI, React Bits, Vercel AI SDK) — don't manually edit these.

## Environment

Backend API URLs are optional; an nginx proxy is used by default:

```
NEXT_PUBLIC_BACKEND_BASE_URL=http://localhost:8001
NEXT_PUBLIC_LANGGRAPH_BASE_URL=http://localhost:8001/api
```

Leave these unset for the standard `make dev` / Docker flow, where nginx serves the public `/api/langgraph/*` prefix and rewrites it to Gateway's native `/api/*` routes.

## Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Core Concepts](https://js.langchain.com/docs/concepts)
- [TanStack Query Documentation](https://tanstack.com/query/latest)
- [Next.js App Router](https://nextjs.org/docs/app)

## Contributing

When adding features:

1. Follow the established `src/` structure
2. Add TypeScript types and proper error handling
3. Write unit tests under `tests/unit/` (`pnpm test`) and E2E tests under `tests/e2e/` (`pnpm test:e2e`)
4. Run `pnpm check` before committing
5. Update this `AGENTS.md` when architecture, commands, or conventions change


Dataset cards and View details open `/workspace/nir/datasets/[dataset_id]`.
`nir-library/dataset-workspace.tsx` owns the Chinese/English read-only sample
values table, single/bulk role edits, cross-page selection, name search, native
SVG spectra and input/drag interval selection. Source labels are separate from
training/validation/test/excluded roles. Missing roles remain unassigned.
Summary, row pages and curve responses must match Dataset/source SHA/Profile;
requests abort on obsolete page/selection/reload. Save pins the source SHA and
confirmed Profile ID; failures retain the draft. Dirty navigation prompts before
discard, browser unload warns, and attachment requires a saved clean revision.
Chart rendering is bounded (24 curves, 512 points); export preserves every
original value in the chosen interval. No original target/spectral values can
be edited. Data overview retains legacy parsing and usage history. Composer
attachment pins the latest saved Profile in its idempotency key and uses actual
attachment size, while run creation still requires manual submission. Existing
Dataset/Model commands stay available. Browser anchor: dataset-workspace.spec.ts.
