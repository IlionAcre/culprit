# CLAUDE.md: culprit

This file is the source of truth for any agent working on this project. It
records decisions already made and *why*, so they aren't re-litigated in a
future session, plus explicit guardrails on what not to change without asking.
Treat this as a living decision log: whenever a new architectural decision is
made, add it here with its rationale before moving on.

## Rule: read `AI_docs/` before doing anything else

Before taking any action in this project, read every file in the `AI_docs/`
folder. That folder holds agent-facing reference material, the detailed
execution roadmap (`AI_docs/PHASES.md`) and any other reference docs added
there over time. This file (`CLAUDE.md`) only holds high-level decisions and
guardrails; the operational detail (current phase, next concrete action, exit
criteria) lives in `AI_docs/`, not here. `AI_docs/` is gitignored: it is local
agent scaffolding, not published content.

## What this is

culprit ingests a failed LLM agent trace and identifies the step where the run
left the space of trajectories that would have succeeded, plus the reason it
did. Existing observability tools (Langfuse, LangSmith, Braintrust, Helicone)
show the trace; none of them identify the cause. That gap is the product.

Architecture is a layered cascade: deterministic detectors and contrastive
sequence alignment against successful runs narrow thousands of steps down to
five candidates, then targeted LLM adjudication judges only those five with
bounded context. Built as a portfolio artifact demonstrating AI engineering
depth, after a market-verification round established the commercially obvious
ideas in this space are already served.

Full original specification: `docs/superpowers/plans/2026-08-12-culprit-backend.md`.

## Brand

Name: **culprit**. Tagline: **"Not where it failed. Where it broke."** Don't
rename the project or change the tagline without the user explicitly asking.

Propagates to: `src/culprit/`, distribution name `culprit`, CLI command
`culprit`, config file `culprit.toml`, `LOGGER_NAME = "culprit"`, `CULPRIT_*`
env vars.

## Architecture decisions (with rationale)

### Naming and scope

- **Name `culprit`.** Names the deliverable: the step that broke the run.
  *Rejected: `postmortem`* (implies a written report after the fact, not a
  live diagnostic tool), *`divergence`* (names the mechanism, not the
  product), *`tracer`* (collides with the observability tools this sits on
  top of, and implies it only shows the trace rather than the cause).

- **Backend only. No frontend.** Keeps the artifact focused on the part that
  is technically distinctive (the analysis pipeline), not UI polish.

- **v1 ships L0-L3 plus L5. L4 and L6 are deferred but designed for.** L0
  normalize/linearize, L1 deterministic detectors, L2 contrastive trajectory
  diff, L3 targeted adjudication, L5 clustering. Deferred: L4 counterfactual
  replay, L6 regression case generation. Neither deferred layer requires
  touching a v1 contract: L4 consumes a `Candidate` and merges evidence back
  through the same path L1 uses (`Candidate.source` is a plain `str`, not an
  enum, specifically so `"l4"` can be added later without a schema change);
  L6 consumes a persisted `Diagnosis` and emits a Litmus-format test case
  (`Diagnosis.layer_versions` is a `dict[str, str]` rather than fixed columns
  for the same reason).

- **The benchmark harness is in scope for v1.** This is what turns the
  project from "a tool I built" into "a tool that scores X on TRAIL",
  matching the evidence-backed posture of the Litmus write-up.

### Why a layered cascade, not one long-context call

Published benchmarks in agent failure attribution (TRAIL, Who&When, MAST,
AgenTracer) show that even strong long-context models perform poorly when
handed a whole trace and asked what went wrong. The naive design ("put the
trace in the context window and ask") does not work, so culprit narrows
candidates deterministically first and spends model calls on a handful of
pre-narrowed steps with bounded context around each. This is also the cost
story: the expensive layer runs on roughly five steps per trace instead of
thousands (see `AI_docs/PHASES.md` for the per-diagnosis cost table).

This is the canonical citation for deterministic-first narrowing: **"Rules of
Machine Learning"** (Google, 2018), Rule 1, do not use ML where a heuristic
suffices.

### Storage and infrastructure

- **Postgres with JSONB and pgvector as the single datastore.** Traces are
  large and nested, analysis needs cross-run analytical queries, and
  clustering needs vector similarity. One datastore holds all three.
  - *Rejected: JSON files + DuckDB (the Litmus pattern)* - fine for Litmus's
    small run files, but trace volume, concurrent worker writes, and vector
    similarity all push past it.
  - **Deliberate departure.** Litmus's `CLAUDE.md` lists "don't introduce a
    database server" as a guardrail. Confirmed with the user before adopting
    Postgres here; the two projects are allowed to make different storage
    calls on their own merits.

- **psycopg 3 with `ConnectionPool`, no SQLAlchemy ORM.** The schema is
  mostly JSONB columns, bulk inserts, and vector operators, none of which an
  ORM improves. Alembic is used purely as a migration runner with
  hand-written DDL, never `--autogenerate`.
  - *Rejected: SQLAlchemy ORM* - would mean maintaining model definitions
    that duplicate the Pydantic schemas for no benefit.

- **Redis + RQ for the queue, separate worker entrypoint sharing the
  package.** Analysis takes minutes per trace and cannot run inside a
  request.
  - *Rejected: arq* - async-native, fights the sync-everywhere convention
    and breaks the established `monkeypatch.setattr` mocking pattern.
  - *Rejected: Celery* - heavier configuration surface than this needs.

- **Sync throughout, including FastAPI routes.** Matches Litmus exactly.
  Fan-out uses `ThreadPoolExecutor(...).map()`, never `asyncio`.

- **`fastembed` + `BAAI/bge-small-en-v1.5` for embeddings, no torch.** ONNX,
  ~130 MB, 384-dim, CPU-only, zero API cost. One model instance in
  `embed.py` serves both `traces.task_embedding` and
  `diagnoses.card_embedding`.
  - *Rejected: `sentence-transformers`* - pulls torch, ~2.5 GB, contradicts
    this codebase's precedent of avoiding heavy ML deps for one feature.
  - *Rejected: `litellm.embedding`* - per-call cost over every diagnosis plus
    a network dependency that breaks the offline-tests guarantee.
  - *Rejected: bge-base (768-dim)* - 3x index size for marginal gain at this
    corpus size.

- **`litellm` pinned `<1.90` on Windows only, unbounded elsewhere.** Copied
  deliberately from Litmus: same machine, same Rust-wheel/linker problem. See
  the Known Gotcha below; do not "simplify" the platform-conditional
  dependency entry back to a single unpinned line.

### Ingestion (L0)

- **Ingest OTLP spans; normalize both attribute vocabularies.** Accept OTLP
  over HTTP (protobuf and JSON) plus direct file upload. Both OpenTelemetry
  GenAI conventions and OpenInference ride on ordinary OTel spans, so one
  transport covers both. OTel GenAI *agent* spans were still Development
  status as of mid-2026 while client spans stabilized; OpenInference is
  richer today and already dual-emits. Neither is safe to bet on
  exclusively.
  - Practical consequence: anyone already emitting OTel points a collector
    at the endpoint rather than instrumenting anything new.
  - **Raw attributes are never discarded**, so a later vocabulary module can
    backfill spans that failed to normalize. This is what makes
    normalization a safe bet rather than an irreversible one. An
    unrecognized span sets `normalize_error` rather than raising.

- **`payload` on `Span` is a plain union, not a discriminated union.** The
  discriminant lives on the sibling `kind` field, and Pydantic needs the tag
  inside the member itself for `Field(discriminator=...)`. Resolution
  happens by `kind` in `normalize.py`.

- **`task_embedding` deliberately does not live on the Pydantic `Trace`.**
  It is a 384-float storage concern that would bloat every API response and
  log line. It exists only as a Postgres column.

- **Linearization rule 1: DFS pre-order, siblings sorted by `(start_ns,
  span_id)`.** The `span_id` tie-break is not cosmetic. OTel exporters
  routinely emit siblings with identical millisecond-truncated start times,
  and without a deterministic secondary key the same trace linearizes
  differently on two runs, silently poisoning every L2 alignment.

- **Linearization rule 2: only semantic spans become steps.** LLM, TOOL,
  RETRIEVER, AGENT produce steps; CHAIN, EMBEDDING, GUARDRAIL, UNKNOWN do
  not, and their ids go into the enclosing step's `collapsed_span_ids`.
  LangChain and LlamaIndex emit 3 to 10 framework spans per real action;
  without collapsing, step indices are dominated by framework noise and
  "step 47" means nothing to a human.
  - *Rejected: keep every span and down-weight in the signature* - preserves
    the noise in the one number the product reports.

- **Linearization rule 3: orphan spans re-parent to root** and are flagged
  in `metadata["orphan_span_ids"]`, never dropped.

- **`synth.py` lives in `src/`, not `tests/`.** It is a product module: it
  powers a `culprit demo` command and supplies the benchmark harness's
  negative controls, not just test fixtures. Test files still write their
  own private `_helper()` wrappers around it, so the "no centralized
  fixtures" convention holds.

### The Signal / Diagnosis contract

- **`Signal.category` is an explicit hint, not a verdict.** L1 knows a tool
  returned empty; it does not know whether that caused the failure.
  Conflating the two is how deterministic detectors become false-positive
  machines. L3's `Adjudication.failure_class` is the verdict.

- **`Diagnosis` carries both `confidence` and `calibrated_confidence`, not
  just the calibrated value.** Reporting only the calibrated number would
  hide how much calibration moved it, which is exactly the number needed to
  tell whether calibration is helping.

- **`degraded_layers` on `Diagnosis` is the isolation rule at trace level.**
  If L2 blows up, the diagnosis still returns with L1+L3 results, and the
  response says so rather than quietly presenting a weaker answer as a full
  one. Same conviction as Litmus's `has_mismatched_cases`.

- **Per-item error isolation everywhere**, at span, detector, candidate, and
  layer granularity. A detector that raises produces exactly one sentinel
  `Signal` with `error` set and is excluded from downstream ranking; the
  other 19 still run.

### L1 detectors

Google's **"Data Validation for Machine Learning"** (2019) informs this
layer directly: the schema-as-versioned-artifact pattern, where an anomaly
is a deviation from an explicit, versioned expectation rather than a
heuristic tuned once. This is why `Diagnosis` carries `layer_versions` and
why the detector catalogue is versioned and independently testable.

- **A shared `DetectorContext` (frozen dataclass) built once in
  `run_detectors.py`, not rebuilt per detector.** Building shared derived
  state (signature runs, arg/result hashes, token series) once instead of 20
  times is the difference between L1 costing 5ms and 200ms per trace.

- **Four detectors are the point of the product**: `empty_tool_result`,
  `silent_history_truncation`, `parameter_drift`, `error_swallowed`. They
  share a property: the trace looks entirely healthy at that step (status
  OK, no exception), and the damage only becomes visible many steps later.
  These get the heaviest test coverage.

- **Every detector's test suite proves both the true-positive case and 20
  clean synthetic successes producing zero false positives.** A noisy
  detector is worse than a missing one. Google's **"Machine Learning: The
  High Interest Credit Card of Technical Debt"** (2014) is the citation for
  why an untested, unversioned pile of heuristics becomes unmaintainable,
  and is the reason this per-detector test requirement is strict rather than
  aspirational.

### L2 contrastive (the differentiator)

LinkedIn's **"Using deep learning to detect abusive sequences of member
activity"** (2021) is the closest prior art in the whole design: sequential
anomaly detection over user activity sequences is structurally the same
problem as detecting an anomalous step sequence in an agent trace.

- **Reference pool resolution abstains below 3 references**
  (`abstain_reason="insufficient_references"`), propagated into
  `degraded_layers`. A contrastive method with two references produces
  confident nonsense. Uber's **"Monitoring Data Quality at Scale with
  Statistical Modeling"** (2017) informs this: comparing current
  observations against a historical distribution, including how to handle
  thin history, is exactly this abstention case.
  - *Rejected: offline clustering of tasks into types* - adds a batch job, a
    staleness window, and a cold-start problem. Query-time kNN is always
    current and costs one indexed query.

- **The coarse step signature encodes the decision an LLM step made, not the
  model that made it** (`llm:{actor}:call:{tool_name}` /
  `llm:{actor}:answer` / `llm:{actor}:plan`). Signed by model name instead,
  every LLM step would be an identical token and carry zero alignment
  information, collapsing the method.

- **Consensus profile by center-star progressive alignment**, not a profile
  HMM. *Rejected: profile HMM with Baum-Welch* - needs far more than 25
  sequences to fit without overfitting, adds a dependency, and is not
  directly interpretable. The profile's output is quoted verbatim to L3 and
  to the user ("82% called `search_orders` here"), so interpretability is a
  requirement, not a nicety.

- **Needleman-Wunsch with affine gaps (Gotoh, three DP matrices), implemented
  readably first, vectorized only if profiling demands it.** A subtle
  traceback bug is invisible in aggregate metrics and silently degrades
  everything downstream, so clarity buys more than speed here.
  - *Rejected: Smith-Waterman* - local alignment deliberately discards the
    divergent tail, which is the entire object of study.
  - *Rejected: DTW* - assumes a monotone warp of a continuous signal with no
    insertion/deletion semantics; agent traces have genuine insertions and
    deletions, which is edit distance, not warping.
  - *Rejected: Zhang-Shasha tree edit distance* - O(n^2 m^2), and the tree
    structure survives as *features* inside the signature instead.
  - *Rejected: `difflib.SequenceMatcher`* - no substitution scoring, so a
    95%-similar step counts as a total mismatch. Kept as a sanity baseline
    in tests, not as the real algorithm.

- **Residual alignability via a reverse Gotoh pass**, one extra O(n*m) pass,
  not a re-alignment per position. `A(i)` operationalizes "the space of
  trajectories that would have succeeded" and `C(i) = A(i-1) - A(i)` is the
  cliff: how much achievable future step i destroyed. This is a literal
  implementation of the product's tagline.

- **An explicit earliness prior** (`- 0.05 * i/n` in the divergence score).
  Given two equally-scoring steps, the earlier is the cause and the later is
  the echo.

- **Two post-filters, both essential.** The point-of-no-return gate (only
  steps where `A(i) < A(0) * 0.85` are eligible) excludes mismatches the run
  recovers from, removing most false-positive volume. Earliest-of-plateau
  collapse (consecutive steps scoring within 0.05 collapse to the earliest)
  is the direct algorithmic answer to "the failure surfaces downstream of
  its cause": a cause and its consequences form a scoring plateau, and the
  cause is its left edge.

### L3 adjudication

- **Candidates merge L1 and L2 by step index, with a co-location bonus**
  (`+0.15` when both sources fire, capped at 1.0). The bonus is the point of
  running two independent layers instead of one.

- **Exactly five candidates per trace, one LLM call each, fanned out with
  `ThreadPoolExecutor(max_workers=4).map()`.** Five, because the whole
  architecture exists to avoid handing a model a whole trace. Five focused
  calls at ~8k tokens each cost less and perform better than one 200k-token
  call, which is what TRAIL and Who&When report.
  - *Rejected: one call with all five candidates* - reintroduces the exact
    long-context degradation the research warns about, lets the model anchor
    on the most verbose payload, and destroys the independence that makes
    cross-candidate confidence comparison meaningful.
  - Zero candidates on a FAILURE trace emits one synthetic candidate at the
    terminal step with `source="fallback"`, so L3 always examines something
    concrete.

- **Context packet hard-capped at 12k tokens**: task goal, a one-line-per-step
  spine for global orientation, a head-and-tail-truncated zoom window on
  steps i-2 to i+2, and terminal failure evidence always included. Budget
  enforcement drops the middle of the spine before touching the zoom window,
  since the zoom window is where the actual evidence for a specific
  candidate lives.

- **21-class failure taxonomy frozen in `taxonomy.py`, `.format()`ed into the
  prompt from the same enum the code validates against.** Prompt and code
  cannot drift apart, enforced by
  `test_describe_all_emits_every_failure_class_so_prompt_and_code_cannot_drift`.

- **Three independent abstention gates**: model self-abstention
  (`is_root_cause: false` is presented as a correct, expected answer, not a
  refusal to be avoided), a confidence floor (0.55 calibrated), and an
  ambiguity margin (top two candidates within 0.08 and different failure
  classes abstains with both surfaced). A coin flip presented as a verdict
  is worse than no verdict. This whole discipline mirrors Uber's **"Project
  RADAR: Intelligent Early Fraud Detection with Humans in the Loop"** (2022):
  presenting ranked suspects to a human rather than a verdict.

- **The coefficients in `src/culprit/confidence.py` are hand-set priors that
  have never been fitted to anything, because no labeled data exists yet.**
  `bench_score.py` reports Brier, ECE, and reliability points from day one so
  they can be fit by logistic regression once labeled data exists. Do not
  describe anything in this system as calibrated; claiming calibration that
  has not been measured is exactly the unearned claim the Litmus decision
  log calls out repeatedly, and culprit does not repeat that mistake.
  - The cite-check term (fraction of `cited_step_indices` actually present
    in the context packet) is the highest-value term in the formula: a
    rationale citing step 47 when only 12-16 were shown is direct evidence
    of confabulation and should crush confidence.

### L5 clustering

- **Embed a canonical diagnosis card, not the trace and not the rationale
  alone**: failure class, step signature, expected vs observed, detector
  names, and the first 300 chars of rationale with IDs, UUIDs, numbers, and
  dates regex-masked. The masking is load-bearing: without it, cluster
  structure is dominated by order IDs and timestamps, producing 400 clusters
  of one trace each. A dedicated test asserts two diagnoses differing only
  by order ID render byte-identical cards.

- **`sklearn.cluster.HDBSCAN`** over L2-normalized embeddings
  (`metric="euclidean"`, monotonically equivalent to cosine, sidesteps
  sklearn HDBSCAN not accepting cosine directly).
  - *Rejected: KMeans* - requires k up front and forces every diagnosis into
    a cluster, with no way to say "this is genuinely novel."
  - *Rejected: agglomerative + threshold* - the threshold is as arbitrary as
    k, and still has no noise concept. HDBSCAN's `-1` noise label is a
    feature: a novel failure mode surfacing as unclustered is useful
    information, not a defect.
  - **Deliberately not UMAP-then-HDBSCAN**, the standard recipe: adds
    `umap-learn` plus `numba` for gains that only matter at a scale this
    project does not have. Documented as the first knob to turn past a few
    thousand diagnoses.

- **One LLM call per cluster, never per trace.** Input is the medoid card
  plus 4 cards sampled at increasing distance, so the label covers the
  cluster's spread rather than only its center. Re-label only when the
  medoid changed or size grew more than 50% since `labeled_at`. A test
  asserts the call count equals exactly the number of clusters; that is the
  cost-control invariant for this layer.

- **`clusters.labeled_size` stores size at labeling time, separate from the
  live `size`.** The re-label policy above compares current size against
  size *when last labeled*; without a persisted `labeled_size`, "grew more
  than 50% since it was last labeled" has nothing to diff against and the
  policy cannot be implemented. `clusters.suggested_fix` holds the
  LLM-authored remediation text alongside `label`. The prose column is named
  `description`, not `summary`, matching this project's own precedent for
  "short code plus longer explanatory text" (`taxonomy.py`'s `_DESCRIPTIONS`
  dict for `FailureClass`).

### Postgres schema

- **`spans`' primary key is composite `(trace_id, span_id)`**, because OTel
  span ids are only unique within a trace, not globally. Getting this wrong
  produces collisions that appear months in.

- **`attributes` and `payload` are JSONB; `kind`, `status`, and timestamps
  are typed columns.** Every query filters on the typed fields; the JSONB
  fields' shape is genuinely open across two vocabularies plus vendor
  extensions, and flattening would force a migration for every new attribute
  any tracing library invents.

- **HNSW indexes, not IVFFlat**, on both vector columns. IVFFlat needs a
  representative training set at build time and degrades as the corpus
  outgrows its list count; traces and diagnoses accumulate continuously, so
  there is never a stable point to train against.

- **The HNSW index on `traces.task_embedding` is partial, `WHERE outcome =
  'success'`.** References for L2 come only from successes, so indexing
  failures wastes half the index and slows every build for no query that
  ever runs.

- **HNSW indexes ship in a later migration than the table DDL.** A large
  initial backfill is not slowed by index maintenance on every insert if the
  index does not exist yet.

- **`diagnoses.trace_id` is deliberately not unique.** Re-running analysis
  after improving a detector must produce a new diagnosis alongside the old,
  and `layer_versions` is what makes the history comparable. Without this,
  "did that change help" is unanswerable, which defeats the point of having
  a benchmark harness at all. This is Apple's **"Overton: A Data System for
  Monitoring and Improving Machine-Learned Products"** (2019) applied
  directly: diagnosis versioning, not diagnosis overwriting.

### Retention policy

- **`prune_attributes(conn_fn, older_than_days)` empties `spans.attributes`
  and sets `attributes_pruned = true`; `payload`, `steps`, `signals`,
  `divergences`, and `diagnoses` are never pruned.** Default 90 days.
  `normalize.py` never re-reads pruned spans, so pruning cannot corrupt
  analysis; it only forecloses future re-normalization of old traces, an
  accepted trade.
  - *Rejected: deleting whole old traces* - destroys the L2 reference pool,
    which is the one thing that gets more valuable with age.
  - *Rejected: compressing rather than emptying* - Postgres already
    TOAST-compresses large JSONB, so the remaining win is small and the code
    to get it is not.

### Benchmark harness

- **Adapters produce canonical `Trace` objects and go into the same tables**
  (`source='benchmark:trail'` etc.) rather than a parallel code path. This
  exercises the production pipeline end to end, which is the entire value of
  having a harness and is easy to accidentally get wrong by special-casing
  benchmark data.

- **TRAIL's error-category-to-`FailureClass` mapping is an explicit dict
  constant**, not inferred at runtime, so the mapping is reviewable,
  versioned, and diffable. It encodes a genuine judgment call about how
  another team's taxonomy corresponds to ours, and that judgment will need
  revision over time.

- **Metrics report all three tolerance bands (@0, @1, @3), never only the
  best one**, plus a signed earliness error (`predicted - ground_truth`,
  positive meaning the system is blaming a symptom downstream of the real
  cause) and candidate recall@k for k in {1, 3, 5}. Candidate recall@k is the
  most important metric in the harness: it separates "L1/L2 narrowing missed
  it" from "L3 picked wrong from a correct shortlist," which have completely
  different fixes and are indistinguishable from end-to-end accuracy alone.

- **Layer ablation (L2 disabled, L1 disabled) is part of the harness**, to
  prove the contrastive layer earns its complexity rather than asserting it.
  Nubank's **"ML Model Monitoring: 9 Tips From the Trenches"** (2021) is the
  citation for reporting metrics you would rather not see, which is the
  posture the harness is built around: **write the measured numbers into
  this file, favourable or not.**

### Project structure and process

- **culprit is a new, standalone git repository**, sibling of `Litmus/`. No
  code is written inside `Litmus/`; every Litmus path referenced in planning
  is a read-only style reference to copy by hand, never a file to edit. v1
  has no runtime dependency on Litmus of any kind. The only place the two
  projects could genuinely couple is L6 regression case generation, which is
  deferred.

- **Every module stays under ~200 lines.** Flat `src/culprit/` layout, one
  module per concern. Only an open plugin set earns a subpackage:
  `vocab/`, `detectors/`, `benchmarks/`. `cli.py` at 200 lines is only
  achievable because `views.py` absorbs output shaping and `pipeline.py`
  absorbs orchestration; if it drifts anyway, that is the one acknowledged
  exception (Litmus's own `cli.py` is 383 lines).

- **Phase 0 (foundation) freezes `pyproject.toml`, `config.py`,
  `culprit.toml`, `schemas.py`, `signals.py`, and `taxonomy.py` before the
  parallel workstreams start.** These are the worst merge-conflict magnets
  and the contracts every workstream compiles against; a workstream finding
  a genuinely missing field records it as an integration item and works
  around it locally rather than editing the shared contract mid-flight.

- **Exactly one Alembic revision during the parallel phase.** New DDL
  discovered mid-phase becomes revision 0002 during Integration, never a
  second parallel revision. Parallel revisions create `down_revision`
  linear-history conflicts that are painful to untangle.

- **`pipeline.py` is Foundation-stubbed and Integration-owned; nobody else
  edits it.** Same reasoning applies to one owner per plugin registry
  (`vocab/registry.py`, `detectors/registry.py`, `benchmarks/registry.py`):
  no cross-registration between workstreams.

- **`CLAUDE.md`, `README.md`, and `AI_docs/PHASES.md` are Integration-only**
  during the parallel workstream phase; each workstream instead records its
  own decisions in its PR description in this same format, and Integration
  merges all of them in one pass. Eight agents appending directly to one
  decision log would conflict on nearly every merge.

- **Duplicate small helpers rather than sharing them across workstreams.**
  Two workstreams both needing text truncation each write their own private
  `_truncate()`. This is the existing Litmus convention taken further: here
  it also removes a whole class of cross-workstream merge contention.

## Known gotcha: litellm's Rust-accelerated wheel breaks on Windows

Inherited directly from Litmus, same machine, same failure. `litellm`
versions `>=1.90` bundle a Rust-accelerated component with no prebuilt wheel
for `win_amd64`/`cp312`, forcing a source build that fails on this machine's
Rust toolchain (Git Bash's coreutils `link` shadows MSVC's `link.exe` on
PATH). `pyproject.toml` therefore declares two PEP 508 marker-conditioned
entries: `litellm<1.90; sys_platform == 'win32'` and `litellm; sys_platform
!= 'win32'`, so only Windows pays the ceiling. Revisit only if the Windows
dev environment's Rust/linker setup changes, or a future `litellm` release
ships Windows wheels again; don't collapse this back to one unconditional
pin.

## Known gotcha: `.env` silently turns on Postgres-gated tests

`src/culprit/cli.py` calls `load_dotenv()` at import, so any variable in a `.env` file is present before pytest collects tests. If `.env` sets `CULPRIT_TEST_DSN`, a plain `uv run pytest -q` runs the 18 database-gated tests instead of skipping them. With services up this is why the suite reports 524 passes; if the Postgres container is stopped, the same command fails instead of skipping. Either keep `.env` unset when you want the offline-only run, or run `unset CULPRIT_TEST_DSN` before pytest.

## Known gotcha: OTLP JSON encodes int64 fields and timestamps as strings

`tests/fixtures/otlp/otel_genai_sample.json` and `openinference_sample.json`
follow real OTLP JSON encoding, which is a source of subtle parsing bugs if
assumed to be plain JSON. Two things to expect when writing `otlp.py`:

- `startTimeUnixNano` / `endTimeUnixNano` are JSON **strings** holding a
  nanosecond epoch integer (`"1755000000000000000"`), not JSON numbers,
  because a JS/JSON double cannot represent a 64-bit integer exactly.
- Any `AnyValue.intValue` attribute (token counts, document counts) is
  **also** a JSON string for the same reason (`{"intValue": "412"}`), while
  `doubleValue` attributes (temperature, retrieval scores) are ordinary JSON
  numbers. The two are easy to conflate if `int(...)` is applied uniformly
  without checking which value variant is present.
- `traceId` / `spanId` are lowercase hex strings (32 and 16 characters), not
  base64, even though they are `bytes` fields in the underlying protobuf and
  strict protobuf-JSON mapping would base64-encode them. This is universal
  practice across real OTLP JSON exporters and the official examples in the
  `opentelemetry-proto` repo, not a fixture-specific choice.

The two vocabulary fixtures (`otel_genai_sample.json`, `openinference_sample.json`)
deliberately encode the same logical run (an agent calling `search_orders`,
retrieving 2 policy documents, then answering) with matching span order,
actor, tool name, document count, and relative timing offsets, so that
correct normalization produces identical `Step` sequences from both. Verified
by comparing `gen_ai.operation.name` against `openinference.span.kind` across
both files: `invoke_agent`/`AGENT`, `chat`/`LLM`, `execute_tool`/`TOOL`,
`retrieve_documents`/`RETRIEVER`, `chat`/`LLM`, with identical millisecond
offsets from the trace start in both files. `raw_upload_sample.json`
represents a third, distinct ingestion path (direct JSON upload, not OTLP)
and intentionally encodes a different, smaller scenario: a refund agent that
receives an empty tool result and reports success anyway, the exact
silent-failure pattern this product exists to catch. It does not need to
match the other two fixtures' step sequence.

## Known gotcha: `culprit worker` does not run on Windows

`culprit worker` launches an RQ worker, and RQ forks the process with
`os.fork`. That attribute does not exist on Windows, so the command fails with
`AttributeError: module 'os' has no attribute 'fork'`. The worker path has not
been verified; run it on Linux or WSL. The inline execution path used for the
2026-08-16 e2e does not prove the worker, but it does prove the job functions
against the real Postgres and Redis services.

## Guardrails - don't do these without asking first

- Don't introduce `asyncio`/`async def` anywhere, including FastAPI routes.
  Fan-out is `ThreadPoolExecutor(...).map()`, full stop.
- Don't add a SQLAlchemy ORM layer over the Postgres schema. Alembic is a
  migration runner only, DDL is hand-written, `--autogenerate` is never used.
- Don't move storage back to JSON files + DuckDB. That was Litmus's choice
  for Litmus's problem; Postgres + pgvector was a deliberate, confirmed
  departure for this project's different data shape.
- Don't swap `fastembed`/`bge-small-en-v1.5` for `sentence-transformers` or a
  `litellm.embedding()` call. Both were rejected explicitly (torch weight,
  offline-tests guarantee); revisiting needs a real reason, not convenience.
- Don't change the `import litellm` / `litellm.completion(...)` dotted-call
  form to a `from litellm import completion` import. The dotted form is
  mandatory for `monkeypatch.setattr("litellm.completion", ...)` to work.
- Don't collapse the platform-conditional `litellm` version pin in
  `pyproject.toml` into a single unconditional entry.
- Don't edit `pipeline.py` outside of Integration tasks.
- Don't add a second Alembic revision during the parallel workstream phase;
  new DDL is an Integration-phase revision 0002.
- Don't rename the project or change the tagline.
- Don't skip the Phase 0 gate (`uv run pytest -v` green, `git tag
  phase-0-complete` present) before dispatching the eight parallel
  workstreams; every contract they compile against lives behind that tag.
- Don't discard raw span `attributes` during normalization, even for
  unrecognized vocabularies. Set `normalize_error` and keep the raw dict; a
  later vocabulary module depends on that data still being there.
- Never read, print, or commit `.env`. It holds the local Gemini API key and
  service DSNs, is gitignored, and is filled in by hand. Load it only with
  `set -a && . ./.env && set +a` so values enter the environment without ever
  appearing in agent context.

## Phase 1 workstream decisions (consolidated)

Decisions that were recorded only in commit messages or module docstrings are
brought into this file here so they are not re-litigated later.

### WS-A: Ingestion and linearization

- **OTLP payloads are decoded by hand, not via `google.protobuf.json_format`.** Real OTLP JSON exporters use lowercase hex strings for `traceId`/`spanId` and encode `int64` fields and nanosecond timestamps as JSON strings; strict protobuf-JSON mapping would base64-encode the IDs and leave numbers as numbers. Decoding both protobuf and JSON wire formats into the same flat span-dict shape keeps `vocab/` and `normalize.py` transport-agnostic.
- **Vocabulary failures are isolated once in `normalize.py`.** A bad JSON attribute or a missing required field in one vocabulary module is caught at dispatch, and the raw attributes are retained with `normalize_error` set so a later vocabulary module can backfill without re-ingesting.

### WS-B: Persistence

- **Read/write round trips are split from analytical queries.** `store_traces.py` handles the `Trace`/`Span`/`Step` round trip; `store_traces_query.py` holds `nearest_successful` and `prune_attributes`. The two change for different reasons (domain model shape versus index strategy and retention policy) and the combined module had grown past the ~200-line ceiling.
- **Diagnoses, cluster assignments, and cluster labels each have their own store module.** `store_diagnoses.py`, `store_clusters.py`, and `store_cluster_labels.py` are split because reading/writing a `Diagnosis`, mapping HDBSCAN integer labels to stable `cluster_id`s, and persisting label text are three different concerns with different reasons to change.

### WS-C: L1 detectors

- **`missing_verification` is gated on `trace.outcome == FAILURE`.** Verification is genuinely optional in roughly half of clean synth runs, so a pure presence check cannot distinguish a legitimate skip from an injected one. The gate is safe only because L1 never runs on successful traces.
- **`unused_retrieval` was restored to the catalogue's literal "<15% token overlap with next prompt" test after `synth.py` began folding retrieved documents into the next LLM prompt.** Before that generator fix, synth prompts were independent hand-authored strings with no retrieved content, so the literal test fired on every retrieval. The detector was temporarily reimplemented as query coverage; restoring the spec required changing the generator, not the detector.
- **`stall_timeout` uses an in-trace proxy instead of the spec's cross-trace ">5x median for that signature".** L1 has no historical reference pool, so it compares each step's duration against the median duration of the same *kind* within this trace. That still catches one step stalling far longer than everything around it.

### WS-D: L2 contrastive

- **`contrast()` takes `spans_by_id`, mirroring `run_detectors(trace, steps, spans_by_id)`.** The original frozen contract passed only `steps`, leaving `signature.sim`'s tool-argument and retrieved-document features structurally unreachable. Passing spans makes those three features real and lets the contrastive layer see payload-derived facts.
- **The point-of-no-return gate uses a length-normalized rate and compares it to the best rate observed anywhere in the trace.** Raw residual alignability `A(i)` shrinks merely because the suffix has fewer steps left, so comparing raw `A(i)` to `A(0)` would mark every tail eligible. A fault near the start also drags `rate(0)` down with itself, muting the drop the gate exists to catch.
- **`signature.sim` treats "both sides lack this feature" as agreement (1.0), not a fixed neutral 0.5.** Two steps that both have no arguments, or both have no retrieved docs, trivially agree on that fact; scoring them 0.5 would punish agreement.
  - *Rejected: keep a fixed 0.5 fallback for missing span data* - would make unrelated steps that simply share the absence of a feature look less similar than they are.

### WS-E: L3 adjudication

- **`adjudicate()` and `build_context_packet()` gained an optional `spans_by_id` keyword (default `None`).** Integration item 2: the original signature passed only `Trace` and `list[Step]`, and `Step.summary` is a short templated sentence rather than raw payload text, so the spec's zoom window could not be built as written. When `spans_by_id` is supplied, neighbours i-2..i+2 render real payload text; existing WS-E tests that omit it keep passing unchanged.

### WS-F: L5 clustering

- **Cluster identity across passes is resolved by membership overlap, not by threading a stable `run_id`.** HDBSCAN integer labels are artifacts of cluster discovery order within one `fit_predict` call, not stable names for cluster content. Pinning `run_id` and deriving `uuid5(run_id, label)` would collide unrelated clusters that happened to both be born "label 0" and would miss real matches when discovery order relabels a stable cluster. `store_clusters.resolve_cluster_identity` takes a majority vote of this pass's member diagnoses against existing `diagnoses.cluster_id`; a previously persisted `cluster_id` can be claimed by only one new group so a split cluster does not hand the same identity to both halves.
- **Only clusters that were actually re-labeled are written back.** Reusing a prior label must not advance `labeled_at`, or the cost-control invariant would lie about when an LLM summary was generated. `store_cluster_labels.write_cluster_labels` re-runs `needs_relabel` to decide which rows to update.
- **`diagnoses.card_text`/`card_embedding` remain unwritten.** Those columns live on `diagnoses`, not `clusters`, and nothing in the plan specifies a caller that renders and embeds a card purely to persist it outside the clustering pass that already consumes it in memory. Left as a known gap rather than guessed at.

### WS-G: Service surface

- **A failed RQ job's `exc_string` is never returned by `GET /jobs/{id}`.** The traceback can contain filesystem paths, a Postgres DSN, a Redis URL, or request/response fragments. `queue.fetch_job_status` logs the detail once by `job_id` and returns a generic message plus the same `job_id`.
- **Every default seam in `jobs.py` resolves to a real implementation.** `_seam()` lazily imports by name so callers compiled against stubs; Integration I2 repointed the cluster-writer seam at `store_clusters`, wired `pipeline.diagnose`, and added `ingest.ingest_payload`.

### WS-H: Benchmark harness

- **TRAIL and Who&When adapters produce canonical `Trace`/`Span`/`Step` objects and run through real production code.** This is the difference between exercising the pipeline and special-casing benchmark data.
- **Who&When's synthetic root is `SpanKind.CHAIN`, not `AGENT`.** `linearize.py` folds CHAIN spans into the next semantic step's `collapsed_span_ids` rather than giving them their own step index. If the root were AGENT it would consume step 0 and shift every subsequent message, breaking the direct `mistake_step == ground_truth_step_index` mapping the plan promises.
- **`ground_truth_failure_class` is always `FailureClass.UNKNOWN` for Who&When.** Who&When's ground truth is a free-text `mistake_reason`, not a labeled taxonomy, so there is nothing to map.

### Integration I1-I3

- **`pipeline.diagnose` threads `model` as a required keyword argument.** Foundation amendment found by WS-G: `adjudicate()` requires a model string, core modules never import `config`, so the caller must supply it.
- **Per-layer isolation is the point of `pipeline.diagnose`.** A layer that raises degrades the diagnosis and records itself in `degraded_layers`; L2 additionally degrades on a clean abstention because an abstained L2 is materially weaker than a working one.
- **L5 clustering is deliberately not called from `pipeline.diagnose`.** `jobs.recluster_job` runs it as a scheduled batch over the accumulated diagnosis corpus; a single new diagnosis is not a meaningful clustering input.
- **`ingest.py` composes the full OTLP/raw-upload to persisted-Trace path.** No frozen contract had assembled `otlp.decode` -> `normalize_span` -> `linearize` -> `store_traces.write_trace`; I2 built that composition.
- **Alembic revision 0002 adds HNSW indexes and makes `steps.actor`/`steps.summary` NOT NULL.** HNSW indexes ship after the table DDL so a large initial backfill is not slowed by index maintenance on every insert; `NOT NULL` tightens the DDL rather than loosening the `Step` model because `linearize.py` always produces concrete strings.

## Measured results so far

Recorded here so they are not lost, and because the plan requires the
unflattering numbers be written down alongside the good ones.

**L2 contrastive, against `synth` traces with known injection index.**
Top-1 40 percent (8 of 20 kinds), recall@5 65 percent (13 of 20). Across the 18
kinds where any mechanism reaches `contrast()` at all: top-1 44 percent,
recall@5 72 percent. The plan's definition of done asks for top-1 at 80 percent
of injection kinds. We are at half that.

No scoring weight, threshold, or gap penalty was tuned to produce these. That
was deliberate: an honest 40 percent with a diagnosis of what limits it is worth
more than a tuned number that means nothing.

**Ablation.** Removing the point-of-no-return gate and the plateau collapse
changes top-1 accuracy by exactly zero, because the cliff term at weight 0.40
decides the ranking on its own. The gate is still load-bearing for suppressing
false positives on identical traces. Do not remove either on the strength of the
top-1 number alone.

**RAG context injection, null result.** Folding retrieved document content into
the following LLM prompt moved L2 accuracy by nothing at all, byte-identical
before and after. Cause: `contrast.py`, `signature.py`, and `reference.py` never
read message text. `sim()`'s eight features and the coarse signature token
operate purely on structured payload fields. Prompt realism cannot move a
feature vector that never inspects prompts. The change still earns its place by
unblocking `unused_retrieval` at L1, which does read message text.

## What was verified vs what remains unverified

A reader must not be able to mistake this for a benchmarked system. The
unbenchmarked part is still true: no real TRAIL or Who&When scores exist.

**Verified 2026-08-16 against live services:**

- **Live Postgres with pgvector.** A podman container (`pgvector/pgvector:pg16`)
  runs at the WSL VM IP `172.27.120.193:55432` (localhost port forwarding is
  broken, a known podman-on-Windows quirk; the VM IP can change if the machine
  is recreated). `CULPRIT_TEST_DSN=... uv run pytest -q` passes 524 tests, 0
  failures. Offline run remains 506 passed, 18 skipped.
- **Alembic revisions 0001 and 0002 applied live.** `alembic upgrade head`
  ran against the `culprit` database, migration tests passed, and revision 0002's
  HNSW indexes, partial-index `WHERE` clause, and `steps.actor`/`summary`
  backfill + `NOT NULL` sequence are verified. Running against live Postgres
  also found and fixed three real bugs in commit `6094f4f`: pgvector 0.5.0 type
  adaptation (`%s::vector` cast needed in expression context), `Vector.to_list()`
  for round trips, and migration-test counting of HNSW index attributes as
  columns.
- **Live LLM adjudication.** One real `litellm` call against
  `gemini/gemini-2.5-flash-lite` on a synthetic trace with `empty_tool_result`
  injected at step 3 parsed cleanly, identified step 3 /
  `silent_empty_result_misread`, confidence 0.90, prompt_tokens=1594,
  completion_tokens=187, cost_usd=0.0002342.
- **CLI e2e against real services.** `alembic upgrade head`,
  `culprit ingest tests/fixtures/otlp/otel_genai_sample.json` (trace
  `a1b2c3d4e5f60718293a4b5c6d7e8f90`), diagnose, `culprit show`, and recluster
  (1 cluster, 1 assignment). The fixture is a clean run and the pipeline
  correctly abstained (step=None, abstained=True) rather than inventing a fault.
  Redis 7 runs in a podman container at `172.27.120.193:56379`. Job functions
  executed inline; the worker path was not verified.

**Still unverified:**

- **Benchmark harness has never scored a real TRAIL or Who&When dataset.** Both
  adapters were built against hand-written 3-record fixtures. `TRAIL_CATEGORY_MAP`
  keys are best-effort guesses at TRAIL's taxonomy, flagged as such in `trail.py`.
- **Confidence coefficients remain hand-set priors.** Nothing has been fitted to
  labeled data. Report Brier, ECE, and reliability points from `bench_score.py`,
  but do not describe the system as calibrated.
- **Failure path through the full CLI+DB pipeline.** A synthetic failure was
  adjudicated in-process in the live smoke test, and the clean OTLP fixture ran
  through the CLI, but a real failure trace has not yet been ingested,
  diagnosed, persisted, and read back via `culprit show`.
- **Queued-worker path.** See the known gotcha below: `culprit worker` cannot run
  on Windows.

## Status

Phase 2 closeout is in progress on branch `phase-2-closeout`. See
`AI_docs/PHASES.md` for the authoritative resume point, the Phase 1
workstream table, and the Phase 2 integration checklist.
