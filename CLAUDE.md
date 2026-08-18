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
  refusal to be avoided), a confidence floor (0.15 calibrated, see below for
  where that number comes from and why it changed from the original 0.55),
  and an ambiguity margin (top two candidates within 0.08 and different
  failure classes abstains with both surfaced). A coin flip presented as a
  verdict is worse than no verdict. This whole discipline mirrors Uber's
  **"Project RADAR: Intelligent Early Fraud Detection with Humans in the
  Loop"** (2022): presenting ranked suspects to a human rather than a
  verdict.

- **Confidence calibration, current state (third revision, 2026-08-17):
  `calibrate_confidence()` fits nine features by logistic regression, not
  four, and not hand-set priors.** The formula is `sigmoid(a0 +
  a1*logit(model_confidence) + a2*prior + a3*agreement +
  a4*evidence_density + a5*rank_score(rank) + a6*step_position +
  a7*depth_norm + a8*n_l1_signals + a9*is_fallback)`. `rank_score` maps
  `Candidate.rank` onto [0,1] (1.0 for the top-ranked candidate);
  `step_position` and `depth_norm` are the candidate's step position and
  nesting depth each normalized against the whole trace; `n_l1_signals` is
  how many L1 detectors fired on that step; `is_fallback` flags the
  zero-evidence synthetic candidate `candidates.py` emits when nothing else
  fired. Current coefficients: `a0=-4.3970 a1=-0.0482 a2=0.2273 a3=-0.0892
  a4=0.0020 a5=0.1206 a6=2.1086 a7=1.1989 a8=0.2925 a9=-1.5935`
  (`DEFAULT_A0`-`DEFAULT_A9` in `confidence.py`, pinned by
  `tests/test_confidence.py`). The confidence floor (`_DEFAULT_MIN_CONFIDENCE`,
  also `CulpritConfig.min_confidence`) is **0.15**, down from 0.55.
  - **Why 0.15, from a precision-at-threshold table on the same 447-row
    population the model is fit on.** At threshold 0.15: 122 candidates
    committed (well over the ~15-count trustworthy floor), 23.8% precision
    (out-of-fold, averaged over 5 CV seeds) against a 10.3% unconditional
    base rate - roughly 2.3x lift, versus 1.4-1.8x at the thresholds below
    it. Raising the floor further (0.18: 41.7% precision) trades most of
    the remaining coverage (13.4% vs 27.3%) for a smaller, more volatile
    committed set; 0.15 was chosen as the lowest threshold that is both
    trustworthy and a real, not marginal, improvement over guessing.
  - **Production-feasibility constraint: no dataset-identity feature ships,
    even though one measured well in research.** A same-day investigation
    (scripts left uncommitted under `data/benchmarks/results/`,
    `richer_features_20260817.py` and `refit_richer_features_20260817.py`)
    found that adding `rank`/`step_position`/`depth_norm`/`n_l1_signals`
    lifts out-of-fold AUC from ~0.61 (the original 4-feature model) to
    ~0.74, and, critically, this lift **survives on TRAIL-only data alone**
    (0.547 -> 0.755 AUC per-dataset), which rules out the richer model
    merely learning "which benchmark is this" via a `dataset_trail`/
    `source_fallback` confound - the researcher explicitly tested a
    `dataset_trail` feature (it measured even higher, ~0.767 combined AUC)
    and excluded it anyway, because a real ingested production trace has no
    such label to read. `is_fallback` (`Candidate.source == "fallback"`) is
    a genuinely different feature from the dataset flag - real, available
    at inference time on production traffic - and was kept after refitting
    confirmed it improves AUC (~0.715 -> ~0.739) and precision on this
    population.
  - **How this state was reached.** Hand-set priors
    (`a0=0.0, a1=1.0, a2=0.5, a3=0.3, a4=1.5`) were replaced 2026-08-17 by
    I7's first fit: logistic regression on 447 real adjudication rows (46
    positive) using only `model_confidence`, `prior`, `agreement`, and
    `evidence_density`, giving `a0=-2.6065 a1=-0.0285 a2=0.7678 a3=0.6492
    a4=0.0053` (Brier 0.878 -> 0.090, ECE 0.886 -> 0.004 on that
    population; 5-fold CV Brier 0.0909 tracked in-sample 0.0902, not
    degenerate). That fit's real finding was that cite-check and raw model
    confidence turned out not to matter, while prior and L1 agreement
    carried nearly all the signal - the opposite of the hand-set
    assumption. But a brute-force scan of that formula's output over its
    whole realistic input domain topped out around 0.28-0.29, permanently
    below the 0.55 floor in place at the time, so `select_diagnosis`
    abstained on every trace regardless of evidence strength - discovered
    while fixing `tests/test_pipeline.py` post-I7 and left as an open issue
    pending a maintainer decision, since editing the coefficients or floor
    was out of scope for that task. The richer-feature refit documented
    above is that decision: it fits ten coefficients on the wider feature
    set (raising the ceiling to ~0.83-0.91 over the realistic input domain)
    and picks a new floor from measured precision, closing the ceiling-vs-
    floor gap instead of only re-deriving the same four terms.
    `tests/test_pipeline.py`'s
    `test_diagnose_wires_l3_and_commits_a_well_supported_root_cause`
    (through two prior names, `..._and_surfaces_a_committed_root_cause`
    then `..._and_correctly_abstains_below_the_fitted_confidence_ceiling`)
    now asserts a correct commit again under a well-supported synthetic
    scenario, verified against the actual coefficients rather than assumed.
  - **What remains thin versus solid.** The whole fit, at every stage, rests
    on the same 447 rows and 46 positives - genuinely small, and the
    biggest reason to treat any of these coefficients as provisional. The
    TRAIL-only AUC replication is the strongest evidence the richer
    features generalize rather than overfitting 447 rows; the combined-
    population AUC number alone would be weaker evidence on its own.
  - **This refit does not touch the dominant unsolved problem.** The same
    research pass measured that 85.9% of traces (269 of 313 scored
    benchmark diagnoses) never have the correct step in their L3 candidate
    shortlist at all - a structural ceiling in L1/L2 candidate recall that
    no L3 calibration change, including this one, can cross. Calibrating
    confidence better on the 14.1% of traces where the right candidate is
    even reachable is a real but secondary improvement; candidate recall is
    still the number that matters most and is still unaddressed.

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

- **`jobs._conn_fn_from_env` fixed to recycle connections, mirroring `bench.pooled_conn_fn`'s already-fixed bug (commit `a138295`).** It previously returned a bare `pool.getconn` with no recycling of any kind - the identical defect `bench.py` hit empirically (pool exhaustion after ~3 traces, since a diagnosis routes through up to ~28 checkouts via L2's reference loads). Within `jobs.py` this bites per-job (each job function builds its own fresh pool via `_conn_fn_from_env()`), and every checked-out connection a finished job never returns leaks a real server-side Postgres connection for the life of the worker process. Fixed by having `_conn_fn_from_env` return `(conn_fn, close)`; each of the four call sites (`diagnose_trace_job`, `recluster_job`, `ingest_trace`, `read_diagnoses_for_trace`) now only builds its own pool when the caller did not inject a `conn_fn`, and releases every connection it issued in a `finally` block via `pool.putconn` (not `conn.close()`, which this psycopg_pool version does not treat as returning the connection to the pool - same empirical finding `bench.py`'s docstring records) before `pool.close()`. Shape deliberately differs from `bench.py`'s per-trace `recycle()` loop: `jobs.py` builds one fresh pool per job call rather than amortizing one pool across a run, so there is nothing to recycle mid-job - the pool is instead sized to one job's worst case (`max_size=32`, mirroring `bench.pooled_conn_fn`'s override of `db.make_pool`'s default of 10) and released once at the end, success or exception alike. Tests added in `tests/test_jobs.py` prove connections are `putconn`'d (not merely dropped) even when the job's work raises, and that the pool is sized to 32.

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

**Real benchmark scores, 2026-08-17 (Integration task I5).** The harness
scored the full public releases, not samples: TRAIL (129 traces, 763
annotated errors, GAIA + SWE Bench splits) and Who&When (184 logs), every
trace persisted through the production tables and diagnosed inline by
`pipeline.diagnose` on `gemini/gemini-2.5-flash-lite`. Total Gemini spend for
all four runs plus the earlier smoke sample: roughly $0.23.

*(I5's benchmark path did not persist diagnoses at the time, see "L3
adjudication" above; the numbers below are I5's original run. I7 re-ran the
harness the same day after fixing that, non-ablated only - see "L3
adjudication" for the persisted-data fit. The re-run's headline numbers are
close but not identical, real-LLM nondeterminism: TRAIL exact accuracy 0.025
-> 0.014, candidate_recall@5 0.065 -> 0.078, earliness_error +1.689 ->
-0.015; Who&When abstention 0.152 -> 0.130. Joint accuracy stayed 0.000 on
both. Full I7 re-run numbers:
`data/benchmarks/results/bench_runs_20260817T210006Z.txt`.)*

TRAIL, all 763 annotated errors (a prediction is credited if it matches ANY
annotated error of its trace):

| Metric | Value |
|---|---|
| Abstention rate | 0.372 |
| Exact step accuracy | 0.025 |
| Tolerance accuracy @1 / @3 | 0.221 / 0.366 |
| **Joint accuracy (step AND class)** | **0.000** |
| Class accuracy | 0.014 |
| Candidate recall @1 / @3 / @5 | 0.008 / 0.065 / 0.065 |
| Earliness error (committed cases) | +1.689 steps |
| Brier / ECE | 0.957 / 0.959 |

TRAIL, earliest annotated error only (129 primary cases): abstention 0.295,
exact 0.000, tolerance@3 0.434, recall@5 0.033, earliness +4.527 steps.

Who&When (184 cases, every case primary): abstention 0.152, exact 0.033,
tolerance @1 / @3 0.120 / 0.250, joint 0.000, class accuracy 0.114 (ground
truth class is always `unknown` by construction, so this only measures how
often the system says `unknown` back), recall@5 0.038, earliness error
+13.436 steps, Brier 0.945.

**L2 ablation on real data: the delta is zero.** `--ablate l2` moves TRAIL
exact 0.025 -> 0.025, joint 0.000 -> 0.000, recall@5 0.065 -> 0.092; Who&When
exact 0.033 -> 0.033, recall@5 0.038 -> 0.039. L2 abstains
`insufficient_references` on essentially every real trace: the reference pool
holds almost no successful runs of comparable tasks, so the contrastive
layer contributes nothing under production-like conditions and removing it
changes nothing measurable. This does not refute L2, it says the layer is
untestable here until the corpus accumulates real successful traces.

**How to read these numbers, and why they are not tuned away.** The binding
constraint is candidate recall, not L3 adjudication quality: the true step
reached L3's shortlist in 6.5 percent (TRAIL) and 3.8 percent (Who&When) of
cases, so end-to-end accuracy is capped near zero no matter how good
adjudication is. L1's detectors, built and unit-tested against `synth.py`
traces, fire sparsely on real OpenInference traces. Second, when the system
does commit it is confidently wrong: Brier ~0.96 with calibrated confidence
saturating near 1.0 against roughly 2-3 percent accuracy, meaning the
hand-set calibration priors (see L3 adjudication above) are now measured to
be wildly overconfident. Third, the earliness error is positive on both
benchmarks (+1.7 TRAIL, +13.4 Who&When): the system blames steps downstream
of the true cause, the precise failure mode the product exists to eliminate,
now measured on real data rather than asserted. Context, not excuse: the
TRAIL paper's best LLM judge localizes around 11 percent and Who&When's best
reported step accuracy is around 14 percent, both using trace-specific
frontier-model prompts; this pipeline is generic and costs roughly $0.0005
per trace.

**Dataset reality vs the adapter's guesses (I5 record).** Source: HuggingFace
`PatronusAI/TRAIL` (gated, pulled via the public ModelScope mirror of the
same files, Apache-2.0) and `github.com/mingyin1/Agents_Failure_Attribution`.
Every guessed key in `TRAIL_CATEGORY_MAP` was wrong - the real taxonomy is 21
title-case categories with spelling/casing variants in the wild, mapped with
documented plurality judgment calls in `trail.py`. Data quality findings: the
public release has 131 traces (100 GAIA, 31 SWE Bench), not the paper's 148;
2 traces carry zero error annotations; 2 of 765 annotations reference spans
that do not exist; 1 annotation file has a literal trailing-comma syntax
error upstream; 1 trace emits the same span twice (deduped in the adapter).
Who&When's real schema needed only a thin prepare script; its `mistake_step`
is a direct history index as the adapter assumed.

## What was verified vs what remains unverified

A reader must not be able to mistake this for a system whose scores are good.
Real TRAIL and Who&When scores now exist and they are close to floor - see
"Measured results so far" for the numbers, joint accuracy 0.000 on both
benchmarks among them.

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

**Verified 2026-08-17:**

- **Benchmark harness scored the real, full TRAIL and Who&When datasets** (129
  traces / 763 annotated errors and 184 logs, respectively), not the
  hand-written 3-record fixtures used to build the adapters. See "Measured
  results so far" above for the numbers; not duplicated here.
- **Confidence coefficients are now fitted, not hand-set - twice over.**
  `bench.py` was wired to persist diagnoses and benchmark cases, the
  harness was re-run, and `confidence.py`'s coefficients were fit by
  logistic regression against 447 real adjudication rows and cross-
  validated: first I7's 4-feature fit (`DEFAULT_A0`-`DEFAULT_A4`), then the
  same day's richer-feature refit (`DEFAULT_A0`-`DEFAULT_A9`) that resolved
  the ceiling-below-floor issue the first fit left behind. See "L3
  adjudication" above for the full history, the fit's sample size, and the
  honest caveat about what population it does and doesn't generalize to. Do
  not describe the *system's end-to-end accuracy* as calibrated beyond what
  this fit measured - only the confidence-score mapping was fit, not a
  claim that the pipeline finds the right answer more often, and the
  85.9%-of-traces candidate-recall ceiling (see "L3 adjudication") remains
  the dominant unsolved problem regardless of how well confidence is
  calibrated on the traces that do reach L3 with a correct candidate.

**Still unverified:**

- **Failure path through the full CLI+DB pipeline.** A synthetic failure was
  adjudicated in-process in the live smoke test, and the clean OTLP fixture ran
  through the CLI, but a real failure trace has not yet been ingested,
  diagnosed, persisted, and read back via `culprit show`.
- **Queued-worker path.** See the known gotcha below: `culprit worker` cannot run
  on Windows.

## Status

Phase 2 closeout landed on `main`. Integration items I1-I7 are done: I7
(calibration) fixed the benchmark harness's missing persistence, re-ran it
against real TRAIL/Who&When data, and fit `confidence.py`'s coefficients
against the result. A same-day follow-up (I7b in `AI_docs/PHASES.md`)
refit those coefficients again on a richer, production-viable feature set
after I7's fit turned out to have an unreachable confidence floor; see
"L3 adjudication" above for the current state and full history. See
`AI_docs/PHASES.md` for the authoritative resume point, the Phase 1
workstream table, and the Phase 2 integration checklist.
