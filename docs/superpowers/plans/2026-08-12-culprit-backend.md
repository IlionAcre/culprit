# culprit Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Not where it failed. Where it broke.**

**Goal:** Build a backend that ingests a failed LLM agent trace and identifies
the step where the run left the space of trajectories that would have succeeded,
plus the reason it did.

**Architecture:** A layered cascade. Deterministic detectors and contrastive
sequence alignment against successful runs narrow thousands of steps down to
five candidates; targeted LLM adjudication then judges only those five with
bounded context. Postgres with pgvector is the single datastore; Redis and RQ
carry long-running analysis off the request path. Published benchmarks show
long-context models fail at whole-trace debugging, so narrowing before
adjudicating is the central bet.

**Tech Stack:** Python 3.12, uv, FastAPI (sync), Pydantic v2, Postgres 16 +
pgvector, psycopg 3, Alembic, Redis + RQ, litellm, fastembed
(bge-small-en-v1.5), scikit-learn HDBSCAN, pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **Python `>=3.12`.** `.python-version` pinned to `3.12`.
- **uv only.** `uv_build` backend, `src/culprit/` layout, `uv.lock` committed.
  Dependencies use `>=` floors only, alphabetized. Dev deps in
  `[dependency-groups] dev`, never `[project.optional-dependencies]`.
- **Every module under ~200 lines.** Flat layout, one module per concern. Only
  an open plugin set earns a subpackage: `vocab/`, `detectors/`, `benchmarks/`.
- **Zero `async def` anywhere**, including FastAPI routes. Fan-out is
  `ThreadPoolExecutor(...).map()`.
- **Pydantic v2 at I/O boundaries only**; `@dataclass` for internal computed
  results; `@dataclass(frozen=True)` for config. No `ConfigDict`, no validators,
  `X | None` never `Optional[X]`.
- **Config precedence: CLI flag > `culprit.toml` > hardcoded default.** Core
  logic modules keep defaults in their own signatures and never import
  `config`. Only `cli.py` and plugin registries read config.
- **`typing.Protocol` + module-level dict registry** for plugins, never ABCs.
- **`import litellm` then `litellm.completion(...)`.** The dotted form is
  mandatory for `monkeypatch.setattr("litellm.completion", ...)` mocking.
- **Per-item error isolation.** No batch operation crashes on one bad item.
  Failures become sentinel results carrying `error: str | None`. Applied at
  span, detector, candidate, and layer granularity.
- **All tests run fully offline by default.** `litellm` monkeypatched, no
  network, no API key. Postgres tests gate on `CULPRIT_TEST_DSN`.
- **No em dashes** in code, comments, docstrings, or prose. Use ` - `.
- **Exceptions** are flat, module-local, subclass `Exception` directly, named
  `<Domain><Verb>Error`, with a docstring explaining why they are raised rather
  than swallowed. `raise ... from e` / `from None` used correctly.
- **Logging** is stdlib `logging` with `logger = logging.getLogger(LOGGER_NAME)`
  per module and every call passing `extra={"event": "snake_case_name", ...}`.
- **Docstrings explain rationale**, name the motivating bug, and list rejected
  alternatives. They never restate the signature. Trivial modules get none.

---

## Context

Production LLM agents fail in ways that are close to undebuggable. A single run
emits thousands of trace lines, and the failure almost always surfaces several
steps downstream of what actually caused it. An engineer sees "the agent
returned the wrong answer at step 47" and has to reconstruct by hand that the
real cause was a malformed tool result at step 12 that the model then reasoned
over confidently for thirty-five steps.

Existing observability (Langfuse, LangSmith, Braintrust, Helicone) shows the
trace. None of them identify the cause. That gap is the product.

**culprit answers one question: at which step did this run leave the space of
trajectories that would have succeeded, and why.**

Built as a portfolio artifact demonstrating AI engineering depth, after a market
verification round (`Portfolio/research/market-verification-2026-08.md`)
established that the commercially obvious ideas in this space are already
served. The goal is a technically distinctive system with measurable results.

### Why this architecture and not the obvious one

Published benchmarks in agent failure attribution (TRAIL, Who&When, MAST,
AgenTracer) show that **even strong long-context models perform poorly when
handed a whole trace and asked what went wrong.** That single finding drives
everything.

The naive design is "put the trace in the context window and ask." It does not
work. So culprit narrows candidates deterministically first, then spends model
calls on a handful of pre-narrowed steps with bounded context around each.
Deterministic layers localize; the model adjudicates. This is also the cost
story: the expensive layer runs on roughly five steps per trace instead of
thousands.

---

## Prior art that informs specific decisions

Drawn from [eugeneyan/applied-ml](https://github.com/eugeneyan/applied-ml).
Listed only where an entry changes a design decision; the rest of that list is
recsys, search, and CV and is not relevant here. Read these before building the
layer they inform.

| Entry | Informs | What to take |
| :--- | :--- | :--- |
| **Data Validation for Machine Learning** (Google, 2019) | L1 detectors | The schema-as-versioned-artifact pattern. An anomaly is a deviation from an *explicit, versioned expectation*, not a heuristic someone tuned once. Directly justifies `layer_versions` on `Diagnosis` and a detector catalogue that is versioned and independently testable. Their anomaly taxonomy is a model for ours. |
| **Using deep learning to detect abusive sequences of member activity** (LinkedIn, 2021) | **L2 contrastive** | The closest prior art in the whole list. Sequential anomaly detection over user activity sequences is structurally the same problem as detecting an anomalous step sequence in a trace. Read before implementing `contrast.py`. |
| **Monitoring Data Quality at Scale with Statistical Modeling** (Uber, 2017) | L2 reference model | Comparing current observations against a historical distribution, including how they handle thin history. That is exactly the `insufficient_references` abstention case. |
| **Rules of Machine Learning** (Google, 2018) | Whole architecture | Rule 1: do not use ML where a heuristic suffices. This is the canonical citation for deterministic-first narrowing, and worth quoting in `CLAUDE.md` next to the L1/L3 split. |
| **Project RADAR: Intelligent Early Fraud Detection with Humans in the Loop** (Uber, 2022) | L3 abstention | Presenting ranked suspects to a human rather than a verdict, and the operational discipline around it. Informs the three abstention gates. |
| **ML Model Monitoring: 9 Tips From the Trenches** (Nubank, 2021) | Benchmark harness | Practical monitoring discipline, especially reporting metrics you would rather not see. |
| **Machine Learning: The High Interest Credit Card of Technical Debt** (Google, 2014) | Detector catalogue | Why an untested, unversioned pile of heuristics becomes unmaintainable. Argues for the strict per-detector test requirement in WS-C. |
| **Overton: A Data System for Monitoring and Improving Machine-Learned Products** (Apple, 2019) | Diagnosis versioning | Why re-running analysis must produce a *new* diagnosis alongside the old, which is why `diagnoses.trace_id` is deliberately not unique. |

**How this enriches the artifact**, beyond the engineering: citing the specific
industry paper behind each design decision is exactly the posture the Litmus
write-up already takes. It converts "I chose deterministic-first" into "I chose
deterministic-first, and here is the Google paper that argues for it." Record
these citations in `CLAUDE.md` next to the decisions they support, not in a
bibliography nobody reads.

---

## Decisions (with rationale)

To be carried into the project's `CLAUDE.md` in the established format.

- **Name `culprit`.** Names the deliverable: the step that broke the run.
  *Rejected: `postmortem`* (implies a written report), *`divergence`* (names the
  mechanism, not the product), *`tracer`* (collides with the observability tools
  this sits on top of).

- **Postgres with JSONB and pgvector as the single datastore.** Traces are large
  and nested, analysis needs cross-run analytical queries, clustering needs
  vector similarity. One datastore holds all three.
  - *Rejected: JSON files + DuckDB (the Litmus pattern)* - fine for Litmus's
    small run files, but trace volume, concurrent worker writes, and vector
    similarity all push past it.
  - **Deliberate departure.** Litmus's `CLAUDE.md` lists "don't introduce a
    database server" as a guardrail. Confirmed with the user before adopting.

- **psycopg 3 with `ConnectionPool`, no SQLAlchemy ORM.** The schema is mostly
  JSONB columns, bulk inserts, and vector operators, none of which an ORM
  improves. Alembic is used purely as a migration runner with hand-written DDL,
  no `--autogenerate`.
  - *Rejected: SQLAlchemy ORM* - would mean maintaining model definitions that
    duplicate the Pydantic schemas for no benefit.

- **Redis + RQ for the queue, separate worker entrypoint sharing the package.**
  Analysis takes minutes per trace and cannot run inside a request.
  - *Rejected: arq* - async-native, fights the sync-everywhere convention and
    breaks the established `monkeypatch.setattr` mocking patterns.
  - *Rejected: Celery* - heavier configuration surface than this needs.

- **Sync throughout, including FastAPI routes.** Matches Litmus exactly.
  Fan-out uses `ThreadPoolExecutor(...).map()`.

- **Ingest OTLP spans; normalize both attribute vocabularies.** Accept OTLP over
  HTTP (protobuf and JSON) plus direct file upload. Both OpenTelemetry GenAI
  conventions and OpenInference ride on ordinary OTel spans, so one transport
  covers both. OTel GenAI *agent* spans were still Development status as of
  mid-2026 while client spans stabilized; OpenInference is richer today and
  already dual-emits. Neither is safe to bet on exclusively.
  - Practical consequence: anyone already emitting OTel points a collector at
    the endpoint rather than instrumenting anything new.
  - **Raw attributes are never discarded**, so a later vocabulary module can
    backfill spans that failed to normalize. This is what makes normalization a
    safe bet rather than an irreversible one.

- **v1 ships L0-L3 plus L5. L4 and L6 are deferred but designed for.**
  L0 normalize/linearize, L1 deterministic detectors, L2 contrastive trajectory
  diff, L3 targeted adjudication, L5 clustering. Deferred: L4 counterfactual
  replay, L6 regression case generation. See "Where the deferred layers slot in"
  below; neither requires touching a v1 contract.

- **The benchmark harness is in scope for v1.** This is what turns the project
  from "a tool I built" into "a tool that scores X on TRAIL", matching the
  evidence-backed posture of the Litmus write-up.

- **Backend only.** No frontend.

---

## Stack

| Concern | Choice |
| :--- | :--- |
| Language / packaging | Python >=3.12, uv, `uv_build`, `src/` layout, `uv.lock` committed |
| API | FastAPI, sync `def` routes, single `api.py` |
| Models | Pydantic v2 at I/O boundaries, `@dataclass` internally |
| Storage | Postgres 16 + pgvector, psycopg 3 (`psycopg[binary,pool]`) |
| Migrations | Alembic, hand-written DDL |
| Queue | Redis + RQ |
| LLM | `litellm` (`import litellm` form, for mockability) |
| Embeddings | `fastembed` + `BAAI/bge-small-en-v1.5`, 384-dim, CPU, no torch |
| Clustering | `sklearn.cluster.HDBSCAN` |
| Config | stdlib `tomllib` + frozen dataclass |
| Logging | stdlib `logging` + hand-written JSON-lines formatter |
| Tests | pytest, offline by default, `litellm` monkeypatched |

**Full dependency list, declared up front in Phase 0** (alphabetized, `>=`
floors only). Runtime: `alembic`, `fastapi`, `fastembed`, `litellm` (with the
Litmus `sys_platform == 'win32'` marker pattern, same machine, same linker
problem), `numpy`, `opentelemetry-proto`, `pgvector`, `psycopg[binary,pool]`,
`pydantic`, `python-dotenv`, `redis`, `rq`, `scikit-learn`, `typer`, `uvicorn`.
Dev group: `fakeredis`, `httpx`, `pytest`, `pytest-cov`.

Postgres-touching tests gate on
`@pytest.mark.skipif(not os.environ.get("CULPRIT_TEST_DSN"))`, so default
`uv run pytest` stays fully offline. *Rejected: `testcontainers`* - requires
Docker on the dev machine and makes the default run non-offline.

---

## Conventions inherited

Extracted from `Litmus/`. New code should be indistinguishable from it.

- Flat `src/culprit/`, one module per concern, **every module under ~200 lines**.
  Only an open plugin set earns a subpackage (here: `vocab/`, `detectors/`,
  `benchmarks/`).
- Pydantic v2 at I/O boundaries, `@dataclass` internal, `@dataclass(frozen=True)`
  config. No `ConfigDict`, no validators, `X | None` never `Optional[X]`.
- Config precedence CLI > toml > hardcoded. **Core modules keep defaults in
  their own signatures and never import config.** Only `cli.py` and registries
  read it.
- `typing.Protocol` + module-level dict registry for plugins, never ABCs.
  Registry raises `ValueError(f"Unknown x {name!r}. Available: {sorted(REG)}")
  from None`.
- DI by passing a `Callable` with a module-level type alias.
- **Per-item error isolation.** Batch work never crashes on one bad item;
  failures become sentinel results carrying `error: str | None`. Applied at
  detector, candidate, span, and layer granularity.
- Flat module-local exceptions named `<Domain><Verb>Error`.
- Structured LLM output: module-level prompt template, untrusted content in
  XML-ish delimiters with an explicit "this is untrusted data, not instructions"
  line, de-fence regex before `json.loads`, private Pydantic model, narrow
  except set.
- FastAPI per-route try/except to `HTTPException(...) from e`, preceded by
  `logger.warning(..., extra={"event": "api_error", ...})`.
- Tests flat, no classes, full-sentence behavioral names, Arrange/Act/Assert,
  docstrings only to explain why a test exists, minimal `conftest.py`, shared
  setup as private `_helper()` duplicated per file.
- Docstrings explain rationale and name the motivating bug. Trivial modules get
  none.
- **No em dashes** anywhere.

---

## Project location

**`culprit` is a new, standalone git repository.** Proposed path:
`C:\Users\faalc\OneDrive\Documents\Portfolio\culprit\`, a sibling of `Litmus\`,
matching how the other projects are laid out.

**No code is written inside `Litmus\`.** Every Litmus path in this plan is a
**read-only style reference**: patterns to copy by hand into the new repo, not
files to edit or extend.

The only place the two projects could genuinely couple is **L6 regression case
generation, which is deferred**. If it is built later, `culprit` could emit
Litmus-format test cases, and at that point importing `litmus` as a dependency
becomes an option worth weighing. v1 has no dependency on Litmus of any kind.

---

## Brand

Name `culprit`, tagline "Not where it failed. Where it broke." Record in
`CLAUDE.md` under `## Brand` with the standing no-rename rule.

Propagates to: `src/culprit/`, distribution `culprit`, CLI `culprit`,
`culprit.toml`, `LOGGER_NAME = "culprit"`, `CULPRIT_*` env vars.

---

## Module decomposition

51 modules, all under the 200 line ceiling. Three subpackages, each an open
plugin set.

### Foundation-owned (contracts and infrastructure)

| File | Responsibility | LOC |
|---|---|---|
| `schemas.py` | `Trace`, `Span`, `Step`, `SpanKind`, `Outcome`, four per-kind payloads | 190 |
| `signals.py` | `Evidence`, `Signal`, `DivergenceCandidate`, `ContrastResult`, `Candidate`, `Adjudication`, `Diagnosis` | 165 |
| `taxonomy.py` | `FailureClass` StrEnum + descriptions, `.format()`ed into the L3 prompt | 95 |
| `config.py` | `CulpritConfig` frozen dataclass, `load_config()`, `ConfigError` | 175 |
| `logging_config.py` | `LOGGER_NAME`, `JsonFormatter`, `configure_logging()` | 75 |
| `llm.py` | `litellm` passthrough, no retries, no logging | 25 |
| `embed.py` | fastembed passthrough, `EmbedFn` alias, lazy model load | 65 |
| `db.py` | psycopg3 `ConnectionPool`, `ConnFn` alias, `register_vector` | 90 |
| `synth.py` | Synthetic trace generator + `inject(kind, at_step)` | 190 |
| `synth_results.py` | Synthetic `Signal`/`DivergenceCandidate`/`Diagnosis` factories | 120 |
| `pipeline.py` | Foundation ships signature + `NotImplementedError`; Integration fills | 15 → 180 |

### L0 ingestion (WS-A)

`otlp.py` (150), `vocab/base.py` (30), `vocab/registry.py` (55),
`vocab/otel_genai.py` (180), `vocab/openinference.py` (185), `normalize.py`
(170), `linearize.py` (180).

### Persistence (WS-B)

`store_traces.py` (190, includes the pgvector kNN reference query),
`store_diagnoses.py` (190).

### L1 detectors (WS-C)

`detectors/base.py` (55), `detectors/registry.py` (55), then one module per
detector family: `tool_errors.py`, `loops.py`, `context.py`, `schema.py`,
`flow.py`, `retrieval.py`, `provenance.py`, `timing.py` (100-180 each), plus
`run_detectors.py` (130).

### L2 contrastive (WS-D)

`fingerprint.py` (155), `signature.py` (175), `align.py` (180), `reference.py`
(190), `contrast.py` (190).

### L3 adjudication (WS-E)

`candidates.py` (145), `context_window.py` (170), `prompts.py` (130),
`adjudicate.py` (195), `confidence.py` (120).

### L5 clustering (WS-F)

`cluster_embed.py` (105), `cluster.py` (155), `cluster_label.py` (135).

### Service surface (WS-G)

`api.py` (195), `views.py` (130), `cli.py` (200), `queue.py` (85), `jobs.py`
(140), `worker.py` (55).

### Benchmarks (WS-H)

`benchmarks/base.py` (45), `benchmarks/registry.py` (50), `benchmarks/trail.py`
(180), `benchmarks/who_and_when.py` (180), `bench_score.py` (175).

> `cli.py` at 200 is achievable only because `views.py` absorbs output shaping
> and `pipeline.py` absorbs orchestration. Litmus's own `cli.py` is 383 lines,
> the one acknowledged exception to the rule. If it drifts, that is the expected
> place and it should not be forced.

---

## Canonical domain model

Pydantic v2 throughout; all of it crosses an I/O boundary (OTLP in, JSONB,
HTTP out).

**Enums.** `SpanKind`: LLM, TOOL, RETRIEVER, AGENT, CHAIN, EMBEDDING, GUARDRAIL,
UNKNOWN. `SpanStatus`: OK, ERROR, UNSET. `Outcome`: SUCCESS, FAILURE, UNKNOWN.

**Payloads.** `LlmPayload` (provider, model, request/response messages, tool
calls, finish_reason, token counts, temperature, `response_format_schema`),
`ToolPayload` (tool_name, call_id, arguments, result_text, result_len, is_error,
error_message), `RetrievalPayload` (query, documents, top_k),
`AgentPayload` (agent_name, role, input/output text, delegated_to).

**Span.** `trace_id`, `span_id`, `parent_span_id`, `name`, `kind`, `status`,
`status_message`, `start_ns`, `end_ns`, `vocabulary`, `attributes` (raw, always
retained), `payload` (union), `normalize_error`.

`payload` is a plain union, not a discriminated union, because the discriminant
lives on the sibling `kind` field and Pydantic needs the tag inside the member.
Resolution happens by `kind` in `normalize.py`.

**Trace.** `trace_id`, `source`, `outcome`, `agent_key`, `task_key`,
`task_goal`, `framework`, `root_span_id`, `span_count`, `step_count`,
timestamps, `metadata`, `ingest_error`.

`task_embedding` deliberately does **not** live on the Pydantic `Trace`. It is a
384-float storage concern that would bloat every API response and log line. It
exists only as a Postgres column.

**Step.** `trace_id`, `step_index` (dense, 0-based, semantic spans only),
`span_id`, `kind`, `actor`, `depth`, `tree_path`, `signature`, `summary`,
timestamps, `duration_ms`, `collapsed_span_ids`.

### Linearization, three rules that all matter

1. **DFS pre-order, siblings sorted by `(start_ns, span_id)`.** The `span_id`
   tie-break is not cosmetic. OTel exporters routinely emit siblings with
   identical millisecond-truncated start times, and without a deterministic
   secondary key the same trace linearizes differently on two runs, silently
   poisoning every L2 alignment. Dedicated test required.

2. **Only semantic spans become steps.** LLM, TOOL, RETRIEVER, AGENT produce
   steps; CHAIN, EMBEDDING, GUARDRAIL, UNKNOWN do not, and their ids go into the
   enclosing step's `collapsed_span_ids`. LangChain and LlamaIndex emit 3 to 10
   framework spans per real action; without collapsing, step indices are
   dominated by framework noise and "step 47" means nothing to a human.
   *Rejected: keep every span and down-weight in the signature* - preserves the
   noise in the one number the product reports.

3. **Orphan spans re-parent to root** and are flagged in
   `metadata["orphan_span_ids"]`, never dropped.

Both representations coexist and neither is derived on the fly. `tree_path` and
`depth` let the tree be reconstructed from the linear sequence, so no analysis
layer re-traverses.

---

## The Signal / Diagnosis contract

This is the interface every parallel workstream compiles against.

```python
@dataclass(frozen=True)
class Evidence:
    span_id: str
    step_index: int
    field: str          # dotted path, e.g. "payload.result_text"
    excerpt: str        # <= 400 chars, head-and-tail truncated
    numeric: float | None = None

@dataclass
class Signal:                    # L1 emits
    detector: str
    step_index: int              # -1 for whole-trace or sentinel
    span_id: str | None
    severity: float              # 0.0 to 1.0
    category: str                # FailureClass HINT, not a verdict
    message: str
    evidence: list[Evidence]
    error: str | None = None
```

A detector that raises produces exactly one sentinel `Signal` with `error` set
and is excluded from downstream ranking; the other 19 still run. `category` is
explicitly a hint: L1 knows a tool returned empty, not whether that caused the
failure. Conflating the two is how deterministic detectors become
false-positive machines.

```python
@dataclass
class DivergenceCandidate:       # L2 emits
    step_index: int
    span_id: str
    divergence_score: float      # the ranked quantity
    cliff_delta: float           # loss of achievable future at this step
    surprisal: float             # nats, Markov model
    profile_surprisal: float     # nats, consensus profile
    reference_count: int
    observed_signature: str
    expected_signatures: list[tuple[str, float]]   # top-3 with probability
    nearest_reference_trace_ids: list[str]
    alignment_op: str            # "mismatch" | "insertion" | "deletion"
    error: str | None = None

@dataclass
class ContrastResult:
    candidates: list[DivergenceCandidate]
    reference_count: int
    abstained: bool
    abstain_reason: str | None

@dataclass
class Candidate:                 # L3 consumes
    step_index: int
    span_id: str
    rank: int
    prior: float
    source: str                  # "l1" | "l2" | "both" | "fallback" | later "l4"
    signals: list[Signal]
    divergence: DivergenceCandidate | None

@dataclass
class Adjudication:              # L3 emits, one per candidate
    step_index: int; span_id: str
    is_root_cause: bool
    failure_class: str
    confidence: float            # model self-reported
    calibrated_confidence: float
    rationale: str
    counterfactual: str          # what a successful run would have done here
    cited_step_indices: list[int]
    abstained: bool
    model: str
    prompt_tokens: int | None; completion_tokens: int | None
    cost_usd: float | None
    error: str | None = None
```

`Diagnosis` is **Pydantic**, not a dataclass, because it crosses two I/O
boundaries:

```python
class Diagnosis(BaseModel):
    diagnosis_id: str
    trace_id: str
    created_at: datetime
    root_cause_step_index: int | None
    root_cause_span_id: str | None
    failure_class: str | None
    confidence: float             # the winning Adjudication's raw confidence
    calibrated_confidence: float  # the winning Adjudication's calibrated value
    abstained: bool
    abstain_reason: str | None
    rationale: str
    counterfactual: str
    candidates_considered: int
    signals: list[Signal]
    divergences: list[DivergenceCandidate]
    adjudications: list[Adjudication]
    layer_versions: dict[str, str]
    degraded_layers: list[str]
    error: str | None = None
```

Both confidence fields are carried, matching the `diagnoses` table columns.
Reporting only the calibrated value would hide how much the calibration moved
it, which is exactly the number needed to tell whether calibration is helping.

### Module-level type aliases (foundation-owned)

Every injection seam in the system, defined once so no workstream invents its
own name:

```python
# llm.py
CallFn = Callable[[str, str], tuple[str, float, float]]
# (model, prompt) -> (raw_output, latency_ms, cost_usd)

# embed.py
EmbedFn = Callable[[list[str]], list[list[float]]]

# db.py
ConnFn = Callable[[], Connection]

# signals.py  (WS-D consumes this instead of importing store_traces)
NeighborFn = Callable[[list[float], int], list[str]]
# (query_embedding, k) -> ordered trace_ids of nearest successful runs
```

`degraded_layers` is the isolation rule at trace level: if L2 blew up you still
get an L1+L3 diagnosis, and the response says so rather than quietly presenting
a weaker answer as a full one. Same conviction as Litmus's
`has_mismatched_cases`.

---

## L1 detector catalogue

All 20 detectors take a shared `DetectorContext` (frozen dataclass: trace,
steps, `spans_by_id`, `args_hash_by_step`, `result_hash_by_step`,
`signature_runs` run-length encoded, `token_series`, `tool_universe`,
`goal_terms`). Building it once in `run_detectors.py` rather than 20 times is
the difference between L1 costing 5ms and 200ms.

| Family | Detectors |
|---|---|
| `tool_errors.py` | `tool_error`, **`empty_tool_result`**, **`error_swallowed`** |
| `loops.py` | `oscillation` (A B A B, period 2-4, >=2 cycles), `repeated_identical_action`, `retry_storm` (repeats with identical result hash = zero progress) |
| `context.py` | `context_overflow`, **`silent_history_truncation`**, `step_budget_exhausted` |
| `schema.py` | `output_schema_violation`, `tool_arg_malformed`, `hallucinated_tool` |
| `flow.py` | `premature_termination`, `missing_verification`, `duplicate_delegation` |
| `retrieval.py` | `unused_retrieval` (<15% token overlap with next prompt), `low_score_retrieval`, `goal_token_drift` (local embedder, free) |
| `provenance.py` | **`parameter_drift`** |
| `timing.py` | `stall_timeout` (>5x median for that signature) |

**The four bolded ones are the point of the product.** They share a property:
the trace looks entirely healthy at that step, status OK, no exception, and the
damage only becomes visible many steps later.

- `empty_tool_result`: status OK but result in `{"", "[]", "{}", "null",
  "No results found"}`. Evidence includes the *next LLM step's* first 200 chars
  so the reader sees what the agent did with nothing.
- `silent_history_truncation`: `prompt_tokens` drops >30% between consecutive
  LLM steps of the same actor while the conversation is monotonically growing.
  A framework silently dropping history, invisible in every existing tool.
- `parameter_drift`: an identifier-shaped token in tool arguments with no
  substring provenance in the goal or any prior step output. Catches fabricated
  order IDs, invented paths, made-up URLs.
- `error_swallowed`: a tool error or empty result followed within 2 steps by an
  LLM step asserting a positive result with no failure language.

These four get the heaviest test coverage.

---

## L2 contrastive algorithm

The differentiator. Five stages.

### Task fingerprinting

- `agent_key` = `sha1(framework | sorted(agent_names) | sorted(tool_universe))`.
- `task_key` = `sha1(normalize(task_goal))` where normalize lowercases,
  collapses whitespace, and regex-replaces UUIDs, integers, ISO dates, emails,
  URLs with typed placeholders. "Refund order 88213" and "Refund order 90114"
  become the same task, which is correct.
- `task_embedding` = 384-dim bge-small vector of the raw goal.

**Reference pool resolution:** exact `(agent_key, task_key, outcome='success')`
→ if fewer than 5, fall back to `agent_key` + pgvector kNN on `task_embedding`
at cosine distance <= 0.18, capped at 25 → **if still fewer than 3, L2 abstains**
with `abstain_reason="insufficient_references"`.

That third branch matters. A contrastive method with two references produces
confident nonsense. Abstention propagates into `degraded_layers`.

*Rejected: offline clustering of tasks into types* - adds a batch job, a
staleness window, and a cold-start problem. Query-time kNN is always current and
costs one indexed query.

### Step signature

**Coarse token** (the alignment alphabet):
`tool:{actor}:{tool_name}:{ok|err|empty}`,
`retr:{actor}:{query_shape}:{hit|miss}`, `agent:{actor}:{invoke|return}`, and
for LLM steps **the decision made, not the model that made it**:
`llm:{actor}:call:{tool_name}` / `llm:{actor}:answer` / `llm:{actor}:plan`.

That last rule is load-bearing. Signed by model name, every LLM step is the
identical token and carries zero alignment information, collapsing the method.

**Fine feature vector** for substitution scoring, 8 features
`[kind_match, actor_match, tool_match, outcome_match, arg_key_jaccard,
arg_value_jaccard, 1-|depth_delta|/4, retrieval_docid_overlap]` with weights
`[.25, .15, .25, .15, .08, .05, .04, .03]` giving `sim(a,b)` in `[0,1]`.
Substitution score is `2*sim - 1`, so identical is `+1`, unrelated `-1`.

### Reference model

Cached per `(agent_key, task_key, pool_hash)`.

1. **Consensus profile by center-star progressive alignment.** Highest
   sum-of-pairs reference becomes the center; align all others to it; merge into
   columns holding Laplace-smoothed `P(sig | column)` plus `P(gap | column)`.
   Yields "at this point, 82% called `search_orders`, 11% `get_customer`, 7%
   skipped".
   *Rejected: profile HMM with Baum-Welch* - needs far more than 25 sequences to
   fit without overfitting, adds a dependency, and is not directly
   interpretable. The profile's output is quoted verbatim to L3 and to the user,
   so interpretability is a requirement.
2. **First-order signature Markov model** with add-0.5 smoothing. Gives
   position-independent local surprisal, complementing the profile's
   position-dependent view. A step can be normal-in-general but wrong-here, or
   fine-here but never-seen-anywhere, and these two models separate those cases.
3. **Count envelopes**: median and MAD of total step count and per-signature
   occurrence count. This is what stops the retry detectors firing on normal
   behavior: a tool successful runs call three times is not a retry storm at
   three calls.

### Alignment

**Needleman-Wunsch with affine gaps (Gotoh, three DP matrices M/Ix/Iy).** Gap
open -1.0, extend -0.2.

Affine because an inserted 5-step retry burst is structurally *one* event, not
five; linear gaps would penalize it five times and drown the real divergence.

O(n·m) per reference. Post-collapse traces run 20-200 steps, so 200 × 200 × 25
is 1M cells, a few hundred ms in plain Python. **Implement readably first**;
numpy anti-diagonal vectorization only if profiling demands it. A subtle
traceback bug is invisible in aggregate metrics, so clarity buys more than speed.

*Rejected: Smith-Waterman* - local alignment deliberately discards the divergent
tail, which is the entire object of study. *Rejected: DTW* - assumes a monotone
warp of a continuous signal with no insertion/deletion semantics; agent traces
have genuine insertions and deletions, which is edit distance. *Rejected:
Zhang-Shasha tree edit distance* - O(n²m²) and the reasoning we want is
sequential; tree structure survives as *features* inside the signature.
*Rejected: `difflib.SequenceMatcher`* - no substitution scoring, so 95%-similar
steps count as a total mismatch. Keep it as a sanity baseline in tests.

### Divergence detection and scoring

**Residual alignability.** Run the forward Gotoh pass and a **reverse pass**.
The reverse pass gives, for free, the optimal alignment score of every suffix
`F[i:]` against `R_j[k:]`. Define `A(i) = max_j A_j(i)`, the best future the run
still has available at step i across all successful references, and
`C(i) = A(i-1) - A(i)`, the **cliff**: how much achievable future step i
destroyed.

`A(i)` is literally an operationalization of "the space of trajectories that
would have succeeded", and `C(i)` is the moment the run left it. One extra
O(n·m) pass, not a re-alignment per position.

```
D(i) = 0.40 * norm(C(i))
     + 0.20 * norm(surprisal(sig_i | sig_{i-1}))
     + 0.20 * norm(-log P_profile(sig_i | column(i)))
     + 0.15 * max_l1_severity_at(i)
     - 0.05 * (i / n)
```

The negative last term is an explicit **earliness prior**: given two
equally-scoring steps, the earlier is the cause and the later is the echo.
Weights are hardcoded in `contrast.py`'s signature per the layering rule; only
`cli.py` may override from config.

**Two post-filters, both essential:**

1. **Point-of-no-return gate.** Only steps where `A(i) < A(0) * 0.85` are
   eligible. A mismatch the run *recovers from* has `A` bounce back and is
   excluded by construction. This removes most of the false-positive volume that
   naive diff approaches produce.
2. **Earliest-of-plateau collapse.** Consecutive steps scoring within 0.05 of
   each other collapse to the earliest. This is the direct algorithmic answer to
   "the failure surfaces downstream of its cause": a cause and its consequences
   form a scoring plateau, and the cause is its left edge.

Top 5 by `D(i)`. For each, `expected_signatures` is the top-3 of
`P_profile(· | column(i))`, producing the highest-value string in the product:

> Step 14: successful runs called `search_orders` here (82%). This run called
> `refund_order`.

---

## L3 adjudication

**Candidate merge.** Union L1 and L2 steps by index.
`prior = max(l2_score, max_l1_severity) + 0.15 if both sources fire`, capped at
1.0. The co-location bonus is the point of running two independent layers.
Sort by prior desc, tie-break earlier step, truncate to 5. Zero candidates on a
FAILURE trace emits one synthetic candidate at the terminal step with
`source="fallback"`, so L3 always examines something concrete.

Five, because the whole architecture exists to avoid handing a model a whole
trace. Five focused calls at ~8k tokens each cost less and perform better than
one 200k-token call, which is what TRAIL and Who&When report.

**Context packet per candidate**, hard capped at 12k tokens, four sections:
task goal (1500 chars); **the spine** (one line per step across the whole trace,
~4k tokens for 200 steps, giving global orientation without dumping payloads);
**the zoom window** (full payloads for steps i-2 to i+2, truncated
**head-and-tail** because the end of a tool result is very often where the error
text lives); and **terminal failure evidence** always, since judging whether
step 14 caused the failure requires knowing what the failure was. Budget
enforcement drops the middle of the spine before touching the zoom window.

**One call per candidate**, fanned out with
`ThreadPoolExecutor(max_workers=4).map()`. The per-candidate function never
raises; a parse or API failure becomes a sentinel abstained `Adjudication`.
*Rejected: one call with all five candidates* - reintroduces exactly the
long-context degradation the research warns about, lets the model anchor on the
most verbose payload, and destroys the independence that makes cross-candidate
confidence comparison meaningful.

**Taxonomy: 21 classes** frozen in `taxonomy.py` as a StrEnum with one-line
descriptions that are `.format()`ed into the prompt, so prompt and code cannot
drift. Groups: specification/planning (3), tool/environment (5), information
handling (3), control flow (3), multi-agent (3), verification (2), output (1),
plus `unknown`.

Prompt follows the Litmus structured-output pattern exactly, including the
untrusted-content framing, which is not optional here: the input is a production
agent trace and may by definition contain adversarial content that reached the
agent.

**Three independent abstention gates:**
1. **Model self-abstention.** `is_root_cause: false` is presented as a correct
   and expected answer. Telling the model that "none of these" is valid is the
   difference between a classifier and a rubber stamp.
2. **Confidence floor** (0.55 calibrated). Nothing clears it, the diagnosis
   abstains and returns ranked suspects.
3. **Ambiguity margin.** Top two within 0.08 *and* different failure classes
   abstains with "ambiguous between X and Y", surfacing both. A coin flip
   presented as a verdict is worse than no verdict.

**Calibration.**
`calibrated = sigmoid(a0 + a1*logit(model_conf) + a2*prior + a3*agreement +
a4*evidence_density)`, where `agreement` is whether the chosen class matches a
co-located L1 hint and `evidence_density` is the fraction of
`cited_step_indices` actually present in the context packet. **A rationale
citing step 47 when only 12-16 were shown is direct evidence of confabulation**
and should crush confidence; this cheap cite-check is the highest-value term.

v1 ships hand-set coefficients documented as **uncalibrated priors, not fitted
values**. `bench_score.py` reports Brier, ECE, and reliability points from day
one so they can be fit by logistic regression once labeled data exists. Claiming
calibration you have not measured is exactly what the Litmus decision log calls
out repeatedly.

---

## L5 clustering

**Embed a canonical diagnosis card**, not the trace and not the rationale alone:
failure class, step signature, expected vs observed, detector names, and the
first 300 chars of rationale with IDs, UUIDs, numbers, and dates regex-masked.
The masking is load-bearing: without it, cluster structure is dominated by order
IDs and timestamps and you get 400 clusters of one trace each. A test asserts
two diagnoses differing only by order ID render byte-identical cards.

**`BAAI/bge-small-en-v1.5` via `fastembed`.** ONNX, ~130 MB, 384-dim, CPU-only,
zero API cost, **no torch**. *Rejected: `sentence-transformers`* - pulls torch,
~2.5 GB, contradicts this codebase's precedent of avoiding heavy ML deps for one
feature. *Rejected: `litellm.embedding`* - per-call cost over every diagnosis
plus a network dependency that breaks the offline-tests guarantee. *Rejected:
bge-base (768)* - 3x index size for marginal gain at this corpus size.

One model instance in `embed.py` serves both `traces.task_embedding` and
`diagnoses.card_embedding`.

**`sklearn.cluster.HDBSCAN`** (in sklearn since 1.3, no separate dep),
`metric="euclidean"` over **L2-normalized** embeddings (monotonically equivalent
to cosine, sidesteps sklearn HDBSCAN not accepting cosine), `min_cluster_size=5`,
`min_samples=3`, `cluster_selection_method="eom"`.

*Rejected: KMeans* - requires k and forces every diagnosis into a cluster.
*Rejected: agglomerative + threshold* - threshold as arbitrary as k, still no
noise concept. HDBSCAN's `-1` noise label is a feature: a novel failure mode
should surface as unclustered, because "we have never seen this" is useful
information. **Deliberately not UMAP-then-HDBSCAN**, the standard recipe: adds
umap-learn plus numba for gains that only matter at scale this does not have.
Documented as the first knob past a few thousand diagnoses.

**One LLM call per cluster, never per trace.** Input is the **medoid** card plus
4 cards sampled at increasing distance, so the label covers the cluster's spread
rather than only its center, plus the failure-class histogram. Re-label only
when the medoid changed or size grew >50% since `labeled_at`. A test asserts the
call count equals exactly the number of clusters; that is the cost-control
invariant.

---

## Postgres schema

**`traces`**: `trace_id` PK, `source`, `outcome`, `agent_key`, `task_key`,
`task_goal`, `task_embedding vector(384)`, `framework`, `root_span_id`, counts,
timestamps, `metadata jsonb`, `ingest_error`.
Indexes: btree `(agent_key, task_key, outcome)`; **partial HNSW** on
`task_embedding` `WHERE outcome = 'success'` (`vector_cosine_ops`, m=16,
ef_construction=64); btree `(ingested_at DESC)`.
The partial index is a real win: references come only from successes, so
indexing failures wastes half the index and slows every build.

**`spans`**: PK `(trace_id, span_id)` **composite, because OTel span ids are
only unique within a trace**, not globally. Getting this wrong produces
collisions that appear months in. Plus `parent_span_id`, `name`, `kind`,
`status`, `status_message`, `start_ns bigint`, `end_ns bigint`, `vocabulary`,
`attributes jsonb`, `payload jsonb`, `normalize_error`.
Indexes: `(trace_id)`, `(trace_id, parent_span_id)`.

JSONB versus typed, drawn deliberately: `kind`, `status`, timestamps are typed
because every query filters on them; `attributes` and `payload` are JSONB
because their shape is genuinely open across two vocabularies plus vendor
extensions, and flattening would force a migration for every new attribute any
tracing library invents. A GIN index on `attributes` is **deferred**: ~30% of
table size, and nothing in v1 queries attributes by value.

**`steps`**: PK `(trace_id, step_index)`, all typed columns, no JSONB. This is
L2's hot path and every field is queried. Index on `(signature)` for Markov and
profile counting.

**`signals`**, **`divergences`**: bigserial PK, `(trace_id)` indexed, evidence
and expected-signatures as JSONB (small heterogeneous records, never queried
field by field).

**`diagnoses`**: `diagnosis_id uuid` PK, `card_text`, `card_embedding
vector(384)`, `layer_versions jsonb`, `degraded_layers text[]`, `cluster_id`.
HNSW on `card_embedding`; btree on `(trace_id)`, `(failure_class)`,
`(cluster_id)`.
**`trace_id` is deliberately not unique.** Re-running analysis after improving a
detector must produce a new diagnosis alongside the old, and `layer_versions` is
what makes the history comparable. Without it you cannot answer "did that change
help", which is the main reason to have a benchmark harness.

**`adjudications`**: per-candidate audit trail plus cost accounting, which is
what lets you prove narrow-then-adjudicate is cheaper than long-context.

**`clusters`**: includes `run_id uuid` because clustering is a batch recompute
and without it there is no way to distinguish current from stale.

**`benchmark_cases`**: `case_id` PK, `benchmark`, `trace_id` FK,
`ground_truth_step_index`, `ground_truth_span_id`, `ground_truth_class`,
`raw jsonb`.

**HNSW not IVFFlat**: IVFFlat needs a representative training set at build time
and degrades as the corpus outgrows its list count; traces and diagnoses
accumulate continuously.

**Migration ordering**: `CREATE EXTENSION IF NOT EXISTS vector` first, then
tables, then indexes. HNSW indexes go in a **later** migration than the table
DDL so a large initial backfill is not slowed by index maintenance per insert.

---

## Benchmark harness

**Principle: adapters produce canonical `Trace` objects and go into the same
tables** with `source='benchmark:trail'`. The benchmark exercises the production
pipeline end to end rather than a parallel code path. That is the entire value
and it is easy to accidentally get wrong.

**TRAIL** ships OpenTelemetry-format traces, so the adapter feeds them straight
through `otlp.py` with no special handling. Ground truth is per-error
`(span_id, category)`, mapped through an **explicit dict constant** so the
mapping is reviewable, versioned, and diffable: it encodes a genuine judgment
about how another team's taxonomy corresponds to ours, and that judgment will be
revised. Multiple annotated errors become multiple `benchmark_cases` rows on one
trace; scoring credits a match against **any** annotated error while separately
reporting whether it found the **earliest**. Conflating those flatters the
system.

**Who&When** has no OTel form (message lists with `(agent, step, reason)` ground
truth). The adapter synthesizes one AGENT span per message under a synthetic
orchestrator root, refining kind from content markers. Because it is already
linear, `mistake_step` maps directly to `ground_truth_step_index`, making it the
cleaner primary benchmark for step-level metrics.

**Metrics** (`bench_score.py`): exact step accuracy; tolerance accuracy @{0,1,3}
reported **all three, never only the best**; span-level accuracy for external
comparability with TRAIL; failure-class accuracy plus the full 21×21 confusion
matrix; joint accuracy (step AND class, the headline); **earliness error**, the
signed mean of `predicted - ground_truth`, where positive bias means the system
is blaming symptoms downstream of the cause, precisely the failure mode this
product exists to eliminate; abstention rate and accuracy-conditional-on-
committing; calibration (Brier, ECE, reliability points); **candidate recall@k**
for k in {1,3,5}.

Candidate recall@k is the most important metric in the harness because it
cleanly separates "L1/L2 narrowing missed it" from "L3 picked wrong from a
correct shortlist". Those have completely different fixes and are
indistinguishable from end-to-end accuracy alone.

Plus **layer ablation** (same suite with L2 disabled, with L1 disabled) to prove
the contrastive layer earns its complexity, and cost/latency per diagnosis
against a long-context single-call baseline.

---

## Parallel execution plan

### Phase 0: Foundation (sequential, one agent, blocks everything)

Its job is to make every workstream's first action be writing implementation
code, never negotiating an interface. Nine tasks, each ending in a commit.

---

#### Task 1: Repo scaffold and dependency manifest

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `LICENSE`,
  `README.md`, `src/culprit/__init__.py`, `src/culprit/py.typed`,
  `tests/conftest.py`

**Interfaces:**
- Produces: an importable `culprit` package and the frozen dependency set every
  later task and workstream relies on.

- [ ] **Step 1: Write `pyproject.toml` with the complete dependency set**

```toml
[project]
name = "culprit"
version = "0.1.0"
description = "Finds the step that actually broke the run"
requires-python = ">=3.12"
dependencies = [
    "alembic>=1.14.0",
    "fastapi>=0.139.2",
    "fastembed>=0.5.0",
    "litellm<1.90; sys_platform == 'win32'",
    "litellm; sys_platform != 'win32'",
    "numpy>=2.5.1",
    "opentelemetry-proto>=1.29.0",
    "pgvector>=0.3.6",
    "psycopg[binary,pool]>=3.2.3",
    "pydantic>=2.13.4",
    "python-dotenv>=1.2.2",
    "redis>=5.2.0",
    "rq>=2.0.0",
    "scikit-learn>=1.6.0",
    "typer>=0.27.0",
    "uvicorn>=0.51.0",
]

[project.scripts]
culprit = "culprit.cli:app"

[build-system]
requires = ["uv_build>=0.11.25,<0.12.0"]
build-backend = "uv_build"

[dependency-groups]
dev = [
    "fakeredis>=2.26.0",
    "httpx>=0.28.1",
    "pytest>=9.1.1",
    "pytest-cov>=7.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

# culprit's own configuration lives in culprit.toml, not here.
```

The `litellm` platform marker is copied deliberately from Litmus, same machine
and same linker problem. Do not "simplify" it to a single unpinned entry.

- [ ] **Step 2: Write the minimal `conftest.py`**

```python
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_log_file(monkeypatch, tmp_path):
    """Every test gets its own log file under tmp_path so a rotating file
    handler in one test cannot hold a Windows file handle open for another."""
    monkeypatch.setenv("CULPRIT_LOG_FILE", str(tmp_path / "culprit-test.jsonl"))


requires_db = pytest.mark.skipif(
    not os.environ.get("CULPRIT_TEST_DSN"),
    reason="needs CULPRIT_TEST_DSN; default test run stays fully offline",
)
```

- [ ] **Step 3: Verify the package installs and imports**

Run: `uv sync --all-groups && uv run python -c "import culprit; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Verify the test suite runs green on an empty suite**

Run: `uv run pytest -v`
Expected: PASS, 0 tests collected, no errors

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore LICENSE README.md src tests
git commit -m "chore: scaffold culprit package and freeze dependency set"
```

---

#### Task 2: Canonical domain model

**Files:**
- Create: `src/culprit/schemas.py`
- Test: `tests/test_schemas.py`

**Interfaces:**
- Produces: `Trace`, `Span`, `Step`, `SpanKind`, `SpanStatus`, `Outcome`,
  `Message`, `ToolCallRequest`, `RetrievedDoc`, `LlmPayload`, `ToolPayload`,
  `RetrievalPayload`, `AgentPayload`. **Every workstream imports from here.**

- [ ] **Step 1: Write the failing test**

```python
from culprit.schemas import Span, SpanKind, SpanStatus, ToolPayload


def test_span_with_an_unrecognized_vocabulary_keeps_raw_attributes_and_records_the_error():
    """Normalization must be reversible: a span we cannot interpret today has
    to survive intact so a later vocabulary module can backfill it."""
    span = Span(
        trace_id="t1", span_id="s1", parent_span_id=None, name="mystery",
        kind=SpanKind.UNKNOWN, status=SpanStatus.UNSET, status_message=None,
        start_ns=0, end_ns=1, vocabulary="unknown",
        attributes={"vendor.weird.key": 42}, payload=None,
        normalize_error="no vocabulary matched",
    )

    assert span.attributes == {"vendor.weird.key": 42}
    assert span.payload is None
    assert span.normalize_error == "no vocabulary matched"


def test_tool_payload_records_result_length_separately_from_result_text():
    """result_len is stored rather than derived because empty_tool_result must
    still fire after attributes are pruned by the retention policy."""
    payload = ToolPayload(
        tool_name="search_orders", call_id="c1", arguments_json="{}",
        arguments={}, result_text="", result_len=0, is_error=False,
        error_message=None,
    )

    assert payload.result_len == 0
    assert payload.is_error is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'culprit.schemas'`

- [ ] **Step 3: Write `schemas.py`**

Implement the full model as specified in "Canonical domain model" above. Shape
of the core types:

```python
class SpanKind(StrEnum):
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    AGENT = "agent"
    CHAIN = "chain"
    EMBEDDING = "embedding"
    GUARDRAIL = "guardrail"
    UNKNOWN = "unknown"


class Span(BaseModel):
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: SpanKind
    status: SpanStatus
    status_message: str | None
    start_ns: int
    end_ns: int
    vocabulary: str
    # Raw attributes are never discarded: normalization stays reversible until
    # the retention policy prunes them explicitly.
    attributes: dict
    payload: LlmPayload | ToolPayload | RetrievalPayload | AgentPayload | None
    normalize_error: str | None = None
```

`payload` is a plain union, not `Field(discriminator=...)`, because the
discriminant lives on the sibling `kind` field.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schemas.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add src/culprit/schemas.py tests/test_schemas.py
git commit -m "feat: add canonical trace, span, and step domain model"
```

---

#### Task 3: Analysis contracts and failure taxonomy

**Files:**
- Create: `src/culprit/signals.py`, `src/culprit/taxonomy.py`
- Test: `tests/test_signals.py`, `tests/test_taxonomy.py`

**Interfaces:**
- Consumes: `culprit.schemas`
- Produces: `Evidence`, `Signal`, `DivergenceCandidate`, `ContrastResult`,
  `Candidate`, `Adjudication`, `Diagnosis`, `NeighborFn`, and
  `FailureClass` with `describe_all() -> str`.

- [ ] **Step 1: Write the failing tests**

```python
from culprit.signals import Signal
from culprit.taxonomy import FailureClass, describe_all


def test_a_detector_that_raised_produces_a_sentinel_signal_excluded_from_ranking():
    """One buggy detector must never cost the whole diagnosis. The sentinel
    shape is what run_detectors emits instead of propagating the exception."""
    sentinel = Signal(
        detector="oscillation", step_index=-1, span_id=None, severity=0.0,
        category="unknown", message="detector failed", evidence=[],
        error="ValueError: bad window",
    )

    assert sentinel.error is not None
    assert sentinel.severity == 0.0
    assert sentinel.step_index == -1


def test_describe_all_emits_every_failure_class_so_prompt_and_code_cannot_drift():
    """The L3 prompt is built by formatting this block in. If a class were
    added to the enum but not the description map, the model would be asked to
    choose from a set the code does not accept."""
    block = describe_all()

    for member in FailureClass:
        assert member.value in block
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_signals.py tests/test_taxonomy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write both modules**

`signals.py` implements the dataclasses exactly as specified in "The Signal /
Diagnosis contract" above, plus the `NeighborFn` alias. `taxonomy.py`:

```python
class FailureClass(StrEnum):
    TASK_MISINTERPRETATION = "task_misinterpretation"
    PLAN_OMISSION = "plan_omission"
    CONSTRAINT_VIOLATION = "constraint_violation"
    WRONG_TOOL_SELECTED = "wrong_tool_selected"
    MALFORMED_TOOL_INPUT = "malformed_tool_input"
    TOOL_FAILURE_UNHANDLED = "tool_failure_unhandled"
    SILENT_EMPTY_RESULT_MISREAD = "silent_empty_result_misread"
    HALLUCINATED_TOOL_OR_PARAMETER = "hallucinated_tool_or_parameter"
    RETRIEVAL_MISS = "retrieval_miss"
    CONTEXT_LOSS = "context_loss"
    INFORMATION_FABRICATION = "information_fabrication"
    INFINITE_LOOP_OR_OSCILLATION = "infinite_loop_or_oscillation"
    PREMATURE_TERMINATION = "premature_termination"
    STEP_BUDGET_EXHAUSTED = "step_budget_exhausted"
    HANDOFF_INFORMATION_LOSS = "handoff_information_loss"
    ROLE_VIOLATION = "role_violation"
    CONFLICTING_SUBRESULTS_UNRECONCILED = "conflicting_subresults_unreconciled"
    MISSING_VERIFICATION = "missing_verification"
    INCORRECT_VERIFICATION = "incorrect_verification"
    OUTPUT_SCHEMA_VIOLATION = "output_schema_violation"
    UNKNOWN = "unknown"


_DESCRIPTIONS: dict[FailureClass, str] = {
    FailureClass.SILENT_EMPTY_RESULT_MISREAD: (
        "A tool returned successfully but with no usable content, and the "
        "agent treated the empty result as a positive finding."
    ),
    # ... one entry per member, no exceptions
}


def describe_all() -> str:
    """Rendered into the L3 prompt. Built from the enum rather than written as
    prose so a new class cannot silently go unlisted."""
    return "\n".join(f"- {m.value}: {_DESCRIPTIONS[m]}" for m in FailureClass)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_signals.py tests/test_taxonomy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/culprit/signals.py src/culprit/taxonomy.py tests/test_signals.py tests/test_taxonomy.py
git commit -m "feat: add analysis contracts and 21-class failure taxonomy"
```

---

#### Task 4: Config and logging

**Files:**
- Create: `src/culprit/config.py`, `src/culprit/logging_config.py`,
  `culprit.toml`
- Test: `tests/test_config.py`, `tests/test_logging_config.py`

**Interfaces:**
- Produces: `CulpritConfig` (frozen dataclass), `load_config()`, `ConfigError`,
  `LOGGER_NAME`, `JsonFormatter`, `configure_logging()`.

`culprit.toml` ships **fully commented out**, documenting every key with its
default, and must already declare every field any workstream will need, because
it freezes after this phase.

- [ ] **Step 1: Write the failing test**

```python
from culprit.config import ConfigError, load_config

import pytest


def test_load_config_overrides_only_the_keys_present_in_the_file():
    ...  # written against tmp_path, mirrors Litmus/tests/test_config.py


def test_config_error_names_culprit_toml_not_a_cli_flag(tmp_path):
    """Typer's own min=/max= validation would blame a flag the user never
    passed. ConfigError exists so the message points at the real source."""
    (tmp_path / "culprit.toml").write_text("min_confidence = 1.7\n")

    with pytest.raises(ConfigError, match="culprit.toml"):
        load_config(start_dir=tmp_path)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement both modules**

Copy the structure of `Litmus/src/litmus/config.py` exactly: module docstring
stating precedence and the layering rule, `_CONFIG_FILENAME`, `ConfigError`,
`@dataclass(frozen=True) class CulpritConfig`, module-level validation tuples,
`_validate()`, `load_config(start_dir)`. Fields must cover at minimum:
`database_url`, `redis_url`, `model`, `max_candidates`, `min_confidence`,
`ambiguity_margin`, `max_workers`, `min_reference_runs`, `max_reference_runs`,
`reference_cosine_max`, `context_token_budget`, `plateau_eps`,
`embedding_model`, `min_cluster_size`, `retention_days`, `log_file`,
`log_level`.

`logging_config.py` copies `Litmus/src/litmus/logging_config.py` with
`LOGGER_NAME = "culprit"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py tests/test_logging_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/culprit/config.py src/culprit/logging_config.py culprit.toml tests/test_config.py tests/test_logging_config.py
git commit -m "feat: add config loading and JSON-lines logging"
```

---

#### Task 5: External seams (LLM, embeddings, database handle)

**Files:**
- Create: `src/culprit/llm.py`, `src/culprit/embed.py`, `src/culprit/db.py`
- Test: `tests/test_llm.py`, `tests/test_embed.py`

**Interfaces:**
- Produces: `litellm_call` + `CallFn`, `embed_texts` + `EmbedFn`,
  `make_pool` + `ConnFn`.

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import MagicMock

from culprit.llm import litellm_call


def _fake_response(content: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_litellm_call_returns_output_latency_and_cost(monkeypatch):
    """The dotted monkeypatch target is why llm.py must use `import litellm`
    and call `litellm.completion(...)` rather than importing the function."""
    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: _fake_response("hello")
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    output, latency_ms, cost_usd = litellm_call("gemini/gemini-2.5-flash-lite", "hi")

    assert output == "hello"
    assert cost_usd == 0.0001
    assert latency_ms >= 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the three seams**

```python
# llm.py - deliberately thin. No retries, no logging: all log calls sit at the
# orchestration layer where context is already assembled.
import time

import litellm


def litellm_call(model: str, prompt: str) -> tuple[str, float, float]:
    start = time.perf_counter()
    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content
    cost_usd = litellm.completion_cost(completion_response=response)
    return output, latency_ms, cost_usd
```

`embed.py` lazily loads one `fastembed.TextEmbedding` instance (the model is
~130MB resident and is shared by L2 reference lookup and L5 clustering).
`db.py` builds a psycopg 3 `ConnectionPool` and calls
`pgvector.psycopg.register_vector` on connect.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm.py tests/test_embed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/culprit/llm.py src/culprit/embed.py src/culprit/db.py tests/test_llm.py tests/test_embed.py
git commit -m "feat: add llm, embedding, and database seams"
```

---

#### Task 6: Database schema (Alembic revision 0001)

**Files:**
- Create: `alembic.ini`, `migrations/env.py`, `migrations/versions/0001_initial.py`
- Test: `tests/test_migrations.py`

**Interfaces:**
- Produces: the complete schema. **Frozen for all of Phase 1**; new DDL becomes
  revision 0002 during Integration.

- [ ] **Step 1: Write the failing test**

```python
from tests.conftest import requires_db


@requires_db
def test_migration_creates_pgvector_columns_at_384_dimensions(db_conn):
    """384 is bge-small-en-v1.5's dimension. Both vector columns must match it
    and each other, because one loaded model serves both."""
    rows = db_conn.execute(
        "SELECT table_name, atttypmod FROM information_schema.columns "
        "JOIN pg_attribute ON attname = column_name "
        "WHERE column_name IN ('task_embedding', 'card_embedding')"
    ).fetchall()

    assert len(rows) == 2
    assert {r[1] for r in rows} == {384}
```

- [ ] **Step 2: Run to verify it fails**

Run: `CULPRIT_TEST_DSN=... uv run pytest tests/test_migrations.py -v`
Expected: FAIL, relation does not exist

- [ ] **Step 3: Write revision 0001**

Hand-written DDL, no `--autogenerate`. Order inside the revision matters:
`CREATE EXTENSION IF NOT EXISTS vector` first, then every table, then btree
indexes. **HNSW indexes go in revision 0002** so a large initial backfill is not
slowed by index maintenance on every insert. Include
`spans.attributes_pruned boolean NOT NULL DEFAULT false` here, per the retention
decision. Tables and columns exactly as specified in "Postgres schema" above.

- [ ] **Step 4: Run to verify it passes**

Run: `CULPRIT_TEST_DSN=... uv run alembic upgrade head && CULPRIT_TEST_DSN=... uv run pytest tests/test_migrations.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add alembic.ini migrations tests/test_migrations.py
git commit -m "feat: add initial schema with pgvector columns"
```

---

#### Task 7: Synthetic trace generator

**Files:**
- Create: `src/culprit/synth.py`
- Test: `tests/test_synth.py`

**This is the single most important task in Phase 0.** Four of the eight
workstreams get all their test data from it and would otherwise block on WS-A.

**Interfaces:**
- Consumes: `culprit.schemas`
- Produces:
  - `successful_run(seed: int) -> Trace` - an "order refund agent" run with
    *natural* variation: reordered independent steps, differing retrieval doc
    counts, length varying by +/-2 steps. Without real variation the L2
    reference model learns a single rigid path and every honest difference
    looks like a divergence.
  - `inject(trace: Trace, kind: str, at_step: int) -> tuple[Trace, int]` -
    returns the mutated trace and **the ground-truth injection index**. `kind`
    must cover the trigger condition of every L1 detector.
  - `INJECTION_KINDS: frozenset[str]`

- [ ] **Step 1: Write the failing test**

```python
from culprit.synth import INJECTION_KINDS, inject, successful_run


def test_successful_runs_vary_in_length_and_order_across_seeds():
    """A reference pool of identical traces teaches the profile nothing, and
    every legitimate variation would then score as a divergence."""
    runs = [successful_run(seed=i) for i in range(20)]

    lengths = {r.step_count for r in runs}
    assert len(lengths) > 1


def test_inject_returns_the_ground_truth_index_it_mutated():
    """WS-C and WS-D both assert against this index. If inject did not return
    it, every downstream test would hardcode a magic number."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="empty_tool_result", at_step=3)

    assert index == 3
    assert mutated.trace_id != base.trace_id


def test_every_injection_kind_is_reachable():
    """Guards the WS-C definition of done, which requires one injection per
    detector."""
    base = successful_run(seed=0)

    for kind in INJECTION_KINDS:
        mutated, index = inject(base, kind=kind, at_step=2)
        assert index >= 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_synth.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `synth.py`**

Seeded `random.Random(seed)` so runs are reproducible. `INJECTION_KINDS` must
contain one entry per detector named in the L1 catalogue.

> Convention note to record in `CLAUDE.md` so no agent re-litigates it:
> `synth.py` lives in `src/` rather than `tests/` because it is a **product
> module**. It powers a `culprit demo` command and supplies the benchmark's
> negative controls. Test files still write their own private `_helper()`
> wrappers around it, so the "no centralized fixtures" convention holds.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_synth.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add src/culprit/synth.py tests/test_synth.py
git commit -m "feat: add synthetic trace generator with fault injection"
```

---

#### Task 8: Synthetic result factories and layer stubs

**Files:**
- Create: `src/culprit/synth_results.py`, `src/culprit/pipeline.py`,
  `src/culprit/run_detectors.py`, `src/culprit/contrast.py`,
  `src/culprit/adjudicate.py`, `src/culprit/cluster.py`
- Test: `tests/test_synth_results.py`

**Interfaces:**
- Produces: `make_signal()`, `make_divergence()`, `make_diagnosis()`, plus every
  layer entrypoint at its **real signature** with a `NotImplementedError` body.
  This is what lets WS-G build the API, CLI, and jobs against real signatures on
  day one.

```python
# pipeline.py
def diagnose(trace: Trace, *, conn_fn: ConnFn, call_fn: CallFn,
             embed_fn: EmbedFn) -> Diagnosis:
    raise NotImplementedError("filled during Integration, task I1")

# run_detectors.py
def run_detectors(trace: Trace, steps: list[Step],
                  spans_by_id: dict[str, Span]) -> list[Signal]:
    raise NotImplementedError("owned by WS-C")

# contrast.py
def contrast(trace: Trace, steps: list[Step], *, neighbor_fn: NeighborFn,
             conn_fn: ConnFn) -> ContrastResult:
    raise NotImplementedError("owned by WS-D")

# adjudicate.py
def adjudicate(candidates: list[Candidate], trace: Trace, steps: list[Step],
               *, call_fn: CallFn, model: str) -> list[Adjudication]:
    raise NotImplementedError("owned by WS-E")

# cluster.py
def cluster_diagnoses(diagnoses: list[Diagnosis], *,
                      embed_fn: EmbedFn) -> dict[str, int]:
    raise NotImplementedError("owned by WS-F")
```

- [ ] **Step 1: Write the failing test**

```python
import pytest

from culprit.contrast import contrast
from culprit.synth_results import make_diagnosis, make_signal


def test_factories_produce_valid_contract_objects_without_running_any_layer():
    """WS-E and WS-F build entirely against these, so they never wait on WS-C
    or WS-D to exist."""
    signal = make_signal(detector="empty_tool_result", step_index=3)
    diagnosis = make_diagnosis(root_cause_step_index=3)

    assert signal.step_index == 3
    assert diagnosis.root_cause_step_index == 3


def test_layer_stubs_raise_not_implemented_with_their_owner_named():
    """The stub message tells whoever hits it which workstream owns the gap."""
    with pytest.raises(NotImplementedError, match="WS-D"):
        contrast(None, [], neighbor_fn=lambda v, k: [], conn_fn=lambda: None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_synth_results.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the factories and stubs**

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_synth_results.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Commit**

```bash
git add src/culprit/synth_results.py src/culprit/pipeline.py src/culprit/run_detectors.py src/culprit/contrast.py src/culprit/adjudicate.py src/culprit/cluster.py tests/test_synth_results.py
git commit -m "feat: add result factories and layer entrypoint stubs"
```

---

#### Task 9: OTLP fixtures and CLAUDE.md seed

**Files:**
- Create: `tests/fixtures/otlp/otel_genai_sample.json`,
  `tests/fixtures/otlp/openinference_sample.json`,
  `tests/fixtures/otlp/raw_upload_sample.json`, `CLAUDE.md`,
  `AI_docs/PHASES.md`

**Interfaces:**
- Produces: the fixtures WS-A's definition of done is written against. Both
  vocabulary samples must encode **the same logical agent run** so WS-A can
  assert they normalize to identical `Step` sequences.

- [ ] **Step 1: Hand-write the two vocabulary fixtures**

Same logical run in both: an agent that calls `search_orders`, retrieves two
documents, then answers. OTel GenAI uses `gen_ai.*` attributes with
`invoke_agent` / `chat` / `execute_tool` span names; OpenInference uses
`openinference.span.kind` with `llm.*`, `tool.*`, `retrieval.*`.

- [ ] **Step 2: Verify both parse as JSON**

Run: `uv run python -c "import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path('tests/fixtures/otlp').glob('*.json')]; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Seed `CLAUDE.md` with every decision from this plan**

Use the Litmus format: `## Brand`, `## Architecture decisions (with rationale)`
with `Rejected: alternative - why` sub-bullets, `## Known gotcha:` sections, and
`## Guardrails - don't do these without asking first`. Include the applied-ml
citations next to the decisions they support.

- [ ] **Step 4: Verify the full suite still passes**

Run: `uv run pytest -v`
Expected: PASS, all tests from tasks 1 through 8

- [ ] **Step 5: Commit and tag the foundation**

```bash
git add tests/fixtures CLAUDE.md AI_docs
git commit -m "feat: add OTLP fixtures and seed decision log"
git tag phase-0-complete
```

**Gate before Phase 1 dispatch:** `uv run pytest -v` green offline, and
`git tag phase-0-complete` exists. Do not dispatch parallel workstreams before
both hold, because every contract they compile against lives behind that tag.

### Phase 1: Eight parallel workstreams, strictly disjoint file ownership

| WS | Deliverable | Owns (src) |
|---|---|---|
| **A** | L0 ingestion + normalization | `otlp.py`, `normalize.py`, `linearize.py`, `vocab/*` |
| **B** | Persistence | `store_traces.py`, `store_diagnoses.py` |
| **C** | L1 detectors | `detectors/*`, `run_detectors.py` |
| **D** | L2 contrastive | `fingerprint.py`, `signature.py`, `align.py`, `reference.py`, `contrast.py` |
| **E** | L3 adjudication | `candidates.py`, `context_window.py`, `prompts.py`, `adjudicate.py`, `confidence.py` |
| **F** | L5 clustering | `cluster_embed.py`, `cluster.py`, `cluster_label.py` |
| **G** | Service surface | `api.py`, `views.py`, `cli.py`, `queue.py`, `jobs.py`, `worker.py` |
| **H** | Benchmark harness | `benchmarks/*`, `bench_score.py` |

Test file names are 1:1 with module names, so `src/` ownership implies `tests/`
ownership.

> **Scope decision, per the writing-plans scope check.** These eight are
> genuinely independent subsystems, so each gets **its own plan document**
> written at dispatch time, in this same bite-sized TDD format, rather than
> being expanded inline here. A single document containing every step for 51
> modules would be unreviewable and stale before execution. **This document is
> the spec plus the Phase 0 plan.** What follows is the interface contract each
> workstream's plan must be written against, which is the part that has to be
> fixed up front so eight agents cannot drift.

### Workstream interface contracts

Every workstream consumes `culprit.schemas`, `culprit.signals`,
`culprit.taxonomy`, `culprit.config`, and `culprit.logging_config`. Listed below
is only what is additional.

**WS-A, ingestion.**
- Consumes: `tests/fixtures/otlp/*`, `culprit.synth` for round-trip checks.
- Produces:
  - `otlp.decode(payload: bytes, content_type: str) -> list[dict]`
  - `normalize.normalize_span(raw: dict, trace_id: str) -> Span`
  - `linearize.linearize(spans: list[Span]) -> list[Step]`
  - `vocab.registry.detect(raw: dict) -> str`

**WS-B, persistence.**
- Consumes: `culprit.db.ConnFn`, `culprit.synth`.
- Produces:
  - `store_traces.write_trace(conn_fn, trace, spans, steps, embedding) -> None`
  - `store_traces.read_trace(conn_fn, trace_id) -> tuple[Trace, list[Span], list[Step]]`
  - `store_traces.nearest_successful(conn_fn, embedding, k) -> list[str]`
    (**this is the concrete `NeighborFn`; WS-D never imports this module**)
  - `store_traces.prune_attributes(conn_fn, older_than_days) -> int`
  - `store_diagnoses.write_diagnosis(conn_fn, diagnosis) -> None`
  - `store_diagnoses.read_diagnoses(conn_fn, trace_id) -> list[Diagnosis]`

**WS-C, detectors.**
- Consumes: `culprit.synth.inject`.
- Produces: `run_detectors(trace, steps, spans_by_id) -> list[Signal]`,
  `detectors.base.Detector` Protocol, `detectors.base.DetectorContext`,
  `detectors.registry.DETECTORS: dict[str, Detector]`.

**WS-D, contrastive.**
- Consumes: `culprit.synth`, an injected `NeighborFn` (**tests pass a
  dict-backed fake; do not import `store_traces`**).
- Produces: `contrast(trace, steps, *, neighbor_fn, conn_fn) -> ContrastResult`,
  `fingerprint.agent_key(trace) -> str`, `fingerprint.task_key(goal) -> str`,
  `signature.signature_of(step) -> str`, `signature.sim(a, b) -> float`,
  `align.gotoh(a, b, ...) -> Alignment`, `reference.build_profile(runs) -> Profile`.

**WS-E, adjudication.**
- Consumes: `culprit.synth_results`, `culprit.llm.CallFn`,
  `taxonomy.describe_all()`.
- Produces: `candidates.merge(signals, contrast_result) -> list[Candidate]`,
  `context_window.build(candidate, trace, steps, budget) -> str`,
  `adjudicate(candidates, trace, steps, *, call_fn, model) -> list[Adjudication]`,
  `confidence.calibrate(adjudication, candidate, packet) -> float`.

**WS-F, clustering.**
- Consumes: `culprit.synth_results`, `culprit.embed.EmbedFn`,
  `culprit.llm.CallFn`.
- Produces: `cluster_embed.render_card(diagnosis) -> str`,
  `cluster_diagnoses(diagnoses, *, embed_fn) -> dict[str, int]`,
  `cluster_label.label(cards, histogram, *, call_fn) -> ClusterLabel`.

**WS-G, service surface.**
- Consumes: every layer stub from Task 8 (**monkeypatched in tests; never
  calls a real layer**).
- Produces: the FastAPI `app`, the Typer `app`, `queue.enqueue_diagnosis(id)`,
  `jobs.diagnose_trace_job(id)`, `jobs.recluster_job()`, `worker` entrypoint.

**WS-H, benchmarks.**
- Consumes: `culprit.schemas`, `otlp.decode` for TRAIL (**the one cross-
  workstream import; if WS-A is not merged yet, work against the checked-in
  fixtures and wire it in Integration**).
- Produces: `benchmarks.base.BenchmarkAdapter` Protocol, `BenchmarkCase`,
  `bench_score.score(predictions, truth) -> BenchReport`.

**Definitions of done:**

- **A.** Protobuf and JSON payloads of the same logical trace produce identical
  canonical `Trace` objects. Both vocabularies normalize the same run to
  identical `Step` sequences. An unrecognized span sets `normalize_error` rather
  than raising, raw attributes preserved. **Linearization provably deterministic
  under sibling timestamp ties (dedicated test).**
- **B.** Round-trip every entity. **5,000-span bulk insert completes in under
  5 seconds** via `executemany`, asserted with `time.perf_counter()` (a naive
  per-row insert takes roughly 30x that, so this catches the regression that
  matters). `vector(384)` round-trips with `pytest.approx(abs=1e-6)`. The kNN
  query returns successes ordered by cosine distance and never returns a
  failure. `prune_attributes` empties `attributes` and sets `attributes_pruned`
  while leaving `payload`, `steps`, and `diagnoses` untouched. DSN-gated.
- **C.** Every detector fires on its `inject(...)` case **and does not fire on
  20 clean synthetic successes.** The false-positive floor is tested as
  rigorously as the true positive, because a noisy detector is worse than a
  missing one. A raising detector produces a sentinel while the other 19 return.
- **D.** Alignment reproduces a **hand-computed 5×5 Needleman-Wunsch example
  exactly, including the traceback path.** Non-negotiable: a subtle traceback
  bug is invisible in aggregate metrics and silently degrades everything
  downstream. Symmetric scores. Identical sequences yield zero candidates. On
  `synth` traces with known injection index, top-1 divergence equals the
  injection index for >=80% of injection kinds. Abstains cleanly below 3
  references.
- **E.** Deterministic merge/rank/truncate with earlier-step-wins ties. Context
  packet respects the 12k cap and drops spine before zoom. De-fencing handles
  all four fence variants (**reuse Litmus's four proven cases verbatim**). An
  unparseable response yields a sentinel abstained `Adjudication`. Each
  abstention gate has its own test. Out-of-window citations measurably lower
  calibrated confidence.
- **F.** Two diagnoses differing only by order ID render byte-identical cards.
  HDBSCAN recovers 3 well-separated synthetic groups plus noise. Labeling makes
  **exactly** `len(clusters)` LLM calls, asserted by call count. Re-label policy
  skips stable clusters, asserted on a second pass.
- **G.** Builds entirely against F13's stubs with layers monkeypatched. Every
  route maps domain exceptions to `HTTPException(...) from e` preceded by an
  `api_error` warning. OTLP endpoint accepts both `application/x-protobuf` and
  `application/json`. Enqueue returns a job id; `GET /jobs/{id}` reports status.
  Queue tests use `fakeredis`. **Zero `async def` anywhere.**
- **H.** A checked-in 3-record sample of each benchmark converts to canonical
  traces with ground truth attached. **Every metric is verified against this
  fixed hand-worked example**, written into the test as literals: predictions
  `[3, 7, 2]` against truth `[3, 8, 2]` gives exact accuracy `2/3 = 0.667`,
  tolerance@1 `3/3 = 1.0`, and earliness error
  `((3-3) + (7-8) + (2-2)) / 3 = -0.333`. The negative earliness value is the
  point of the example: it proves the sign convention is right, where positive
  means blaming symptoms downstream of the cause. Unknown ground-truth
  categories map to `unknown` with a logged warning and never crash.

### Dependency graph

```
              Phase 0 Foundation (sequential)
                          |
  +-----+-----+-----+-----+-----+-----+-----+
  |     |     |     |     |     |     |     |
 WS-A  WS-B  WS-C  WS-D  WS-E  WS-F  WS-G  WS-H
  |     |     |     |     |     |     |     |
  +-----+-----+-----+-----+-----+-----+-----+
                          |
              Phase 2 Integration (sequential)
```

**All eight are mutually non-blocking.** That is a design outcome, not luck, and
rests on three decisions: `synth.py`/`synth_results.py` mean C, D, E, F never
need A's ingestion for test data; **WS-D takes its pgvector neighbor lookup as
an injected `NeighborFn = Callable[[list[float], int], list[str]]`** rather than
importing `store_traces`, so it does not depend on B (tests pass a dict-backed
fake); F13's stubs mean G depends on no analysis layer.

### Conflict-avoidance rules

1. **`pyproject.toml` frozen after Foundation.** Worst merge-conflict magnet in
   a parallel Python build. Foundation declares the complete list precisely so
   this costs nothing. A workstream needing an undeclared dep stops and asks.
2. **`config.py` and `culprit.toml` frozen after Foundation**, same reasoning.
3. **`schemas.py`, `signals.py`, `taxonomy.py` frozen after Foundation.** A
   workstream finding a genuinely missing field records it as an integration
   item and works around it locally. Eight agents serializing on one contract
   file destroys the entire parallelism gain.
4. **Exactly one Alembic revision during Phase 1.** Parallel revisions create
   `down_revision` linear-history conflicts that are painful to untangle. New
   DDL becomes revision 0002 in integration.
5. **Nobody edits `pipeline.py`.** Foundation-stubbed, Integration-owned.
6. **One owner per registry:** `vocab/registry.py` → A, `detectors/registry.py`
   → C, `benchmarks/registry.py` → H. No cross-registration.
7. **No shared test files.** `conftest.py` frozen after Foundation, which the
   conventions want anyway.
8. **`tests/fixtures/` partitioned by subdirectory:** `fixtures/otlp/` is A's,
   `fixtures/benchmarks/` is H's.
9. **`CLAUDE.md`, `README.md`, `AI_docs/PHASES.md` are Integration-only.**
   Litmus's `CLAUDE.md` is 46 KB; eight agents appending decision entries would
   conflict on nearly every merge. Each workstream instead emits a
   `## WS-x decisions` block in its PR description in the established format
   (bolded decision, rationale, `Rejected: alternative - why`), and Integration
   merges all eight in one pass.
10. **Duplicate small helpers rather than sharing them.** Two workstreams both
    needing text truncation each write their own private `_truncate()`. The
    conventions already prefer duplicated `_helper()` functions, and here that
    removes a whole class of cross-workstream contention.

### Phase 2: Integration (sequential)

| ID | Deliverable |
|---|---|
| I1 | Fill `pipeline.py`: wire L0 → L1 → L2 → L3 → L5 with injected callables. **Per-layer isolation:** a failing layer degrades the diagnosis and records itself in `degraded_layers`, never crashes the run |
| I2 | Replace WS-G's monkeypatched stubs with real wiring; `jobs.py` calls the real pipeline |
| I3 | Alembic revision 0002 for fields discovered during Phase 1 |
| I4 | End-to-end test, DSN-gated: OTLP fixture → ingest → diagnose → persist → API read → cluster |
| I5 | Run the harness against TRAIL and Who&When samples; **record the real numbers in `CLAUDE.md`, including the bad ones** |
| I6 | Consolidate the eight decision blocks into `CLAUDE.md`; write the `AI_docs/PHASES.md` status table and resume point |
| I7 | Fit calibration coefficients by logistic regression if labels suffice; **if not, record explicitly that they remain uncalibrated priors** |

`recluster_job()` is deliberately off the per-trace path: L5 is a scheduled
batch over accumulated diagnoses, so `jobs.py` exposes `diagnose_trace_job(id)`
and `recluster_job()`.

---

## Where the deferred layers slot in

Neither requires touching a v1 contract, which was a design constraint rather
than a happy accident.

- **L4 counterfactual replay** (`replay.py`, `replay_harness.py`) consumes a
  `Candidate`, re-executes from that step with a modified action, and merges
  evidence into `Candidate.prior` through the same path L1 uses. This is why
  `Candidate.source` is a `str` and not an enum.
- **L6 regression case generation** (`regress_gen.py`) consumes a persisted
  `Diagnosis` and emits a Litmus-format test case. This is why
  `Diagnosis.layer_versions` is a `dict[str, str]` rather than fixed columns.

---

## Cost to run

Prices are approximate and should be re-checked before committing; verify
current provider pricing rather than trusting these figures.

### LLM cost per diagnosis

This is where the architecture pays for itself. Each diagnosis is 5 adjudication
calls at roughly 8k input tokens and 500 output tokens each, so about **40k
input and 2.5k output per trace**. Cluster labelling is one call per cluster,
amortised across hundreds of diagnoses and effectively free. Embeddings run
locally on CPU and cost nothing.

| Model tier | Per diagnosis | 1,000 diagnoses/mo |
| :--- | ---: | ---: |
| Gemini Flash Lite class | ~$0.005 | ~$5 |
| Gemini Flash class | ~$0.02 | ~$20 |
| Frontier (Sonnet / GPT class) | ~$0.16 | ~$160 |

**The naive long-context baseline, for comparison:** one call carrying a
200k-token trace costs roughly 4x more per diagnosis at every tier, and performs
worse. Narrow-then-adjudicate is both the accuracy argument and the cost
argument, and `adjudications.cost_usd` exists specifically so this can be proven
with measured numbers rather than asserted.

### Infrastructure

| Component | Option | Cost |
| :--- | :--- | ---: |
| Postgres + pgvector | Neon / Supabase free tier | $0 |
| | Render Postgres starter | ~$7/mo |
| Redis | Upstash free tier (10k cmd/day) | $0 |
| API | Cloud Run, scales to zero | ~$0 at low traffic |
| **Worker** | Fly.io shared-cpu-1x, 512MB | ~$3-5/mo |
| | Render background worker | ~$7/mo |

**The worker is the one unavoidable always-on cost.** RQ polls Redis, so it
cannot scale to zero the way the API can, and it needs at least 512MB of RAM
because fastembed's model is ~130MB resident alongside scikit-learn. That single
constraint sets the floor.

### Realistic totals

| Scenario | Monthly |
| :--- | ---: |
| Portfolio demo only, handful of traces, all free tiers | **~$0** |
| A few users, ~1,000 diagnoses/mo, cheap model | **~$15-20** |
| Same volume on a frontier model | **~$175** |

**Model choice dominates, not infrastructure.** Infrastructure is roughly $15/mo
regardless; the LLM tier swings the total by an order of magnitude. The
architecture is what makes the cheap tier viable, because each call is small and
focused rather than a whole trace dumped into a large context window.

### Retention policy (decided, in scope for WS-B)

Storage is the sleeper cost. Raw span attributes are retained by design, since
that is what makes normalization reversible, but a 100-span trace can be several
hundred KB of JSONB, so a thousand traces passes a 500MB free tier.

**Decision:** `store_traces.py` exposes
`prune_attributes(conn_fn, older_than_days: int) -> int` which sets
`spans.attributes = '{}'::jsonb` and `spans.attributes_pruned = true` for spans
whose trace was ingested more than `older_than_days` ago. `payload`, `steps`,
`signals`, `divergences`, and `diagnoses` are never pruned. Default 90 days,
declared in `culprit.toml`. Exposed as `culprit prune --older-than 90`.

- `spans.attributes_pruned boolean NOT NULL DEFAULT false` is added to the
  Alembic 0001 schema, not a later migration.
- `normalize.py` never re-reads pruned spans, so pruning cannot corrupt
  analysis; it only forecloses future re-normalization of old traces, which is
  the accepted trade.
- *Rejected: deleting whole old traces* - destroys the L2 reference pool, which
  is the one thing that gets more valuable with age.
- *Rejected: compressing rather than emptying* - Postgres already TOAST-
  compresses large JSONB, so the remaining win is small and the code is not.

**Model is per-deployment configurable** through `culprit.toml` already, so
running Flash Lite in production and a frontier model only for benchmark runs is
a config change. Make that explicit in the config file's commented
documentation.

---

## Verification

**Per workstream:** `uv run pytest tests/test_<owned_modules>.py -v`, fully
offline. Postgres-touching tests need `CULPRIT_TEST_DSN` set.

**Full suite:** `uv run pytest -v`. Must pass with no network and no API key.

**End to end (Integration, DSN required):**
```bash
docker compose up -d postgres redis
uv run alembic upgrade head
uv run culprit worker &
uv run culprit ingest tests/fixtures/otlp/otel_genai_sample.json
uv run culprit diagnose <trace_id>
uv run culprit show <trace_id>
uv run culprit cluster
```

**Benchmark scoring:**
```bash
uv run culprit bench trail --sample tests/fixtures/benchmarks/trail_sample.json
uv run culprit bench who_and_when --ablate l2
```
Report joint accuracy, candidate recall@5, earliness error, and the ablation
delta. **Write the measured numbers into `CLAUDE.md`, favourable or not.**

**Live smoke test, once, out of band:** one real `litellm` adjudication call
against a real trace, because Litmus's own decision log records that mocked
tests demonstrably missed a markdown-fence bug that only a live call surfaced.

---

## Critical reference files

- `Litmus/CLAUDE.md` - decision-log format and rationale style to replicate
- `Litmus/src/litmus/config.py` - frozen dataclass + tomllib + strict layering,
  the pattern for `config.py`
- `Litmus/src/litmus/scoring/llm_judge.py` - prompt constant, XML delimiter,
  de-fence regex, private Pydantic model, narrow except set. The pattern for
  `adjudicate.py` and `cluster_label.py`
- `Litmus/src/litmus/api.py` - plain `def` routes, per-route exception mapping,
  `api_error` logging
- `Litmus/src/litmus/cli.py` - Typer callback, `ThreadPoolExecutor(...).map()`,
  never-raising `_process_case` isolation
- `Litmus/tests/test_scoring_llm_judge.py` - test naming, `_helper()`
  duplication, `monkeypatch.setattr("litellm.completion", ...)`, and the four
  code-fence cases to reuse verbatim

---

## Self-review results

Run against the writing-plans checklist after the plan was complete.

**1. Spec coverage.** Every architectural section maps to a task or a
workstream. Two gaps were found and closed:
- The retention policy was a literal "add this to the plan" TODO. Now a decided
  design with a named function, a column added to revision 0001, a CLI command,
  and rejected alternatives. Assigned to WS-B.
- `NeighborFn` was referenced in the dependency graph but defined nowhere. Now
  declared with the other type aliases in `signals.py`, foundation-owned.

**2. Placeholder scan.** Three fixed:
- WS-B's "bulk insert within a stated budget" now states 5 seconds for 5,000
  spans, with the reason that number is the meaningful one.
- WS-H's "verified against a hand-worked example" now contains the actual
  example with literal expected values.
- Task 3's taxonomy showed one description and `# ...`; the requirement that
  **every** member needs an entry is now enforced by
  `test_describe_all_emits_every_failure_class_so_prompt_and_code_cannot_drift`
  rather than by a comment asking nicely.

**3. Type consistency.** One real inconsistency found and fixed: the Pydantic
`Diagnosis` carried only `confidence` while the `diagnoses` table defined both
`confidence` and `calibrated_confidence`. A later workstream persisting a
`Diagnosis` would have hit a missing column. `Diagnosis` now carries both, and
the reason to keep both is recorded: reporting only the calibrated value hides
how far calibration moved it, which is exactly the number needed to judge
whether calibration helps.

Verified consistent across sections: layer entrypoint names against module
names (`run_detectors`, `contrast`, `adjudicate`, `cluster_diagnoses`), the
`CallFn` / `EmbedFn` / `ConnFn` / `NeighborFn` aliases against their use sites,
and `ContrastResult` against `signals.py` ownership.

**Known deviation from the skill, stated rather than hidden:** the skill
specifies saving to `docs/superpowers/plans/YYYY-MM-DD-<feature>.md`. Plan mode
restricts editing to the assigned plan file, so this lives at
`.claude/plans/mellow-dancing-pebble.md`. Copy it to
`docs/superpowers/plans/2026-08-12-culprit-backend.md` inside the new repo when
scaffolding, so it ships with the code it describes.
