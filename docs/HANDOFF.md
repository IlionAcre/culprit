# Handoff: Phase 2 closeout

Branch: `phase-2-closeout`, cut from `main` at `e1a17c3`.

You are finishing the last integration work on culprit. Everything below is
current as of this branch point. Read this file and
`AI_docs/INTEGRATION_ITEMS.md` (gitignored, present on this machine) and not
much else. Do not read the full plan and do not survey the codebase; several
previous sessions died on quota and orientation cost is why.

---

## What culprit is

It ingests a failed LLM agent trace and identifies which step *caused* the
failure, versus where it merely *surfaced*. Tagline: "Not where it failed.
Where it broke."

The cascade: L0 normalize and linearize, L1 deterministic detectors, L2
contrastive trajectory diff against successful runs, L3 targeted LLM
adjudication of about five candidates, L5 batch clustering of diagnoses.

The architecture exists because published benchmarks (TRAIL, Who&When, MAST,
AgenTracer) show long-context models perform poorly when handed a whole trace
and asked what went wrong. So deterministic layers narrow first, and the model
adjudicates a handful of pre-narrowed steps with bounded context.

It is a **portfolio artifact**. The written record of why each decision was
made is not documentation overhead, it is half the deliverable.

---

## State at this branch point

`uv run pytest -q` gives **506 passed, 18 skipped**. All 18 skips are
Postgres-gated on `CULPRIT_TEST_DSN` and that is the designed offline state,
not a failure.

**Complete:** all eight build workstreams (L0 ingestion, persistence, L1
detectors, L2 contrastive, L3 adjudication, L5 clustering, service surface,
benchmark harness). Integration I1 (pipeline wiring with per-layer isolation),
I2 (jobs calling the real pipeline, plus `ingest.py`), I3 (Alembic revision
0002). Integration backlog items 1 through 5.

**Not started:** everything in "Your scope" below.

---

## Your scope

### 1. Backlog item 6, token counts

`Adjudication.prompt_tokens` and `completion_tokens` are permanently `None`,
because `CallFn` in `src/culprit/llm.py` returns
`(raw_output, latency_ms, cost_usd)` with no token counts.

This matters more than it looks. The central architectural claim of the project
is that narrowing to five small calls beats one long-context call on both
accuracy and cost, and the `adjudications` table exists specifically so that
can be **proven from measurements rather than asserted**. Right now half that
evidence is missing. litellm's response object carries usage data.

Fix it if the change stays contained. `CallFn` is a foundation-owned type alias
used across layers, so if widening it ripples further than you judge
worthwhile, say so and explain what it would cost rather than doing it badly.

### 2. I6, consolidate the decision log

Eight workstreams each made real decisions that currently live only in commit
messages and module docstrings. Bring them into `CLAUDE.md` in its established
format: bolded decision, rationale, `Rejected: alternative - why` sub-bullets.
Read `CLAUDE.md` first and match it exactly. `git log --oneline` shows what
landed; do not read the diffs.

The parallel-phase rule that kept agents out of `CLAUDE.md` no longer applies.

Also write the `AI_docs/PHASES.md` status table and resume point, and mark
backlog items 2 and 4 resolved in `AI_docs/INTEGRATION_ITEMS.md` (they were
fixed during I1/I2 but the document was never updated).

### 3. I7, calibration, and handle this one with care

The coefficients in `src/culprit/confidence.py` are **hand-set priors that have
never been fitted to anything**, because no labeled data exists yet. Record
that in `CLAUDE.md` in those words.

Do not describe anything in this system as calibrated. This project's decision
log criticizes unearned claims repeatedly and it would be a poor irony to close
it with one.

---

## The measured numbers go in CLAUDE.md, including the bad ones

Full detail in `AI_docs/INTEGRATION_ITEMS.md`. Three matter most:

**L2 contrastive is at 40 percent top-1 against a definition of done that asked
for 80 percent.** Recall@5 is 65 percent (13 of 20 injection kinds); across the
18 kinds where any mechanism reaches `contrast()` at all, top-1 is 44 percent
and recall@5 is 72 percent. Write the real numbers. No weight, threshold, or
gap penalty was tuned to produce them, which is the only reason they are worth
anything.

**Ablation:** removing the point-of-no-return gate and the plateau collapse
changes top-1 by exactly zero, because the cliff term at weight 0.40 decides
ranking on its own. The gate is still load-bearing against false positives on
identical traces, so neither should be removed on the strength of the top-1
number alone.

**RAG null result:** folding retrieved document content into the following LLM
prompt moved L2 accuracy by nothing, byte-identical before and after, because
`contrast.py`, `signature.py`, and `reference.py` never read message text at
all. The change still earned its place by unblocking `unused_retrieval` at L1.

---

## What was never run, and must be stated plainly

A reader must not be able to mistake this for a benchmarked system.

- **No live Postgres exists on this machine.** A container attempt failed on
  host port forwarding, not on code. 18 tests and all of Alembic revision 0002
  are written but unexecuted.
- **The benchmark harness has never scored a real dataset.** TRAIL and
  Who&When are not available offline, so both adapters were built against
  hand-written 3-record fixtures. `TRAIL_CATEGORY_MAP`'s keys are best-effort
  guesses at TRAIL's taxonomy, flagged as such in `trail.py`.

---

## Blocked, not yours

**I4**, the end-to-end DSN-gated test, needs a live Postgres.
**I5**, running the harness against real TRAIL and Who&When data, needs the
datasets. **I7's actual coefficient fitting** needs labels from I5, so the
honest outcome is recording that they remain unfitted priors.

These two are what would convert "builds and passes 506 of its own tests" into
"scores X on a published benchmark." Leave them documented, not faked.

---

## Constraints, non-negotiable

- Python >=3.12, uv only, `src/` layout.
- Zero `async def` anywhere, including FastAPI routes. Fan-out is
  `ThreadPoolExecutor(...).map()`.
- `import litellm` then `litellm.completion(...)`, dotted form, because tests
  monkeypatch `"litellm.completion"`.
- Modules around 200 lines.
- Pydantic v2 at I/O boundaries only, `@dataclass` internally,
  `@dataclass(frozen=True)` for config. `X | None`, never `Optional[X]`.
- Tests fully offline: no network, no API key. Postgres tests stay gated on
  `CULPRIT_TEST_DSN`.
- Per-item and per-layer error isolation. Sentinel results carrying
  `error: str | None`, never a crash that costs the batch.
- Core logic modules keep defaults in their own signatures and never import
  `config`. Only `cli.py` and plugin registries read config.
- **No em dashes anywhere**, in code, comments, docstrings, or prose. Use ` - `.
- Docstrings explain rationale and name the motivating bug. They never restate
  the signature.
- New DDL becomes revision 0003. Do not amend 0001 or 0002.

The Phase 1 freeze on `schemas.py`, `signals.py`, `taxonomy.py`, and
`config.py` is lifted, since it existed only so eight parallel agents would not
serialize on one file. Treat that as a scalpel, not a licence: 506 tests depend
on those shapes. Change one only when the alternative is worse, and say why.

---

## How to work

- `uv run pytest tests/test_<file>.py -q`, only the files your last edit
  touched. **Never `-v`**, it prints a line per test across 506 tests and that
  output is a real fraction of what has been burning sessions.
- Full suite once, at the end, with `-q`.
- Do not re-read files you have already read or just wrote.
- Commit in the smallest standalone units, in dependency order, with explicit
  paths: `git add <paths>` then `git commit`, checking `git diff --cached
  --stat` first. Three agents have died on quota mid-task; small commits are
  what makes that cheap rather than expensive.

---

## What to report

What you changed, what you verified by running it, what stays unverified and
why, and any contract you changed with the reason.

If something in the plan turns out to be wrong now that these pieces are
meeting, say so plainly. That has happened four times in this project and every
single time the plan was the thing at fault, not the code.
