# culprit

![Python](https://img.shields.io/badge/python-3.12%2B-blue)

**Not where it failed. Where it broke.**

culprit ingests a failed LLM agent trace and identifies the step where the run
left the space of trajectories that would have succeeded, plus the reason it
did. Existing observability tools show the trace; culprit finds the cause.

## Status

Pipeline is wired end to end: ingestion (L0), deterministic detectors (L1),
contrastive trajectory alignment (L2), targeted LLM adjudication (L3), and
clustering (L5), plus the service surface (CLI, queue, jobs).

Tested: 506 offline tests pass, 524 pass against a live Postgres+pgvector
instance. One live adjudication call verified against `gemini/gemini-2.5-flash-lite`
with token counts and cost populated. CLI e2e verified against live Postgres and
Redis on a clean OTLP fixture: the pipeline correctly abstained instead of
inventing a fault.

Still not benchmarked: no real TRAIL or Who&When scores exist. Confidence
coefficients are hand-set priors, not fitted to labeled data. The RQ worker
path is unverified on Windows because RQ calls `os.fork`.

## Quickstart

```bash
uv sync --all-groups
uv run pytest -q
```

With live services, create a `.env` (gitignored) containing
`GEMINI_API_KEY`, `CULPRIT_DATABASE_URL`, `CULPRIT_REDIS_URL`, and optionally
`CULPRIT_TEST_DSN`. Load it before each command:

```bash
set -a && . ./.env && set +a
uv run alembic upgrade head
uv run culprit ingest tests/fixtures/otlp/otel_genai_sample.json
uv run culprit diagnose a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit show a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit recluster
```

`culprit worker` is the queue worker entrypoint; it needs Linux or WSL because
RQ forks the process. See `CLAUDE.md` for the full decision log and
`AI_docs/PHASES.md` for the execution roadmap.
