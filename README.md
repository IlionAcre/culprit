# culprit

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-custom-blue)

**Not where it failed. Where it broke.**

Observability shows you the trace. culprit finds the step that caused the failure - the point where the run left the space of trajectories that would have succeeded.

Long-context models do badly when you hand them an entire trace and ask what went wrong. culprit does not do that. It narrows deterministically, then spends a small model budget adjudicating a handful of pre-narrowed candidates.

## How it works

L0 normalize/linearize   - OTLP GenAI semconv, OpenInference, or raw JSON
       ↓
L1 deterministic detect  - 21-class failure taxonomy
       ↓
L2 contrastive diff      - pgvector reference runs, top candidates
       ↓
L3 targeted adjudication - ~5 candidates via litellm, any provider
       ↓
L5 batch clustering      - recurring failure modes surface

## Seen it work

- Injected fault: a synth trace with an `empty_tool_result` fault was diagnosed as step 3, `silent_empty_result_misread`, confidence 0.90. One call to `gemini/gemini-2.5-flash-lite`, $0.00023.
- Clean trace: the pipeline abstained rather than invent a fault.

Offline: 506 tests pass. With live Postgres + pgvector: 524 tests pass. CLI e2e (migrate, ingest, diagnose, show, recluster) verified against real Postgres and Redis.

## Cost

About 40k input + 2.5k output tokens per diagnosis. On a Flash-Lite tier that is roughly $0.005 per diagnosis, or about $5 per 1,000 diagnoses/month. The naive one-long-call baseline costs roughly 4x more and performs worse. Every adjudication row stores prompt/completion tokens and cost so the claim is provable.

## Not yet

- No TRAIL / Who&When / MAST / AgenTracer benchmark scores yet; harness and adapters exist, datasets were unavailable offline.
- L2 contrastive top-1 is 40% against an 80% aspiration - recorded openly.
- Confidence coefficients are unfitted hand-set priors.
- `culprit worker` needs Linux or WSL because RQ uses `os.fork`; it does not run on Windows.

## Quickstart

Offline:
```bash
uv sync --all-groups
uv run pytest -q
```

With services, create a gitignored `.env` with `GEMINI_API_KEY`, `CULPRIT_DATABASE_URL`, `CULPRIT_REDIS_URL`, and optionally `CULPRIT_TEST_DSN`. Load it before each command:

```bash
set -a && . ./.env && set +a
uv run alembic upgrade head
uv run culprit ingest tests/fixtures/otlp/otel_genai_sample.json
uv run culprit diagnose a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit show a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit recluster
```

## Where the thinking lives

- `CLAUDE.md` - full decision log, every architectural choice with rationale and rejected alternatives.
- `AI_docs/PHASES.md` - roadmap and execution status.
