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
- Benchmark validation: on TRAIL, L1 evidenced candidates achieve 36.9% (211/572) precision against a 2.7% (2/73) filler baseline, with 76.7% (585/763) case coverage and 93.8% (121/129) trace-level coverage. On Who&When, reasoning mistakes remain harder to separate deterministically: L1 evidenced reaches 9.1% (16/176) against a 9.8% (65/661) filler baseline, with 8.7% (16/184) coverage.

Offline: 574 tests pass, 19 skipped (593 collected). With live Postgres + pgvector: 593 tests pass. CLI e2e (migrate, ingest, diagnose, show, recluster) verified against real Postgres and Redis.

## Cost

About 40k input + 2.5k output tokens per diagnosis. On a Flash-Lite tier that is roughly $0.005 per diagnosis, or about $5 per 1,000 diagnoses/month. The naive one-long-call baseline costs roughly 4x more and performs worse. Every adjudication row stores prompt/completion tokens and cost so the claim is provable.

## Not yet

- Who&When reasoning mistakes trail filler: L1 evidenced candidates reach 9.1% against a 9.8% filler baseline. Who&When places 98.9% of its ground truth on agent reasoning turns rather than tool calls, where deterministic string and structural checks struggle to match semantic intent.
- L2 contrastive top-1 is 44% across the 18 injection kinds that reach `contrast()` (against an 80% aspiration), and contributes nothing measurable on real benchmark traces.
- Confidence calibration is inverted, and its historical fit population (~3,021 rows) was lost with no committed fit script. A calibration refit requires live services and benchmark spend, and is out of scope for this phase.
- `culprit worker` needs Linux or WSL because RQ uses `os.fork`; it does not run on Windows.

## Getting the benchmark datasets

Neither prepare script downloads anything; both merge a dataset you already have on disk.

- **TRAIL.** The public release is HuggingFace's `PatronusAI/TRAIL`, gated (request access there), or the ungated ModelScope mirror of the same files (Apache-2.0). Unpack it into one directory holding `GAIA/`, `SWE Bench/`, `processed_annotations_gaia/`, and `processed_annotations_swe_bench/` side by side (see `scripts/prepare_trail.py`'s module docstring for the exact layout). Default location: `data/benchmarks/trail`.
- **Who&When.** Clone `github.com/mingyin1/Agents_Failure_Attribution`. `scripts/prepare_who_and_when.py` reads its `Who&When/Algorithm-Generated/` and `Who&When/Hand-Crafted/` directories (see that script's module docstring). Default location: `data/benchmarks/who_and_when/repo/Who&When`.

With both in place:
```bash
uv run python scripts/prepare_trail.py
uv run python scripts/prepare_who_and_when.py
```

## Quickstart

Walk from clone to a reproduced number:

1. Install dependencies:
```bash
uv sync --all-groups
```

2. Run the offline test suite:
Blank `CULPRIT_TEST_DSN` so the suite skips database-gated tests instead of hanging when containers are stopped:
```bash
CULPRIT_TEST_DSN= uv run pytest -q
```

3. Start services:
```bash
docker compose up -d
# or: podman-compose up -d
```

4. Configure environment (this overwrites an existing `.env`):
```bash
cp .env.example .env
```
Fill in `.env` before sourcing it. For the `docker compose` setup above, set `CULPRIT_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/culprit` and `CULPRIT_REDIS_URL=redis://localhost:6379/0`; set `GEMINI_API_KEY` to your own key for step 6's diagnosis. Then:
```bash
set -a && . ./.env && set +a
```

5. Apply database migrations:
```bash
uv run alembic upgrade head
```

6. Ingest and diagnose a fixture:
```bash
uv run culprit ingest tests/fixtures/otlp/otel_genai_sample.json
uv run culprit diagnose a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit show a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit recluster
```

7. Prepare benchmark data (see "Getting the benchmark datasets" above) and run the harness:
```bash
uv run python scripts/prepare_trail.py
uv run python scripts/prepare_who_and_when.py
uv run python -m culprit.l1_eval
```

## Reproducing the benchmark numbers

The offline evaluation harness requires no services, no API key, and no external spend, but it does need both benchmark datasets downloaded once first (see "Getting the benchmark datasets" above). It evaluates the deterministic detector catalogue against both benchmarks in about two minutes on the full datasets:

```bash
uv run python scripts/prepare_trail.py
uv run python scripts/prepare_who_and_when.py
uv run python -m culprit.l1_eval
```

## Where the thinking lives

- `CLAUDE.md`: full decision log, every architectural choice with rationale and rejected alternatives.
