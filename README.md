# culprit

[![CI](https://github.com/IlionAcre/culprit/actions/workflows/ci.yml/badge.svg)](https://github.com/IlionAcre/culprit/actions/workflows/ci.yml)
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
- Benchmark validation: on TRAIL, L1 evidenced candidates achieve 36.9% (211/572) precision against a 2.7% (2/73) filler baseline, with 76.7% (585/763) case coverage and 93.8% (121/129) trace-level coverage. On Who&When, L1 evidenced candidates achieve 17.4% (15/86) precision against a 10.3% (73/709) filler baseline, with 8.7% (16/184) coverage, after filtering benign execution_result spans in tool_error.

Offline: 617 tests pass, 21 skipped (638 collected) from a clean clone, which is what CI runs. Preparing the benchmark datasets turns two of those skips into passes. With live Postgres + pgvector the 19 database-gated tests run as well; that path was last measured green on 2026-09-06, before the CLI tests were added. CLI e2e (migrate, ingest, diagnose, show, recluster) verified against real Postgres and Redis.

## Cost

About 40k input + 2.5k output tokens per diagnosis. On a Flash-Lite tier that is roughly $0.005 per diagnosis, or about $5 per 1,000 diagnoses/month. The naive one-long-call baseline costs roughly 4x more and performs worse. Every adjudication row stores prompt/completion tokens and cost so the claim is provable.

## Not yet

- Who&When now beats filler (+7.1 percentage points: 17.4% vs 10.3%), but 98.9% of Who&When ground truth is on agent reasoning turns where deterministic checks have modest coverage (8.7%).
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
# or: podman compose up -d
```
If using Podman on Windows, install `podman-compose` first with `uv tool install podman-compose`.

4. Configure environment (this overwrites an existing `.env`):
```bash
cp .env.example .env
```
Fill in `.env` before sourcing it.

When running Docker on Linux or macOS, use localhost:
- `CULPRIT_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/culprit`
- `CULPRIT_REDIS_URL=redis://localhost:6379/0`

When running Podman on Windows, Podman publishes container ports to the WSL2 virtual machine IP instead of Windows localhost. If a native Windows PostgreSQL service runs on port 5432, connecting to localhost reaches the host service without pgvector instead of the container. Redis connections to localhost fail with connection refused.

To find the WSL2 virtual machine IP, run:
```bash
wsl -d podman-machine-default ip addr show eth0
```
Use the returned IPv4 address:
- `CULPRIT_DATABASE_URL=postgresql://postgres:postgres@<vm-ip>:5432/culprit`
- `CULPRIT_REDIS_URL=redis://<vm-ip>:6379/0`

Before running step 6, set `GEMINI_API_KEY` to your key. Then export the variables:
```bash
set -a && . ./.env && set +a
```

5. Apply database migrations:
```bash
uv run alembic upgrade head
```

6. Diagnose a fixture. `culprit run` ingests, diagnoses, and prints the verdict in one command, with no queue and no worker in the path:
```bash
uv run culprit run tests/fixtures/otlp/otel_genai_sample.json
```

The fixture is a clean trace, so the honest result is an abstention rather than an invented fault. To re-read a verdict later, or to see every signal behind it:
```bash
uv run culprit show a1b2c3d4e5f60718293a4b5c6d7e8f90
uv run culprit show a1b2c3d4e5f60718293a4b5c6d7e8f90 --verbose
```

The queued path is there for production, where diagnosis runs out of band. It needs a worker process, and `culprit worker` needs Linux or WSL:
```bash
uv run culprit worker          # in a second terminal
uv run culprit diagnose a1b2c3d4e5f60718293a4b5c6d7e8f90
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
