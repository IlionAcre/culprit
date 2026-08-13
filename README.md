# culprit

![Python](https://img.shields.io/badge/python-3.12%2B-blue)

**Not where it failed. Where it broke.**

culprit ingests a failed LLM agent trace and identifies the step where the run
left the space of trajectories that would have succeeded, plus the reason it
did. Existing observability tools show the trace; culprit finds the cause.

## Status

Early scaffold. The pipeline (ingestion, deterministic detectors, contrastive
trajectory alignment, targeted LLM adjudication, clustering) is under active
development; nothing is wired up yet.

## Quickstart

```bash
uv sync --all-groups
uv run pytest -v
```
