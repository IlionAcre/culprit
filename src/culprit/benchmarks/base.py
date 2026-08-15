"""Shared benchmark-adapter shape: `BenchmarkCase`, the canonical unit every
adapter (`trail.py`, `who_and_when.py`) produces, and the `BenchmarkAdapter`
Protocol both conform to.

Follows the house pattern from `detectors/base.py`: a `Protocol` plus
module-level callables registered in `registry.py`, not classes. A
`BenchmarkCase` bundles a real canonical `Trace`/`Span`/`Step` triple (the
same shapes `store_traces.write_trace` persists) with one ground-truth
annotation, never a parallel benchmark-only representation - that is the
entire point of the harness (CLAUDE.md, "Benchmark harness": adapters produce
canonical `Trace` objects that go into the same tables as production
traffic). `@dataclass`, not `BaseModel`: a `BenchmarkCase` never crosses an
I/O boundary itself, it just carries objects that do.

A single TRAIL trace with multiple annotated errors becomes multiple
`BenchmarkCase` rows sharing one `trace`/`spans`/`steps` object graph, one per
annotated error (see `trail.py`). `is_primary` marks the earliest of those,
which is the row `bench_score.py` uses for earliness/step-accuracy metrics;
the non-primary rows exist only so "credit any annotated error, but also
report whether the earliest one was found" (CLAUDE.md) can be implemented
without a second data structure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from culprit.schemas import Span, Step, Trace


@dataclass
class BenchmarkCase:
    case_id: str
    trace: Trace
    spans: list[Span]
    steps: list[Step]
    ground_truth_step_index: int
    ground_truth_span_id: str | None
    ground_truth_failure_class: str  # a culprit.taxonomy.FailureClass value
    is_primary: bool = True


class BenchmarkAdapter(Protocol):
    """`(path) -> list[BenchmarkCase]`. One fixture file, however many cases
    it contains; per-record malformed data is each adapter's own concern to
    isolate (CLAUDE.md: per-item error isolation everywhere), not this
    Protocol's."""

    def __call__(self, path: Path) -> list[BenchmarkCase]: ...
