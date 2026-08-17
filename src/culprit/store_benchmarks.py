"""`BenchmarkCase` persistence: the writer `benchmark_cases` never had
(Integration task I7). `bench.py::run_benchmark` produces `BenchmarkCase`
rows per trace (see `benchmarks/base.py`) but, before this module, nothing
wrote them anywhere - the ground truth existed only in the in-memory
`BenchReport` a bench run printed, never as rows a later session could join
against `diagnoses` by `trace_id` to fit `confidence.py`'s coefficients.

Split from `store_diagnoses.py` rather than added there: that module's
docstring already states its one concern is the `Diagnosis`/`Signal`/
`DivergenceCandidate`/`Adjudication` object graph, with "a different reason
to change" cited as the justification for splitting `store_clusters.py` out
of the same module. `BenchmarkCase` is benchmark-harness-owned
(`benchmarks/base.py`), not part of that graph, and changes for a different
reason (adapter/ground-truth shape, not diagnosis shape) - a new module
keeps that boundary rather than blurring it, matching this project's
"one store module per concern" convention (`store_traces.py`,
`store_diagnoses.py`, `store_clusters.py`, `store_cluster_labels.py`).

No reader lives here yet. Step 4 of I7 (the calibration fit) reads
`benchmark_cases` with a one-off join query, not a reusable reader function -
if a real caller needs one later (e.g. re-scoring a bench run from persisted
data instead of the in-memory `BenchRun`), it belongs here.
"""

from culprit.benchmarks.base import BenchmarkCase
from culprit.db import ConnFn


def write_benchmark_cases(conn_fn: ConnFn, benchmark: str, cases: list[BenchmarkCase]) -> None:
    """Upserts one `benchmark_cases` row per case. `ON CONFLICT (case_id) DO
    UPDATE`, matching `store_traces.write_trace`'s idempotent-rerun pattern,
    so re-running a benchmark does not fail on the second run's primary-key
    collision. `raw` is left at the DDL's `{}` default: no current adapter or
    caller has adapter-specific detail worth storing there."""
    if not cases:
        return
    conn = conn_fn()
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO benchmark_cases (
                    case_id, benchmark, trace_id, ground_truth_step_index,
                    ground_truth_span_id, ground_truth_class
                ) VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (case_id) DO UPDATE SET
                    benchmark = EXCLUDED.benchmark,
                    trace_id = EXCLUDED.trace_id,
                    ground_truth_step_index = EXCLUDED.ground_truth_step_index,
                    ground_truth_span_id = EXCLUDED.ground_truth_span_id,
                    ground_truth_class = EXCLUDED.ground_truth_class
                """,
                [
                    (
                        case.case_id,
                        benchmark,
                        case.trace.trace_id,
                        case.ground_truth_step_index,
                        case.ground_truth_span_id,
                        case.ground_truth_failure_class,
                    )
                    for case in cases
                ],
            )
