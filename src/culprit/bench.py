"""Benchmark runner: executes the harness the plan's verification section
references (`uv run culprit bench trail --sample ...`) but Phase 1 never
wired - WS-H shipped the adapters and `bench_score.py`, and the command
itself fell to Integration task I5.

Flow per benchmark trace: `benchmarks/<name>.load_cases` produces canonical
`BenchmarkCase`s, each trace is persisted through `store_traces.write_trace`
(the same table production traffic lands in, `source='benchmark:<name>'`,
upsert so re-runs are idempotent), its cases are persisted through
`store_benchmarks.write_benchmark_cases`, then `pipeline.diagnose` runs
**inline** - no RQ, because the worker cannot run on Windows (RQ calls
`os.fork`, see PHASES.md) and the bench path must work on this development
machine. The resulting `Diagnosis` is persisted through
`store_diagnoses.write_diagnosis` (Integration task I7: `culprit bench`
previously only ever wrote the trace, never the diagnosis or the ground-truth
cases, so a benchmark run left nothing in Postgres for a later calibration
fit to join against - see CLAUDE.md's "L3 adjudication" section) and paired
with every case of its trace into `bench_score.CaseResult`s and scored.

**`--ablate l2` replaces `pipeline.contrast` with a stub that abstains**
(`abstain_reason="ablated:l2"`), which is exactly the "L2 contributed
nothing" path the pipeline already degrades through honestly - the ablation
measures the cascade without L2 by routing through the same code that runs
when L2 fails for real, not through a special-case branch.

**Sampling is deterministic and split-fair**: every k-th trace of the
(sorted-by-id) case list, so a `--sample N` run covers the file evenly
(both TRAIL splits proportionally) and re-runs score the same subset.
"""

import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from culprit import pipeline as pipeline_mod
from culprit.bench_score import BenchReport, CaseResult, layer_ablation_delta, score
from culprit.benchmarks.base import BenchmarkCase
from culprit.benchmarks.registry import BENCHMARKS
from culprit.db import ConnFn, make_pool
from culprit.embed import EmbedFn
from culprit.llm import CallFn
from culprit.logging_config import LOGGER_NAME
from culprit.signals import ContrastResult, Diagnosis
from culprit.store_benchmarks import write_benchmark_cases
from culprit.store_diagnoses import write_diagnosis
from culprit.store_traces import write_trace

logger = logging.getLogger(LOGGER_NAME)

DEFAULT_DATABASE_URL = "postgresql://localhost:5432/culprit"


def pooled_conn_fn(dsn: str | None = None) -> tuple[ConnFn, Callable[[], None], Callable[[], None]]:
    """Build `(conn_fn, recycle, close)` over one pool for a long bench run.

    Every store-layer function checks a connection OUT via `conn_fn()` and
    never returns it - the documented contract is "callers own pooling"
    (store_traces.py), which makes the caller responsible for the lifecycle.
    With a bare `pool.getconn` the bench run exhausts a 10-connection pool
    within a few traces (found empirically in the first TRAIL run: every
    trace after the third failed with "couldn't get a connection after
    30.00 sec"). `recycle()` closes every connection issued since the last
    call, returning it to the pool; `run_benchmark` calls it after each
    trace, once all of that trace's reads and writes are finished. The pool
    is sized above one trace's worst case (write + read + `nearest_successful`
    + `contrast.MAX_REFERENCES` reference loads = ~28 checkouts), so no
    single trace can exhaust it between recycles.

    `recycle()` must use `pool.putconn`, NOT `conn.close()`: with this
    psycopg_pool version, closing a connection obtained via `pool.getconn`
    does not return it to the pool (verified empirically - the pool keeps
    counting it as checked out), it just kills the connection while the
    pool still blocks at max_size."""
    pool = make_pool(dsn or os.environ.get("CULPRIT_DATABASE_URL", DEFAULT_DATABASE_URL), max_size=32)
    pool.open()
    issued: list = []

    def conn_fn():
        conn = pool.getconn()
        # Read queries otherwise leave the connection INTRANS, which makes
        # putconn roll back noisily on recycle; explicit `with
        # conn.transaction():` write blocks are unaffected by autocommit.
        conn.autocommit = True
        issued.append(conn)
        return conn

    def recycle() -> None:
        for conn in issued:
            pool.putconn(conn)
        issued.clear()

    def close() -> None:
        recycle()
        pool.close()

    return conn_fn, recycle, close


@dataclass
class BenchRun:
    """Everything one scored run produced, so the CLI can print and the
    caller can inspect: the headline report over all case rows (a prediction
    is credited when it matches ANY annotated error of the trace), the
    primary-only report (did it find the EARLIEST annotated error), and the
    measured LLM spend."""

    report_all: BenchReport
    report_primary: BenchReport
    n_traces: int
    total_cost_usd: float
    per_trace_errors: dict[str, str] = field(default_factory=dict)


def _contrast_ablated(*args, **kwargs) -> ContrastResult:
    return ContrastResult(
        candidates=[], reference_count=0, abstained=True, abstain_reason="ablated:l2"
    )


def _case_results(cases: list[BenchmarkCase], diagnosis: Diagnosis | None) -> list[CaseResult]:
    results = []
    for case in cases:
        if diagnosis is None:
            results.append(CaseResult(
                truth_step=case.ground_truth_step_index,
                truth_span_id=case.ground_truth_span_id,
                truth_class=case.ground_truth_failure_class,
            ))
            continue
        results.append(CaseResult(
            truth_step=case.ground_truth_step_index,
            predicted_step=diagnosis.root_cause_step_index,
            truth_span_id=case.ground_truth_span_id,
            predicted_span_id=diagnosis.root_cause_span_id,
            truth_class=case.ground_truth_failure_class,
            predicted_class=diagnosis.failure_class,
            predicted_confidence=diagnosis.calibrated_confidence,
            # The ranked shortlist L1/L2 handed to L3, in adjudication order.
            # Populated even when L3 abstains, because adjudication already ran
            # on a real shortlist; only a pipeline that never ran stays empty.
            candidate_steps=[a.step_index for a in diagnosis.adjudications],
            evidenced_candidate_steps=[a.step_index for a in diagnosis.adjudications if a.source in {"l1", "l2", "both"}],
        ))
    return results


def _group_by_trace(cases: list[BenchmarkCase]) -> dict[str, list[BenchmarkCase]]:
    groups: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        groups[case.trace.trace_id].append(case)
    return dict(groups)


def _sample_groups(groups: dict[str, list[BenchmarkCase]], sample: int | None) -> list[str]:
    trace_ids = sorted(groups)
    if sample is None or sample >= len(trace_ids):
        return trace_ids
    stride = len(trace_ids) / sample
    return [trace_ids[int(i * stride)] for i in range(sample)]


def run_benchmark(
    benchmark: str,
    data: Path,
    *,
    sample: int | None = None,
    ablate: str | None = None,
    conn_fn: ConnFn,
    call_fn: CallFn,
    embed_fn: EmbedFn,
    model: str,
    recycle_fn: Callable[[], None] | None = None,
) -> BenchRun:
    """Score `benchmark` (a `BENCHMARKS` registry key) from the merged JSON
    file at `data`. Per-trace isolation: a trace whose pipeline run raises
    contributes abstained rows for all its cases and is named in
    `per_trace_errors`, never aborts the run. `recycle_fn`, when given, is
    called once after each trace so a pool-backed `conn_fn` can reclaim the
    connections that trace checked out (see `pooled_conn_fn`)."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; known: {sorted(BENCHMARKS)}")
    if ablate not in (None, "l2"):
        raise ValueError(f"unsupported ablation {ablate!r}; only 'l2' is wired")

    original_contrast = pipeline_mod.contrast
    if ablate == "l2":
        pipeline_mod.contrast = _contrast_ablated

    try:
        cases = BENCHMARKS[benchmark](data)
        groups = _group_by_trace(cases)
        trace_ids = _sample_groups(groups, sample)

        results: list[CaseResult] = []
        per_trace_errors: dict[str, str] = {}
        total_cost = 0.0
        for i, trace_id in enumerate(trace_ids):
            own = groups[trace_id]
            try:
                case0 = own[0]
                write_trace(conn_fn, case0.trace, case0.spans, case0.steps)
                write_benchmark_cases(conn_fn, benchmark, own)
                diagnosis = pipeline_mod.diagnose(
                    case0.trace, conn_fn=conn_fn, call_fn=call_fn, embed_fn=embed_fn, model=model,
                )
                write_diagnosis(conn_fn, diagnosis)
                total_cost += sum(a.cost_usd or 0.0 for a in diagnosis.adjudications)
            except Exception as e:  # noqa: BLE001 - per-trace isolation, see docstring
                logger.warning(
                    "benchmark trace failed, scored as abstained",
                    extra={"event": "bench_trace_failed", "trace_id": trace_id, "error": str(e)},
                )
                per_trace_errors[trace_id] = str(e)
                diagnosis = None
            results.extend(_case_results(own, diagnosis))
            if recycle_fn is not None:
                recycle_fn()
            logger.info(
                "benchmark trace scored",
                extra={"event": "bench_trace_scored", "trace_id": trace_id,
                       "progress": f"{i + 1}/{len(trace_ids)}"},
            )
    finally:
        pipeline_mod.contrast = original_contrast

    return BenchRun(
        report_all=score(results),
        report_primary=score([r for r, c in zip(results, _flatten_cases(groups, trace_ids)) if c.is_primary]),
        n_traces=len(trace_ids),
        total_cost_usd=total_cost,
        per_trace_errors=per_trace_errors,
    )


def _flatten_cases(groups: dict[str, list[BenchmarkCase]], trace_ids: list[str]) -> list[BenchmarkCase]:
    return [case for trace_id in trace_ids for case in groups[trace_id]]


def ablation_delta(with_layer: BenchRun, without_layer: BenchRun) -> dict[str, float]:
    """The three numbers the layer-ablation study reports (see
    `bench_score.layer_ablation_delta`), computed on the all-rows report."""
    return layer_ablation_delta(with_layer.report_all, without_layer.report_all)
