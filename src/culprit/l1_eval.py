"""Offline L1 evaluation harness.

Loads each benchmark through its existing adapter, runs the L1 detector
 catalogue once per trace, merges candidates with no L2 divergence input, and
 scores the resulting shortlist against annotated ground truth.

Runs entirely offline: adapters, detectors, and `candidates.py` only.  No
`psycopg` import, no model call, no database, no API key.

Intended as both an acceptance bar and a regression guard for the L1 layer.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from culprit.benchmarks.base import BenchmarkCase
from culprit.benchmarks.registry import BENCHMARKS
from culprit.candidates import merge_candidates
from culprit.detectors.base import Detector, build_context
from culprit.detectors.registry import DETECTORS
from culprit.schemas import Span, Step, Trace
from culprit.signals import Signal
from culprit.taxonomy import FailureClass

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Paths are relative to the project root (where `uv run python -m culprit.l1_eval`
# is executed).  `__file__` is src/culprit/l1_eval.py, so two parents up is root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DATA = {
    "trail": PROJECT_ROOT / "data" / "benchmarks" / "trail" / "trail_all.json",
    "who_and_when": PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "who_and_when"
    / "who_and_when_all.json",
}
BENCHMARK_PREPARE_SCRIPTS = {
    "trail": "scripts/prepare_trail.py",
    "who_and_when": "scripts/prepare_who_and_when.py",
}
README_DATASETS_SECTION = "Getting the benchmark datasets"


def _format_absent_dataset_message(name: str, path: Path | None = None) -> str:
    target_path = path or BENCHMARK_DATA[name]
    script = BENCHMARK_PREPARE_SCRIPTS.get(name, f"prepare script for {name}")
    return (
        f"Benchmark dataset {name!r} is absent: {target_path} not found. "
        f"To produce it, run {script}. "
        f"See '{README_DATASETS_SECTION}' in README.md."
    )


SOURCE_ORDER = ("filler", "l1", "l2", "both", "fallback")
EVIDENCED_SOURCES = {"l1", "l2", "both"}


def _detectors_without(name: str) -> dict[str, Detector]:
    """Return a copy of the detector registry with one detector removed."""
    return {n: d for n, d in DETECTORS.items() if n != name}


def _run_detectors(
    trace: Trace,
    steps: Sequence[Step],
    spans_by_id: dict[str, Span],
    detectors: dict[str, Detector],
) -> list[Signal]:
    """Run a detector set once, mirroring `run_detectors.py` but accepting an
    arbitrary detector dictionary so leave-one-out analysis can stay local."""
    ctx = build_context(trace, list(steps), spans_by_id)
    signals: list[Signal] = []
    for detector_name, detector in detectors.items():
        try:
            signals.extend(detector(ctx))
        except Exception as exc:  # noqa: BLE001 - per-detector isolation
            signals.append(
                Signal(
                    detector=detector_name,
                    step_index=-1,
                    span_id=None,
                    severity=0.0,
                    category=FailureClass.UNKNOWN.value,
                    message=f"detector {detector_name!r} raised {type(exc).__name__}: {exc}",
                    evidence=[],
                    error=str(exc),
                )
            )
    return signals


@dataclass
class TraceScore:
    """Per-trace scoring results."""

    source_counts: dict[str, list[int]] = field(
        default_factory=lambda: {src: [0, 0] for src in SOURCE_ORDER}
    )
    detector_counts: dict[str, list[int]] = field(
        default_factory=lambda: {name: [0, 0] for name in DETECTORS}
    )
    cases_total: int = 0
    cases_l1_flagged: int = 0
    cases_evidenced: int = 0
    trace_l1_covered: bool = False
    trace_has_truth: bool = False
    failed: bool = False


def score_trace(
    cases: Sequence[BenchmarkCase],
    candidates: Sequence,
    signals: Sequence[Signal],
) -> TraceScore:
    """Score one trace's candidates and signals against its cases.

    A candidate is "on-truth" when its step index matches any annotated ground
    truth step for this trace.  A signal is a "hit" under the same rule.
    """
    score = TraceScore()
    truth_steps = {c.ground_truth_step_index for c in cases}
    score.cases_total = len(cases)
    score.trace_has_truth = bool(truth_steps)

    for cand in candidates:
        src = cand.source
        if src not in score.source_counts:
            continue
        score.source_counts[src][0] += 1
        if cand.step_index in truth_steps:
            score.source_counts[src][1] += 1

    clean_signals = [s for s in signals if s.error is None and s.step_index >= 0]
    l1_signal_steps: set[int] = set()
    for s in clean_signals:
        l1_signal_steps.add(s.step_index)
        if s.detector in score.detector_counts:
            score.detector_counts[s.detector][0] += 1
            if s.step_index in truth_steps:
                score.detector_counts[s.detector][1] += 1

    evidenced_candidate_steps = {
        c.step_index for c in candidates if c.source in EVIDENCED_SOURCES
    }
    score.cases_l1_flagged = sum(
        1 for c in cases if c.ground_truth_step_index in l1_signal_steps
    )
    score.cases_evidenced = sum(
        1 for c in cases if c.ground_truth_step_index in evidenced_candidate_steps
    )
    score.trace_l1_covered = bool(l1_signal_steps & truth_steps)
    return score


@dataclass
class BenchmarkResult:
    """Aggregate scoring results for one benchmark."""

    name: str
    total_traces: int
    processed_traces: int
    failed_traces: int
    source_counts: dict[str, list[int]]
    detector_counts: dict[str, list[int]]
    cases_total: int
    cases_l1_flagged: int
    cases_evidenced: int
    traces_with_truth: int
    traces_l1_covered: int
    filler_rate: float
    evidenced_rate: float
    leave_one_out: dict[str, dict[str, float]] = field(default_factory=dict)


def _add_trace_scores(scores: Sequence[TraceScore]) -> tuple:
    """Sum a list of per-trace scores into aggregate containers."""
    source_counts: dict[str, list[int]] = {src: [0, 0] for src in SOURCE_ORDER}
    detector_counts: dict[str, list[int]] = {name: [0, 0] for name in DETECTORS}
    cases_total = 0
    cases_l1_flagged = 0
    cases_evidenced = 0
    traces_with_truth = 0
    traces_l1_covered = 0
    for score in scores:
        for src, (count, hits) in score.source_counts.items():
            source_counts[src][0] += count
            source_counts[src][1] += hits
        for det, (count, hits) in score.detector_counts.items():
            detector_counts[det][0] += count
            detector_counts[det][1] += hits
        cases_total += score.cases_total
        cases_l1_flagged += score.cases_l1_flagged
        cases_evidenced += score.cases_evidenced
        traces_with_truth += int(score.trace_has_truth)
        traces_l1_covered += int(score.trace_l1_covered)
    return (
        source_counts,
        detector_counts,
        cases_total,
        cases_l1_flagged,
        cases_evidenced,
        traces_with_truth,
        traces_l1_covered,
    )


def _compute_rates(
    source_counts: dict[str, list[int]],
) -> tuple[float, float]:
    """Return (filler_rate, evidenced_rate) from source counts."""
    filler_count, filler_hits = source_counts.get("filler", [0, 0])
    filler_rate = (filler_hits / filler_count) if filler_count else 0.0

    evidenced_count = sum(
        source_counts[src][0] for src in EVIDENCED_SOURCES if src in source_counts
    )
    evidenced_hits = sum(
        source_counts[src][1] for src in EVIDENCED_SOURCES if src in source_counts
    )
    evidenced_rate = (evidenced_hits / evidenced_count) if evidenced_count else 0.0
    return filler_rate, evidenced_rate


def evaluate_benchmark(
    name: str,
    detectors: dict[str, Detector] | None = None,
    data_path: Path | None = None,
) -> BenchmarkResult:
    """Load a benchmark, run L1, and score every trace.

    `detectors` defaults to the full registry; callers may pass a subset for
    leave-one-out analysis.  Per-trace failures are caught and logged rather
    than aborting the run.
    """
    detectors = DETECTORS if detectors is None else detectors
    path = data_path or BENCHMARK_DATA[name]
    if not path.exists():
        raise FileNotFoundError(_format_absent_dataset_message(name, path))
    adapter = BENCHMARKS[name]
    cases = adapter(path)
    cases_by_trace: dict[str, list[BenchmarkCase]] = defaultdict(list)
    for case in cases:
        cases_by_trace[case.trace.trace_id].append(case)

    trace_scores: list[TraceScore] = []
    failed_traces = 0
    for trace_id, trace_cases in cases_by_trace.items():
        try:
            case = trace_cases[0]
            spans_by_id = {s.span_id: s for s in case.spans}
            signals = _run_detectors(
                case.trace, case.steps, spans_by_id, detectors
            )
            candidates = merge_candidates(signals, [], case.trace, case.steps)
            trace_scores.append(score_trace(trace_cases, candidates, signals))
        except Exception as exc:  # noqa: BLE001 - per-trace isolation
            failed_traces += 1
            logger.warning(
                "trace failed in offline L1 evaluation",
                extra={
                    "event": "l1_eval_trace_failed",
                    "benchmark": name,
                    "trace_id": trace_id,
                    "error": str(exc),
                },
            )

    (
        source_counts,
        detector_counts,
        cases_total,
        cases_l1_flagged,
        cases_evidenced,
        traces_with_truth,
        traces_l1_covered,
    ) = _add_trace_scores(trace_scores)
    filler_rate, evidenced_rate = _compute_rates(source_counts)

    return BenchmarkResult(
        name=name,
        total_traces=len(cases_by_trace),
        processed_traces=len(trace_scores),
        failed_traces=failed_traces,
        source_counts=source_counts,
        detector_counts=detector_counts,
        cases_total=cases_total,
        cases_l1_flagged=cases_l1_flagged,
        cases_evidenced=cases_evidenced,
        traces_with_truth=traces_with_truth,
        traces_l1_covered=traces_l1_covered,
        filler_rate=filler_rate,
        evidenced_rate=evidenced_rate,
    )


def _leave_one_out(name: str, data_path: Path | None = None) -> dict[str, dict[str, float]]:
    """Recompute filler and evidenced rates with each detector unregistered."""
    result: dict[str, dict[str, float]] = {}
    full = evaluate_benchmark(name, data_path=data_path)
    result["(all registered)"] = {
        "filler_rate": full.filler_rate,
        "evidenced_rate": full.evidenced_rate,
    }
    for detector_name in DETECTORS:
        subset = _detectors_without(detector_name)
        run = evaluate_benchmark(name, detectors=subset, data_path=data_path)
        result[detector_name] = {
            "filler_rate": run.filler_rate,
            "evidenced_rate": run.evidenced_rate,
        }
    return result


def _format_rate(hits: int, count: int) -> str:
    if count == 0:
        return "0/0  ---"
    rate = hits / count
    return f"{hits}/{count}  {rate:6.1%}"


def _print_benchmark(result: BenchmarkResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {result.name}")
    print(f"Traces: {result.processed_traces}/{result.total_traces} processed")
    if result.failed_traces:
        print(f"Failed traces: {result.failed_traces}")

    print("\n1. Per candidate source (filler is the baseline)")
    for src in SOURCE_ORDER:
        count, hits = result.source_counts[src]
        print(f"   {src:12s} {_format_rate(hits, count)}")
    print(
        f"   {'evidenced':12s} {_format_rate(sum(result.source_counts[src][1] for src in EVIDENCED_SOURCES), sum(result.source_counts[src][0] for src in EVIDENCED_SOURCES))}"
    )

    print("\n2. Per detector (per-signal rate, benchmark filler rate shown)")
    print(f"   {'detector':30s} {'signals':>10s} {'hits':>8s} {'rate':>8s} {'vs filler':>10s}")
    for det in sorted(result.detector_counts):
        count, hits = result.detector_counts[det]
        rate = (hits / count) if count else 0.0
        marker = "=" if count and rate >= result.filler_rate else "<"
        print(
            f"   {det:30s} {count:>10d} {hits:>8d} {rate:>7.1%} {marker} {result.filler_rate:>6.1%}"
        )

    print("\n3. Leave-one-out (evidenced and filler rates)")
    print(f"   {'unregistered':30s} {'evidenced':>10s} {'filler':>10s}")
    for det, rates in result.leave_one_out.items():
        print(
            f"   {det:30s} {rates['evidenced_rate']:>9.1%} {rates['filler_rate']:>9.1%}"
        )

    print("\n4. Coverage")
    case_l1_rate = (
        result.cases_l1_flagged / result.cases_total if result.cases_total else 0.0
    )
    case_evidenced_rate = (
        result.cases_evidenced / result.cases_total if result.cases_total else 0.0
    )
    trace_rate = (
        result.traces_l1_covered / result.traces_with_truth
        if result.traces_with_truth
        else 0.0
    )
    print("   Per case:")
    print(
        f"      L1 flagged truth step at all  : {_format_rate(result.cases_l1_flagged, result.cases_total)}"
    )
    print(
        f"      Reached evidenced shortlist   : {_format_rate(result.cases_evidenced, result.cases_total)}"
    )
    print("   Trace-level:")
    print(
        f"      L1 flagged at least one truth : {_format_rate(result.traces_l1_covered, result.traces_with_truth)}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for `uv run python -m culprit.l1_eval`."""
    import argparse
    import sys

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Offline L1 evaluation harness.")
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARK_DATA.keys()),
        default=None,
        help="Specifically requested benchmark dataset (trail or who_and_when).",
    )
    args = parser.parse_args(argv)

    if args.dataset:
        path = BENCHMARK_DATA[args.dataset]
        if not path.exists():
            print(_format_absent_dataset_message(args.dataset, path), file=sys.stderr)
            sys.exit(1)
        result = evaluate_benchmark(args.dataset)
        result.leave_one_out = _leave_one_out(args.dataset)
        _print_benchmark(result)
        return

    present = [name for name in ("trail", "who_and_when") if BENCHMARK_DATA[name].exists()]
    absent = [name for name in ("trail", "who_and_when") if not BENCHMARK_DATA[name].exists()]

    if not present:
        for name in absent:
            print(
                _format_absent_dataset_message(name, BENCHMARK_DATA[name]),
                file=sys.stderr,
            )
        sys.exit(1)

    for name in present:
        result = evaluate_benchmark(name)
        result.leave_one_out = _leave_one_out(name)
        _print_benchmark(result)

    for name in absent:
        print(f"\n{_format_absent_dataset_message(name, BENCHMARK_DATA[name])}")


if __name__ == "__main__":
    main()
