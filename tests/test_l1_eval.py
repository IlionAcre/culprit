"""Tests for the offline L1 evaluation harness.

These tests pin the scoring logic with a hand-built trace and guard the TRAIL
evidenced per-candidate rate against regression.  They are intentionally
offline: no database, no model call.
"""

from culprit.benchmarks.base import BenchmarkCase
from culprit.l1_eval import (
    EVIDENCED_SOURCES,
    SOURCE_ORDER,
    _compute_rates,
    evaluate_benchmark,
    score_trace,
)
from culprit.schemas import AgentPayload, Outcome, Span, SpanKind, Step, Trace
from culprit.signals import Candidate, Signal
from culprit.taxonomy import FailureClass


def _make_step(index: int, span_id: str, kind: SpanKind = SpanKind.AGENT) -> Step:
    return Step(
        trace_id="test:trace-001",
        step_index=index,
        span_id=span_id,
        kind=kind,
        actor="agent",
        depth=0,
        tree_path=str(index),
        signature="sig",
        summary="summary",
        start_ns=index * 1_000_000,
        end_ns=(index + 1) * 1_000_000,
        duration_ms=1.0,
    )


def _make_span(index: int, span_id: str) -> Span:
    return Span(
        trace_id="test:trace-001",
        span_id=span_id,
        parent_span_id=None,
        name=f"step-{index}",
        kind=SpanKind.AGENT,
        status="ok",
        status_message=None,
        start_ns=index * 1_000_000,
        end_ns=(index + 1) * 1_000_000,
        vocabulary="test",
        attributes={},
        payload=AgentPayload(
            agent_name="agent",
            role="",
            input_text="",
            output_text=f"output {index}",
        ),
    )


def _make_trace(num_steps: int = 4) -> Trace:
    return Trace(
        trace_id="test:trace-001",
        source="test",
        outcome=Outcome.FAILURE,
        span_count=num_steps,
        step_count=num_steps,
    )


def _make_case(ground_truth_step_index: int, num_steps: int = 4) -> BenchmarkCase:
    span_ids = [f"span-{i}" for i in range(num_steps)]
    trace = _make_trace(num_steps)
    steps = [_make_step(i, span_ids[i]) for i in range(num_steps)]
    spans = [_make_span(i, span_ids[i]) for i in range(num_steps)]
    return BenchmarkCase(
        case_id=f"test:case-{ground_truth_step_index}",
        trace=trace,
        spans=spans,
        steps=steps,
        ground_truth_step_index=ground_truth_step_index,
        ground_truth_span_id=span_ids[ground_truth_step_index],
        ground_truth_failure_class=FailureClass.UNKNOWN.value,
        is_primary=True,
    )


def _make_candidate(step_index: int, source: str) -> Candidate:
    return Candidate(
        step_index=step_index,
        span_id=f"span-{step_index}",
        rank=0,
        prior=0.5 if source != "filler" else 0.0,
        source=source,
        signals=[],
        divergence=None,
    )


def _make_signal(step_index: int, detector: str = "tool_error") -> Signal:
    return Signal(
        detector=detector,
        step_index=step_index,
        span_id=f"span-{step_index}",
        severity=1.0,
        category=FailureClass.UNKNOWN.value,
        message="signal",
        evidence=[],
        error=None,
    )


def test_score_trace_counts_sources_and_detectors():
    """A hand-built trace with known ground truth pins the scoring math."""
    cases = [_make_case(1), _make_case(2)]
    # truth steps are {1, 2}
    candidates = [
        _make_candidate(0, "filler"),
        _make_candidate(1, "l1"),
        _make_candidate(2, "l1"),
    ]
    signals = [
        _make_signal(1, "tool_error"),
        _make_signal(2, "tool_error"),
        _make_signal(0, "oscillation"),  # false positive
    ]

    score = score_trace(cases, candidates, signals)

    assert score.cases_total == 2
    assert score.cases_l1_flagged == 2
    assert score.cases_evidenced == 2
    assert score.trace_l1_covered is True

    # filler: 1 candidate, 0 on truth
    assert score.source_counts["filler"] == [1, 0]
    # l1: 2 candidates, both on truth
    assert score.source_counts["l1"] == [2, 2]
    # l2 and both are empty in an L1-only run
    assert score.source_counts["l2"] == [0, 0]
    assert score.source_counts["both"] == [0, 0]
    assert score.source_counts["fallback"] == [0, 0]

    # tool_error fired twice and hit twice; oscillation fired once and missed
    assert score.detector_counts["tool_error"] == [2, 2]
    assert score.detector_counts["oscillation"] == [1, 0]
    # every other detector fired zero signals
    for name, counts in score.detector_counts.items():
        if name not in {"tool_error", "oscillation"}:
            assert counts == [0, 0], f"unexpected activity for {name}"


def test_score_trace_separates_l1_flagged_from_evidenced_shortlist():
    """A signal that is ranked out of the shortlist still counts as L1-flagged."""
    cases = [_make_case(5, num_steps=6)]
    # six equally-severe L1 signals; default max_candidates=5 drops step 5.
    signals = [_make_signal(i, "tool_error") for i in range(6)]
    candidates = [
        Candidate(
            step_index=i,
            span_id=f"span-{i}",
            rank=0,
            prior=1.0,
            source="l1",
            signals=[signals[i]],
            divergence=None,
        )
        for i in range(5)
    ]

    score = score_trace(cases, candidates, signals)

    # L1 saw step 5 (the signal exists), but the candidate did not survive truncation.
    assert score.cases_l1_flagged == 1
    assert score.cases_evidenced == 0


def test_compute_rates_aggregates_evidenced_sources():
    source_counts = {src: [0, 0] for src in SOURCE_ORDER}
    source_counts["filler"] = [100, 25]
    source_counts["l1"] = [50, 10]
    source_counts["l2"] = [10, 2]
    source_counts["both"] = [5, 1]
    filler_rate, evidenced_rate = _compute_rates(source_counts)
    assert filler_rate == 0.25
    assert evidenced_rate == (10 + 2 + 1) / (50 + 10 + 5)


def test_evidenced_sources_set_excludes_fallback_and_filler():
    assert EVIDENCED_SOURCES == {"l1", "l2", "both"}


def test_trail_evidenced_per_candidate_rate_regression_guard():
    """The TRAIL evidenced per-candidate rate must stay near today's 36.9%.
    The floor sits at 0.30 so ordinary drift passes and a regression that
    gives back the Phase 5 gain fails.
    """
    import pytest
    from culprit.l1_eval import BENCHMARK_DATA

    trail_path = BENCHMARK_DATA["trail"]
    if not trail_path.exists():
        pytest.skip(f"benchmark dataset {trail_path} not found; run scripts/prepare_trail.py")
    result = evaluate_benchmark("trail")
    assert result.evidenced_rate >= 0.30, (
        f"TRAIL evidenced per-candidate rate {result.evidenced_rate:.3f} "
        f"dropped below 0.30 regression floor"
    )
