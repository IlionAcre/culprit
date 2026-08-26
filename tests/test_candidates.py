from pathlib import Path

from culprit.candidates import _spread_evenly, merge_candidates
from culprit.schemas import Outcome, SpanKind, Step, Trace
from culprit.synth_results import make_divergence, make_signal


def _trace(outcome: Outcome = Outcome.FAILURE) -> Trace:
    return Trace(
        trace_id="t1", source="test", outcome=outcome, span_count=0, step_count=0,
        task_goal="do the thing",
    )


def _step(index: int, kind: SpanKind = SpanKind.TOOL) -> Step:
    return Step(
        trace_id="t1", step_index=index, span_id=f"span-{index}", kind=kind,
        actor="agent", depth=0, tree_path="0", signature="tool:agent:x:ok",
        summary="does something", start_ns=0, end_ns=1_000_000, duration_ms=1.0,
    )


def test_merge_prioritizes_step_where_both_l1_and_l2_fire_via_colocation_bonus():
    """The +0.15 co-location bonus is the entire point of running two
    independent layers; a step both agree on must outrank a step only one
    layer flagged, even at a lower raw score."""
    steps = [_step(0), _step(1)]
    signals = [make_signal(step_index=0, severity=0.5), make_signal(step_index=1, severity=0.6)]
    divergences = [make_divergence(step_index=0, divergence_score=0.4)]

    candidates = merge_candidates(signals, divergences, _trace(), steps)

    assert candidates[0].step_index == 0
    assert candidates[0].source == "both"
    assert candidates[0].prior == 0.65  # max(0.5, 0.4) + 0.15


def test_merge_caps_prior_at_one_even_when_colocation_bonus_would_exceed_it():
    steps = [_step(0)]
    signals = [make_signal(step_index=0, severity=0.95)]
    divergences = [make_divergence(step_index=0, divergence_score=0.9)]

    candidates = merge_candidates(signals, divergences, _trace(), steps)

    assert candidates[0].prior == 1.0


def test_merge_ranks_by_descending_prior():
    steps = [_step(0), _step(1), _step(2)]
    signals = [
        make_signal(step_index=0, severity=0.3),
        make_signal(step_index=1, severity=0.9),
        make_signal(step_index=2, severity=0.6),
    ]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert [c.step_index for c in candidates] == [1, 2, 0]
    assert [c.rank for c in candidates] == [1, 2, 3]


def test_merge_earlier_step_wins_on_a_prior_tie():
    """Given two equally-scoring candidates, the earlier step is presented
    as the cause and the later one as the echo - the same earliness prior
    that governs L2's own D(i) scoring."""
    steps = [_step(3), _step(7)]
    signals = [make_signal(step_index=3, severity=0.5), make_signal(step_index=7, severity=0.5)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert [c.step_index for c in candidates] == [3, 7]


def test_merge_truncates_to_max_candidates():
    steps = [_step(i) for i in range(8)]
    signals = [make_signal(step_index=i, severity=0.1 * (i + 1)) for i in range(8)]

    candidates = merge_candidates(signals, [], _trace(), steps, max_candidates=5)

    assert len(candidates) == 5
    assert [c.step_index for c in candidates] == [7, 6, 5, 4, 3]


def test_merge_excludes_sentinel_signals_with_error_set():
    """A detector that raised must never manufacture a phantom candidate -
    per-item error isolation (CLAUDE.md). Uses a SUCCESS trace so the
    zero-candidates fallback (tested separately) does not mask this."""
    steps = [_step(0)]
    signals = [make_signal(step_index=0, severity=0.9, error="ValueError: boom")]

    candidates = merge_candidates(signals, [], _trace(Outcome.SUCCESS), steps)

    assert candidates == []


def test_merge_excludes_divergences_with_error_set():
    steps = [_step(0)]
    divergences = [make_divergence(step_index=0, divergence_score=0.9, error="blew up")]

    candidates = merge_candidates([], divergences, _trace(Outcome.SUCCESS), steps)

    assert candidates == []


def test_merge_excludes_whole_trace_sentinel_signals_at_step_index_negative_one():
    """step_index=-1 marks a whole-trace or sentinel signal (signals.py);
    there is no single step to anchor a candidate to."""
    steps = [_step(0)]
    signals = [make_signal(step_index=-1, severity=0.9)]

    candidates = merge_candidates(signals, [], _trace(Outcome.SUCCESS), steps)

    assert candidates == []


def test_merge_emits_fallback_candidate_on_failure_trace_with_zero_real_candidates():
    """L3 must always examine something concrete on a genuine failure, even
    when no L1 detector and no L2 divergence caught anything. Fillers pad the
    shortlist around the fallback, but the fallback itself still leads."""
    steps = [_step(0), _step(1), _step(2)]

    candidates = merge_candidates([], [], _trace(Outcome.FAILURE), steps)

    assert len(candidates) == 3
    assert candidates[0].step_index == 2  # terminal fallback leads
    assert candidates[0].source == "fallback"
    assert candidates[0].rank == 1
    filler_indices = {c.step_index for c in candidates if c.source == "filler"}
    assert filler_indices == {0, 1}


def test_merge_does_not_fallback_on_a_non_failure_trace_with_zero_candidates():
    steps = [_step(0), _step(1)]

    candidates = merge_candidates([], [], _trace(Outcome.SUCCESS), steps)

    assert candidates == []


def test_merge_returns_empty_when_there_are_no_steps_at_all_to_fall_back_to():
    candidates = merge_candidates([], [], _trace(Outcome.FAILURE), [])

    assert candidates == []


def test_merge_fills_one_l1_signal_to_max_candidates_with_real_at_rank_one():
    """Done-when 1: a single real signal must still produce a full shortlist,
    with the evidenced candidate leading and fillers padding the budget."""
    steps = [_step(i, SpanKind.LLM) for i in range(5)]
    signals = [make_signal(step_index=2, severity=0.8)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert len(candidates) == 5
    assert candidates[0].step_index == 2
    assert candidates[0].source == "l1"
    assert candidates[0].rank == 1
    assert all(c.source == "filler" and c.prior == 0.0 for c in candidates[1:])
    assert set(c.step_index for c in candidates) == {0, 1, 2, 3, 4}


def test_merge_leaves_five_real_candidates_unchanged():
    """Done-when 2: when L1/L2 already fill the budget, no fillers enter."""
    steps = [_step(i) for i in range(5)]
    signals = [make_signal(step_index=i, severity=0.5) for i in range(5)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert len(candidates) == 5
    assert all(c.source == "l1" for c in candidates)
    assert all(c.prior == 0.5 for c in candidates)


def test_merge_filler_never_duplicates_real_candidate_step_index():
    """Done-when 3: fillers skip indices already occupied by real candidates
    or the fallback."""
    steps = [_step(i, SpanKind.LLM) for i in range(5)]
    signals = [make_signal(step_index=2, severity=0.8)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    seen: set[int] = set()
    for c in candidates:
        assert c.step_index not in seen
        seen.add(c.step_index)
    assert len(seen) == 5


def test_merge_respects_trace_length_when_fewer_than_max_candidates_steps():
    """Done-when 4: never produce more candidates than there are steps."""
    steps = [_step(i, SpanKind.LLM) for i in range(3)]
    signals = [make_signal(step_index=1, severity=0.8)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert len(candidates) == 3
    assert {c.step_index for c in candidates} == {0, 1, 2}


def test_merge_prefers_llm_kind_steps_then_falls_back_to_semantic_steps():
    """Fillers prefer LLM-kind steps spread evenly; when those run out they
    use other semantic steps rather than unknown/collapsed framework glue."""
    steps = [
        _step(0, SpanKind.LLM),
        _step(1, SpanKind.TOOL),
        _step(2, SpanKind.TOOL),
        _step(3, SpanKind.LLM),
        _step(4, SpanKind.UNKNOWN),
    ]
    signals = [make_signal(step_index=1, severity=0.8)]

    candidates = merge_candidates(signals, [], _trace(), steps)

    fillers = [c for c in candidates if c.source == "filler"]
    # Budget is 5; step 1 is real, step 4 is UNKNOWN, so 3 semantic fillers
    # remain and the shortlist stops at 4 candidates total.
    assert len(fillers) == 3
    filler_kinds = {steps[c.step_index].kind for c in fillers}
    assert SpanKind.UNKNOWN not in filler_kinds
    # Both available LLM-kind slots should be used before falling back to TOOL.
    assert SpanKind.LLM in filler_kinds


def test_spread_evenly_single_filler_picks_midpoint():
    """Task E: exactly one filler requested must not divide by zero; it should
    pick the midpoint of the available items."""
    steps = [_step(i, SpanKind.LLM) for i in range(6)]

    indices = _spread_evenly(steps, set(), 1)

    assert indices == [3]


def test_spread_evenly_single_filler_greedy_top_up_when_midpoint_used():
    """Task E: when the midpoint is already occupied, count==1 still returns
    one available index via the existing greedy top-up."""
    steps = [_step(i, SpanKind.LLM) for i in range(6)]

    indices = _spread_evenly(steps, {3}, 1)

    assert len(indices) == 1
    assert indices[0] in {0, 1, 2, 4, 5}


def test_merge_one_filler_against_many_candidates_does_not_raise():
    """Task E done-when 1: a trace needing exactly one filler against three
    or more candidate steps returns five candidates rather than raising.

    The regression is in _spread_evenly(count=1): the even-spacing formula
    divides by (count - 1), which is zero when a single filler is needed.
    """
    steps = [_step(i, SpanKind.LLM) for i in range(6)]
    signals = [
        make_signal(step_index=0, severity=0.8),
        make_signal(step_index=1, severity=0.7),
        make_signal(step_index=2, severity=0.6),
        make_signal(step_index=3, severity=0.5),
    ]

    candidates = merge_candidates(signals, [], _trace(), steps)

    assert len(candidates) == 5
    assert [c.source for c in candidates[:4]] == ["l1", "l1", "l1", "l1"]
    assert candidates[4].source == "filler"


def test_who_and_when_traces_merge_without_exception():
    """Task E done-when 2: all 184 Who&When traces merge without exception.

    Before the fix, five traces hit the _spread_evenly division-by-zero and
    lost their whole shortlist.
    """
    from culprit.benchmarks.registry import BENCHMARKS
    from culprit.run_detectors import run_detectors

    data = Path(__file__).parents[1] / "data" / "benchmarks" / "who_and_when" / "who_and_when_all.json"
    cases = BENCHMARKS["who_and_when"](data)

    by_trace: dict[tuple[str, int], tuple[Trace, list[Step], dict[str, Span]]] = {}
    for case in cases:
        key = (case.trace.trace_id, id(case.trace))
        if key not in by_trace:
            by_trace[key] = (case.trace, case.steps, {s.span_id: s for s in case.spans})

    failures = []
    for trace, steps, spans_by_id in by_trace.values():
        try:
            signals = run_detectors(trace, steps, spans_by_id)
            merge_candidates(signals, [], trace, steps)
        except Exception as exc:  # noqa: BLE001 - collect all failures for the assertion
            failures.append((trace.trace_id, str(exc)))

    assert len(by_trace) == 184
    assert failures == []
