from culprit.candidates import merge_candidates
from culprit.schemas import Outcome, SpanKind, Step, Trace
from culprit.synth_results import make_divergence, make_signal


def _trace(outcome: Outcome = Outcome.FAILURE) -> Trace:
    return Trace(
        trace_id="t1", source="test", outcome=outcome, span_count=0, step_count=0,
        task_goal="do the thing",
    )


def _step(index: int) -> Step:
    return Step(
        trace_id="t1", step_index=index, span_id=f"span-{index}", kind=SpanKind.TOOL,
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
    when no L1 detector and no L2 divergence caught anything."""
    steps = [_step(0), _step(1), _step(2)]

    candidates = merge_candidates([], [], _trace(Outcome.FAILURE), steps)

    assert len(candidates) == 1
    assert candidates[0].step_index == 2  # terminal step
    assert candidates[0].source == "fallback"
    assert candidates[0].rank == 1


def test_merge_does_not_fallback_on_a_non_failure_trace_with_zero_candidates():
    steps = [_step(0), _step(1)]

    candidates = merge_candidates([], [], _trace(Outcome.SUCCESS), steps)

    assert candidates == []


def test_merge_returns_empty_when_there_are_no_steps_at_all_to_fall_back_to():
    candidates = merge_candidates([], [], _trace(Outcome.FAILURE), [])

    assert candidates == []
