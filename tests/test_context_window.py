from culprit.context_window import build_context_packet
from culprit.schemas import Outcome, SpanKind, Step, Trace
from culprit.signals import Candidate
from culprit.synth_results import make_divergence, make_signal


def _trace(outcome: Outcome = Outcome.FAILURE, task_goal: str = "refund order 88213") -> Trace:
    return Trace(
        trace_id="t1", source="test", outcome=outcome, span_count=0, step_count=0,
        task_goal=task_goal,
    )


def _step(index: int, summary: str = "does something") -> Step:
    return Step(
        trace_id="t1", step_index=index, span_id=f"span-{index}", kind=SpanKind.TOOL,
        actor="agent", depth=0, tree_path="0", signature=f"tool:agent:tool{index}:ok",
        summary=summary, start_ns=0, end_ns=1_000_000, duration_ms=1.0,
    )


def _candidate(step_index: int, signals=None, divergence=None) -> Candidate:
    return Candidate(
        step_index=step_index, span_id=f"span-{step_index}", rank=1, prior=0.6,
        source="l1", signals=signals or [], divergence=divergence,
    )


def test_packet_respects_the_token_budget_on_a_long_trace():
    steps = [_step(i) for i in range(300)]
    candidate = _candidate(150)

    packet = build_context_packet(candidate, _trace(), steps, token_budget=1000)

    assert packet.token_estimate <= 1000


def test_budget_enforcement_drops_the_spine_before_touching_the_zoom_window():
    """The zoom window is where a specific candidate's real evidence lives;
    the spine is only for orientation and must give way first."""
    steps = [_step(i) for i in range(300)]
    ev_signal = make_signal(step_index=150, message="tool returned nothing")
    candidate = _candidate(150, signals=[ev_signal])

    full_budget_packet = build_context_packet(candidate, _trace(), steps, token_budget=12_000)
    tight_packet = build_context_packet(candidate, _trace(), steps, token_budget=600)

    assert tight_packet.token_estimate <= 600
    assert "omitted" in tight_packet.spine
    # The candidate's own zoom block (its detector message) survives even
    # under a budget that forces the spine down to an omission marker.
    assert ev_signal.message in tight_packet.zoom
    assert len(tight_packet.spine) < len(full_budget_packet.spine)


def test_zoom_window_includes_l1_signal_evidence_for_the_candidate_step():
    from culprit.signals import Evidence

    signal = make_signal(
        step_index=5, detector="empty_tool_result", message="search_orders returned nothing",
        evidence=[Evidence(span_id="span-5", step_index=5, field="payload.result_text", excerpt="[]")],
    )
    steps = [_step(i) for i in range(10)]
    candidate = _candidate(5, signals=[signal])

    packet = build_context_packet(candidate, _trace(), steps)

    assert "empty_tool_result" in packet.zoom
    assert "search_orders returned nothing" in packet.zoom
    assert "[]" in packet.zoom


def test_zoom_window_includes_l2_divergence_info_for_the_candidate_step():
    divergence = make_divergence(
        step_index=5, observed_signature="tool:agent:refund_order:ok",
        expected_signatures=[("tool:agent:search_orders:ok", 0.82)],
    )
    steps = [_step(i) for i in range(10)]
    candidate = _candidate(5, divergence=divergence)

    packet = build_context_packet(candidate, _trace(), steps)

    assert "tool:agent:refund_order:ok" in packet.zoom
    assert "search_orders" in packet.zoom


def test_zoom_window_excludes_evidence_for_non_candidate_neighbor_steps():
    """A signal firing at a neighboring step (not the candidate's own step)
    must not bleed its evidence into the candidate's zoom block - it belongs
    to a different candidate's own adjudication call."""
    from culprit.signals import Evidence

    neighbor_signal = make_signal(
        step_index=4, evidence=[Evidence(span_id="span-4", step_index=4, field="x", excerpt="NEIGHBOR-ONLY-TEXT")],
    )
    steps = [_step(i) for i in range(10)]
    candidate = _candidate(5, signals=[])

    packet = build_context_packet(candidate, _trace(), steps)

    assert "NEIGHBOR-ONLY-TEXT" not in packet.zoom


def test_visible_step_indices_covers_spine_zoom_and_terminal():
    steps = [_step(i) for i in range(10)]
    candidate = _candidate(5)

    packet = build_context_packet(candidate, _trace(), steps)

    assert 0 in packet.visible_step_indices  # spine start
    assert 5 in packet.visible_step_indices  # zoom / candidate step
    assert 9 in packet.visible_step_indices  # terminal


def test_task_goal_is_truncated_to_the_char_limit():
    steps = [_step(0)]
    candidate = _candidate(0)
    long_goal = "x" * 5000

    packet = build_context_packet(
        candidate, _trace(task_goal=long_goal), steps, goal_char_limit=100,
    )

    assert len(packet.task_goal) <= 100


def test_terminal_section_reflects_the_trace_outcome():
    steps = [_step(i) for i in range(3)]
    candidate = _candidate(1)

    failure_packet = build_context_packet(candidate, _trace(Outcome.FAILURE), steps)
    success_packet = build_context_packet(candidate, _trace(Outcome.SUCCESS), steps)

    assert "failure" in failure_packet.terminal
    assert "success" in success_packet.terminal


def test_pathological_zoom_content_still_respects_the_budget():
    """Many detectors firing at the same step, each with several 400-char
    evidence excerpts, must not be able to blow the hard cap - the budget is
    a real invariant the fan-out relies on, not a typical-case target."""
    from culprit.signals import Evidence

    big_excerpt = "y" * 400
    many_signals = [
        make_signal(
            step_index=5, detector=f"detector_{i}",
            evidence=[Evidence(span_id="span-5", step_index=5, field="x", excerpt=big_excerpt)] * 3,
        )
        for i in range(20)
    ]
    steps = [_step(i) for i in range(10)]
    candidate = _candidate(5, signals=many_signals)

    packet = build_context_packet(candidate, _trace(), steps, token_budget=800)

    assert packet.token_estimate <= 800


def test_empty_steps_produce_a_packet_with_no_steps_recorded_terminal():
    candidate = _candidate(0)

    packet = build_context_packet(candidate, _trace(), [])

    assert "no steps recorded" in packet.terminal
    assert packet.visible_step_indices == frozenset()


def test_none_trace_falls_back_to_unknown_outcome_and_no_goal():
    candidate = _candidate(0)
    steps = [_step(0)]

    packet = build_context_packet(candidate, None, steps)

    assert "unknown" in packet.terminal
    assert "no task goal recorded" in packet.task_goal
