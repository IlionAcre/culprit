import pytest

from culprit.schemas import Outcome, SpanStatus, Trace
from culprit.synth import INJECTION_KINDS, SynthInjectionError, inject, spans_of, steps_of, successful_run


def test_successful_runs_vary_in_length_and_order_across_seeds():
    """A reference pool of identical traces teaches the profile nothing, and
    every legitimate variation would then score as a divergence."""
    runs = [successful_run(seed=i) for i in range(20)]

    lengths = {r.step_count for r in runs}
    assert len(lengths) > 1

    signatures_by_seed = [tuple(s.signature for s in steps_of(r)) for r in runs]
    assert len(set(signatures_by_seed)) > 1


def test_inject_returns_the_ground_truth_index_it_mutated():
    """WS-C and WS-D both assert against this index. If inject did not return
    it, every downstream test would hardcode a magic number."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="empty_tool_result", at_step=3)

    assert index == 3
    assert mutated.trace_id != base.trace_id
    assert mutated.outcome == Outcome.FAILURE


def test_every_injection_kind_is_reachable():
    """Guards the WS-C definition of done, which requires one injection per
    detector."""
    base = successful_run(seed=0)

    for kind in INJECTION_KINDS:
        mutated, index = inject(base, kind=kind, at_step=2)
        assert index >= 0
        assert index < mutated.step_count


def test_injection_kinds_cover_all_twenty_l1_detectors():
    """One entry per detector named in the plan's L1 catalogue: tool_errors
    (3), loops (3), context (3), schema (3), flow (3), retrieval (3),
    provenance (1), timing (1)."""
    assert len(INJECTION_KINDS) == 20


def test_successful_run_is_reproducible_for_the_same_seed():
    """Generation is seeded: rerunning with the same seed must not silently
    drift, or a benchmark comparing two runs of the harness would be
    comparing apples to noise."""
    first = successful_run(seed=7)
    second = successful_run(seed=7)

    assert [s.signature for s in steps_of(first)] == [s.signature for s in steps_of(second)]
    assert first.step_count == second.step_count


def test_spans_and_steps_align_by_index():
    """Every injector below indexes `spans[i]` and `steps[i]` together and
    assumes they describe the same event; this is the invariant that makes
    that safe."""
    trace = successful_run(seed=1)

    spans, steps = spans_of(trace), steps_of(trace)

    assert len(spans) == len(steps)
    assert all(span.span_id == step.span_id for span, step in zip(spans, steps))


def test_inject_never_mutates_the_original_trace_data():
    """A shared base run is reused across many `inject` calls (as in
    `test_every_injection_kind_is_reachable`); one injection corrupting the
    original would silently poison every other call in the same test."""
    base = successful_run(seed=2)
    before = [s.model_copy(deep=True) for s in steps_of(base)]

    inject(base, kind="tool_error", at_step=3)

    assert steps_of(base) == before


def test_empty_tool_result_injection_stays_healthy_at_the_point_of_damage():
    """The four detectors this product exists to catch (empty_tool_result,
    silent_history_truncation, parameter_drift, error_swallowed) share one
    property: the trace looks entirely healthy right where the damage
    happens. This is the injection realism the whole product bets on."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="empty_tool_result", at_step=3)

    span = spans_of(mutated)[index]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False
    assert span.payload.result_text in {"", "[]", "{}", "null", "No results found"}


def test_silent_history_truncation_injection_stays_healthy_at_the_point_of_damage():
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="silent_history_truncation", at_step=1)

    span = spans_of(mutated)[index]
    assert span.status == SpanStatus.OK


def test_parameter_drift_injection_stays_healthy_at_the_point_of_damage():
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="parameter_drift", at_step=3)

    span = spans_of(mutated)[index]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False
    assert "ORD-99999-ZZZZ" in span.payload.arguments["order_id"]


def test_error_swallowed_injection_pairs_a_healthy_tool_step_with_a_confident_llm_step():
    """The tool result is empty (status OK, no exception) and the very next
    LLM step asserts success anyway, with no failure language, which is the
    exact silent-failure pattern this detector exists to catch."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="error_swallowed", at_step=3)

    tool_span = spans_of(mutated)[index]
    assert tool_span.status == SpanStatus.OK
    assert tool_span.payload.result_text in {"", "[]", "{}", "null", "No results found"}

    steps, spans = steps_of(mutated), spans_of(mutated)
    followers = [spans[j] for j in (index + 1, index + 2) if j < len(spans) and steps[j].kind.value == "llm"]
    assert any("successfully" in s.payload.response_messages[0].content for s in followers)


def test_unknown_injection_kind_raises():
    base = successful_run(seed=0)

    with pytest.raises(SynthInjectionError):
        inject(base, kind="not_a_real_kind", at_step=0)


def test_injecting_a_trace_this_module_did_not_generate_raises():
    foreign = Trace(trace_id="not-synthetic", source="other", outcome=Outcome.SUCCESS,
                     span_count=0, step_count=0)

    with pytest.raises(SynthInjectionError):
        inject(foreign, kind="tool_error", at_step=0)
