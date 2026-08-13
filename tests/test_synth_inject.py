import pytest

from culprit.schemas import Outcome, SpanStatus
from culprit.synth import successful_run
from culprit.synth_inject import INJECTION_KINDS, SynthInjectionError, inject


def test_inject_returns_the_ground_truth_index_it_mutated():
    """WS-C and WS-D both assert against this index. If inject did not return
    it, every downstream test would hardcode a magic number."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="empty_tool_result", at_step=3)

    assert index == 3
    assert mutated.trace.trace_id != base.trace.trace_id
    assert mutated.trace.outcome == Outcome.FAILURE


def test_every_injection_kind_is_reachable():
    """Guards the WS-C definition of done, which requires one injection per
    detector."""
    base = successful_run(seed=0)

    for kind in INJECTION_KINDS:
        mutated, index = inject(base, kind=kind, at_step=2)
        assert index >= 0
        assert index < mutated.trace.step_count


def test_injection_kinds_cover_all_twenty_l1_detectors():
    """One entry per detector named in the plan's L1 catalogue: tool_errors
    (3), loops (3), context (3), schema (3), flow (3), retrieval (3),
    provenance (1), timing (1)."""
    assert len(INJECTION_KINDS) == 20


def test_every_injection_kind_survives_a_fuzz_sweep_of_seeds_and_at_step_values():
    """Every injector must produce a valid ground-truth index for any seed
    and any at_step, including out-of-range ones, since WS-C's own tests
    will call these with at_step values this module did not anticipate."""
    for seed in range(10):
        base = successful_run(seed=seed)
        for at_step in range(-1, base.trace.step_count + 2):
            for kind in INJECTION_KINDS:
                mutated, index = inject(base, kind=kind, at_step=at_step)
                assert 0 <= index < mutated.trace.step_count
                assert len(mutated.spans) == len(mutated.steps) == mutated.trace.step_count
                assert mutated.trace.step_count == mutated.trace.span_count


def test_inject_never_mutates_the_original_run_data():
    """A shared base run is reused across many inject() calls (as in
    test_every_injection_kind_is_reachable); one injection corrupting the
    original would silently poison every other call in the same test."""
    base = successful_run(seed=2)
    before = [s.model_copy(deep=True) for s in base.steps]

    inject(base, kind="tool_error", at_step=3)

    assert base.steps == before


def test_empty_tool_result_injection_stays_healthy_at_the_point_of_damage():
    """The four detectors this product exists to catch (empty_tool_result,
    silent_history_truncation, parameter_drift, error_swallowed) share one
    property: the trace looks entirely healthy right where the damage
    happens. This is the injection realism the whole product bets on."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="empty_tool_result", at_step=3)

    span = mutated.spans[index]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False
    assert span.payload.result_text in {"", "[]", "{}", "null", "No results found"}


def test_silent_history_truncation_injection_stays_healthy_at_the_point_of_damage():
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="silent_history_truncation", at_step=1)

    assert mutated.spans[index].status == SpanStatus.OK


def test_parameter_drift_injection_stays_healthy_at_the_point_of_damage():
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="parameter_drift", at_step=3)

    span = mutated.spans[index]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False
    assert "ORD-99999-ZZZZ" in span.payload.arguments["order_id"]


def test_error_swallowed_injection_pairs_a_healthy_tool_step_with_a_confident_llm_step():
    """The tool result is empty (status OK, no exception) and the very next
    LLM step asserts success anyway, with no failure language, which is the
    exact silent-failure pattern this detector exists to catch."""
    base = successful_run(seed=0)

    mutated, index = inject(base, kind="error_swallowed", at_step=3)

    tool_span = mutated.spans[index]
    assert tool_span.status == SpanStatus.OK
    assert tool_span.payload.result_text in {"", "[]", "{}", "null", "No results found"}

    followers = [
        mutated.spans[j] for j in (index + 1, index + 2)
        if j < len(mutated.spans) and mutated.steps[j].kind.value == "llm"
    ]
    assert any("successfully" in s.payload.response_messages[0].content for s in followers)


def test_unknown_injection_kind_raises():
    base = successful_run(seed=0)

    with pytest.raises(SynthInjectionError):
        inject(base, kind="not_a_real_kind", at_step=0)


def test_injecting_something_that_is_not_a_synth_run_raises_with_a_clear_message():
    """A caller who passes a real ingested Trace (from WS-A) instead of a
    SynthRun gets a message naming the mistake, not a bare KeyError from a
    lookup table keyed by trace id."""
    base = successful_run(seed=0)

    with pytest.raises(SynthInjectionError, match="SynthRun"):
        inject(base.trace, kind="tool_error", at_step=0)
