"""Heavy coverage for the four detectors the product exists to catch
(CLAUDE.md's "L1 detectors" section): `empty_tool_result`,
`silent_history_truncation`, `parameter_drift`, `error_swallowed`. Each
shares the property that the trace looks entirely healthy at the point of
damage (status OK, no exception), so beyond "does it fire", these tests
assert on the healthy-looking span itself and on the evidence a reader would
actually see.
"""

from culprit.detectors.base import build_context
from culprit.detectors.provenance import parameter_drift
from culprit.detectors.tool_errors import empty_tool_result, error_swallowed
from culprit.detectors.context import silent_history_truncation
from culprit.schemas import SpanStatus
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id), spans_by_id


# --- empty_tool_result -------------------------------------------------

def test_empty_tool_result_fires_at_ground_truth_across_every_empty_variant():
    for variant_seed in range(5):
        base = successful_run(seed=variant_seed)
        mutated, gt = inject(base, kind="empty_tool_result", at_step=3)
        ctx, _ = _ctx(mutated)
        signals = empty_tool_result(ctx)
        assert any(s.step_index == gt for s in signals), variant_seed


def test_empty_tool_result_span_is_healthy_looking_status_ok_no_exception():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="empty_tool_result", at_step=3)
    span = mutated.spans[gt]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False


def test_empty_tool_result_evidence_includes_the_next_llm_steps_own_text():
    """The catalogue entry requires evidence from the *next* LLM step so a
    reader sees what the agent did with nothing."""
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="empty_tool_result", at_step=3)
    ctx, _ = _ctx(mutated)
    [signal] = [s for s in empty_tool_result(ctx) if s.step_index == gt]
    fields = {e.field for e in signal.evidence}
    assert "payload.result_text" in fields
    assert "payload.response_messages" in fields


def test_empty_tool_result_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        ctx, _ = _ctx(run)
        assert empty_tool_result(ctx) == []


# --- error_swallowed -----------------------------------------------------

def test_error_swallowed_fires_at_the_troubled_tool_step_not_the_llm_followup():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="error_swallowed", at_step=3)
    ctx, _ = _ctx(mutated)
    signals = error_swallowed(ctx)
    assert any(s.step_index == gt for s in signals)


def test_error_swallowed_tool_step_looks_healthy_and_llm_followup_claims_success():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="error_swallowed", at_step=3)
    tool_span = mutated.spans[gt]
    assert tool_span.status == SpanStatus.OK
    assert tool_span.payload.is_error is False

    followers = [
        mutated.spans[j] for j in (gt + 1, gt + 2)
        if j < len(mutated.spans) and mutated.steps[j].kind.value == "llm"
    ]
    assert any("successfully" in s.payload.response_messages[0].content for s in followers)


def test_error_swallowed_requires_both_a_troubled_tool_and_an_unacknowledging_followup():
    """A tool error alone, with no misleading follow-up, must not fire this
    detector (that is `tool_error`'s job)."""
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="tool_error", at_step=3)
    ctx, _ = _ctx(mutated)
    signals = error_swallowed(ctx)
    assert signals == []


def test_error_swallowed_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        ctx, _ = _ctx(run)
        assert error_swallowed(ctx) == []


# --- silent_history_truncation -------------------------------------------

def test_silent_history_truncation_fires_at_the_drop_step():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="silent_history_truncation", at_step=1)
    ctx, _ = _ctx(mutated)
    signals = silent_history_truncation(ctx)
    assert any(s.step_index == gt for s in signals)


def test_silent_history_truncation_drop_step_looks_healthy():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="silent_history_truncation", at_step=1)
    assert mutated.spans[gt].status == SpanStatus.OK


def test_silent_history_truncation_evidence_carries_the_numeric_drop():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="silent_history_truncation", at_step=1)
    ctx, _ = _ctx(mutated)
    [signal] = [s for s in silent_history_truncation(ctx) if s.step_index == gt]
    assert signal.evidence[0].numeric == mutated.spans[gt].payload.prompt_tokens


def test_silent_history_truncation_does_not_fire_when_tokens_rise_between_calls():
    """The clean run's own final answer step has a higher prompt_tokens than
    the step before it (900 vs 600): a rise must never be mistaken for a
    truncation."""
    for seed in range(20):
        run = successful_run(seed)
        ctx, _ = _ctx(run)
        assert silent_history_truncation(ctx) == []


# --- parameter_drift -------------------------------------------------------

def test_parameter_drift_fires_on_the_fabricated_identifier():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="parameter_drift", at_step=3)
    ctx, _ = _ctx(mutated)
    signals = parameter_drift(ctx)
    assert any(s.step_index == gt for s in signals)
    fired = [s for s in signals if s.step_index == gt][0]
    assert "ORD-99999-ZZZZ" in fired.message


def test_parameter_drift_step_looks_healthy():
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="parameter_drift", at_step=3)
    span = mutated.spans[gt]
    assert span.status == SpanStatus.OK
    assert span.payload.is_error is False


def test_parameter_drift_does_not_flag_the_orders_own_legitimate_id():
    """The order id that genuinely originates in the task goal must never be
    flagged, on any seed: it is identifier-shaped but has real provenance."""
    for seed in range(20):
        run = successful_run(seed)
        ctx, _ = _ctx(run)
        assert parameter_drift(ctx) == []


def test_parameter_drift_ignores_non_string_arguments():
    """A numeric argument like `amount=42.0` must never be treated as an
    identifier, regardless of provenance."""
    base = successful_run(seed=0)
    mutated, gt = inject(base, kind="parameter_drift", at_step=3)
    ctx, _ = _ctx(mutated)
    signals = parameter_drift(ctx)
    assert all("42" not in s.message for s in signals)
