"""instruction_noncompliance detector tests."""

from culprit.detectors.base import build_context
from culprit.detectors.compliance import instruction_noncompliance
from culprit.schemas import LlmPayload, Message, Outcome, Span, SpanKind, SpanStatus, Step, Trace
from culprit.synth import successful_run


def _ctx_from_llm(request_text: str, response_text: str) -> build_context:
    """Build a one-step trace with a single LLM span and run the detector."""
    trace_id = "test:compliance"
    span_id = "span-0"
    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name="plan",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        status_message=None,
        start_ns=0,
        end_ns=1_000_000,
        vocabulary="test",
        attributes={},
        payload=LlmPayload(
            provider="test",
            model="test-model",
            request_messages=[Message(role="user", content=request_text)],
            response_messages=[Message(role="assistant", content=response_text)],
            tool_calls=[],
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )
    step = Step(
        trace_id=trace_id,
        step_index=0,
        span_id=span_id,
        kind=SpanKind.LLM,
        actor="agent",
        depth=0,
        tree_path="0",
        signature="llm:plan",
        summary="plan",
        start_ns=0,
        end_ns=1_000_000,
        duration_ms=1.0,
    )
    trace = Trace(
        trace_id=trace_id,
        source="test",
        outcome=Outcome.FAILURE,
        root_span_id=span_id,
        span_count=1,
        step_count=1,
    )
    spans_by_id = {span_id: span}
    return build_context(trace, [step], spans_by_id)


def test_instruction_noncompliance_fires_on_missing_end_plan():
    """TRAIL's canonical plan prompt requires <end_plan>; absence fires."""
    request = (
        "Develop a high-level plan. "
        "After writing the final step of the plan, write the '\\n<end_plan>' tag and stop there."
    )
    ctx = _ctx_from_llm(request, "1. Do thing A.\n2. Do thing B.")
    signals = instruction_noncompliance(ctx)
    assert len(signals) == 1
    assert signals[0].detector == "instruction_noncompliance"
    assert signals[0].step_index == 0
    assert "<end_plan>" in signals[0].message


def test_instruction_noncompliance_fires_on_missing_end_code():
    """Code-generation prompts that require <end_code> fire when absent."""
    request = (
        "You are a coding assistant. "
        "The code sequence must end with '<end_code>' sequence."
    )
    ctx = _ctx_from_llm(request, "x = 1 + 1")
    signals = instruction_noncompliance(ctx)
    assert len(signals) == 1
    assert signals[0].detector == "instruction_noncompliance"
    assert "<end_code>" in signals[0].message


def test_instruction_noncompliance_no_signal_when_literal_present():
    """A response that includes the required literal is compliant."""
    request = (
        "Develop a high-level plan. "
        "After writing the final step of the plan, write the '\\n<end_plan>' tag and stop there."
    )
    ctx = _ctx_from_llm(request, "1. Do thing A.\n2. Do thing B.\n<end_plan>")
    assert instruction_noncompliance(ctx) == []


def test_instruction_noncompliance_ignores_history_instructions():
    """A required literal mentioned only in prior-turn history, not the
    current instruction, must not fire on this step."""
    trace_id = "test:compliance-history"
    span_id = "span-0"
    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=None,
        name="execute",
        kind=SpanKind.LLM,
        status=SpanStatus.OK,
        status_message=None,
        start_ns=0,
        end_ns=1_000_000,
        vocabulary="test",
        attributes={},
        payload=LlmPayload(
            provider="test",
            model="test-model",
            request_messages=[
                Message(role="system", content="You are a coding assistant."),
                # First user message is the current task; no <end_plan> requirement here.
                Message(role="user", content="New task: execute the plan."),
                # Later user messages are history and contain the stale constraint.
                Message(role="user", content="Earlier: write the '\\n<end_plan>' tag."),
            ],
            response_messages=[Message(role="assistant", content="Executing step 1.")],
            tool_calls=[],
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        ),
    )
    step = Step(
        trace_id=trace_id,
        step_index=0,
        span_id=span_id,
        kind=SpanKind.LLM,
        actor="agent",
        depth=0,
        tree_path="0",
        signature="llm:execute",
        summary="execute",
        start_ns=0,
        end_ns=1_000_000,
        duration_ms=1.0,
    )
    trace = Trace(
        trace_id=trace_id,
        source="test",
        outcome=Outcome.FAILURE,
        root_span_id=span_id,
        span_count=1,
        step_count=1,
    )
    ctx = build_context(trace, [step], {span_id: span})
    assert instruction_noncompliance(ctx) == []


def test_instruction_noncompliance_never_fires_on_twenty_clean_synth_runs():
    """Clean synth runs have no plan/code delimiter instructions."""
    for seed in range(20):
        run = successful_run(seed)
        spans_by_id = {s.span_id: s for s in run.spans}
        ctx = build_context(run.trace, run.steps, spans_by_id)
        assert instruction_noncompliance(ctx) == []


def test_instruction_noncompliance_no_signal_for_html_noise():
    """Common HTML-ish tags introduced by instruction phrases are ignored."""
    request = "End with <b>bold</b> and <i>italic</i>."
    ctx = _ctx_from_llm(request, "no bold here")
    assert instruction_noncompliance(ctx) == []
