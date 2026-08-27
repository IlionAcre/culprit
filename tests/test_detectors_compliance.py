"""instruction_noncompliance detector tests."""

from culprit.detectors.base import build_context
from culprit.detectors.compliance import instruction_noncompliance
from culprit.schemas import LlmPayload, Message, Outcome, Span, SpanKind, SpanStatus, Step, Trace
from culprit.synth import successful_run


def _ctx_from_llm_turns(turns: list[tuple[str, str]]) -> build_context:
    """Build a trace of consecutive LLM steps, one per (request, response)."""
    trace_id = "test:compliance"
    spans_by_id = {}
    steps = []
    for i, (request_text, response_text) in enumerate(turns):
        span_id = f"span-{i}"
        spans_by_id[span_id] = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            name="plan",
            kind=SpanKind.LLM,
            status=SpanStatus.OK,
            status_message=None,
            start_ns=i * 1_000_000,
            end_ns=(i + 1) * 1_000_000,
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
        steps.append(Step(
            trace_id=trace_id,
            step_index=i,
            span_id=span_id,
            kind=SpanKind.LLM,
            actor="agent",
            depth=0,
            tree_path=str(i),
            signature="llm:plan",
            summary="plan",
            start_ns=i * 1_000_000,
            end_ns=(i + 1) * 1_000_000,
            duration_ms=1.0,
        ))
    trace = Trace(
        trace_id=trace_id,
        source="test",
        outcome=Outcome.FAILURE,
        root_span_id="span-0",
        span_count=len(turns),
        step_count=len(turns),
    )
    return build_context(trace, steps, spans_by_id)


def _ctx_from_llm(request_text: str, response_text: str) -> build_context:
    """Build a one-step trace with a single LLM span and run the detector."""
    return _ctx_from_llm_turns([(request_text, response_text)])


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
    """Clean synth runs issue no required-literal instructions, so the
    detector has nothing to check. This is the project's per-detector
    false-positive bar: zero signals across 20 clean successes."""
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


def test_instruction_noncompliance_fires_outside_the_trail_vocabulary():
    """A required literal earns a signal on its own shape, not because its
    text happens to contain "plan" or "code"."""
    ctx = _ctx_from_llm(
        "When you are finished, end with <end_answer> and stop there.",
        "Here is my answer. Done.",
    )
    signals = instruction_noncompliance(ctx)
    assert len(signals) == 1
    assert "<end_answer>" in signals[0].message


def test_instruction_noncompliance_fires_on_every_noncompliant_step():
    """Each step is judged against its own instruction, so a trace that
    violates the same constraint three times yields three signals."""
    turns = [("End with <end_plan> please.", "a plan with no delimiter")] * 3
    signals = instruction_noncompliance(_ctx_from_llm_turns(turns))
    assert [s.step_index for s in signals] == [0, 1, 2]
