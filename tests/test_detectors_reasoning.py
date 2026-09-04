"""reasoning_turn_defect detector tests."""

from culprit.detectors.base import build_context
from culprit.detectors.reasoning import reasoning_turn_defect
from culprit.schemas import AgentPayload, Outcome, Span, SpanKind, SpanStatus, Step, Trace
from culprit.synth import successful_run


def _ctx_from_agent_messages(messages: list[str]) -> build_context:
    trace_id = "test:reasoning"
    spans_by_id = {}
    steps = []
    for i, content in enumerate(messages):
        span_id = f"span-{i}"
        spans_by_id[span_id] = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            name=f"agent:{i}",
            kind=SpanKind.AGENT,
            status=SpanStatus.OK,
            status_message=None,
            start_ns=i * 1_000_000,
            end_ns=(i + 1) * 1_000_000,
            vocabulary="test",
            attributes={},
            payload=AgentPayload(
                agent_name="expert",
                role="",
                input_text="",
                output_text=content,
                delegated_to=[],
            ),
        )
        steps.append(
            Step(
                trace_id=trace_id,
                step_index=i,
                span_id=span_id,
                kind=SpanKind.AGENT,
                actor="agent",
                depth=0,
                tree_path=str(i),
                signature=f"agent:step-{i}",
                summary=f"step {i}",
                start_ns=i * 1_000_000,
                end_ns=(i + 1) * 1_000_000,
                duration_ms=1.0,
            )
        )
    trace = Trace(
        trace_id=trace_id,
        source="test",
        outcome=Outcome.FAILURE,
        root_span_id="span-0",
        span_count=len(messages),
        step_count=len(messages),
    )
    return build_context(trace, steps, spans_by_id)


def test_reasoning_turn_defect_fires_on_simulated_dataset():
    """Agents creating simulated datasets without retrieval fire."""
    content = "The service timed out. As an alternative, we can simulate the data to identify the result."
    ctx = _ctx_from_agent_messages([content])
    signals = reasoning_turn_defect(ctx)
    assert len(signals) == 1
    assert signals[0].detector == "reasoning_turn_defect"
    assert signals[0].step_index == 0
    assert "simulate" in signals[0].message


def test_reasoning_turn_defect_fires_on_unverified_assumption():
    """Agents introducing unverified assumptions fire."""
    content = "Let's assume that Bob's guesses will always match the coins in each box."
    ctx = _ctx_from_agent_messages([content])
    signals = reasoning_turn_defect(ctx)
    assert len(signals) == 1
    assert "assume" in signals[0].message


def test_reasoning_turn_defect_fires_on_malformed_code_block():
    """Code blocks in reasoning turns that fail AST parsing fire."""
    content = "Here is the plan:\n```python\n2. **Processing the Data**:\n   - Load the dataset.\n```"
    ctx = _ctx_from_agent_messages([content])
    signals = reasoning_turn_defect(ctx)
    assert len(signals) == 1
    assert "malformed or unparseable code block" in signals[0].message


def test_reasoning_turn_defect_no_signal_on_valid_code_and_reasoning():
    """Valid reasoning and valid python code produce no signal."""
    content = "Extracted data cleanly.\n```python\nx = [1, 2, 3]\ntotal = sum(x)\n```\nAll steps verified."
    ctx = _ctx_from_agent_messages([content])
    signals = reasoning_turn_defect(ctx)
    assert signals == []


def test_reasoning_turn_defect_never_fires_on_twenty_clean_synth_runs():
    """Clean synth runs have no assumption or syntax defects in agent turns."""
    for seed in range(20):
        run = successful_run(seed)
        spans_by_id = {s.span_id: s for s in run.spans}
        ctx = build_context(run.trace, run.steps, spans_by_id)
        assert reasoning_turn_defect(ctx) == []
