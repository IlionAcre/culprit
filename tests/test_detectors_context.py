"""context_overflow, step_budget_exhausted. `silent_history_truncation`
(same module) gets its own heavy coverage in test_detectors_flagship.py."""

from culprit.detectors.base import build_context
from culprit.detectors.context import context_overflow, step_budget_exhausted
from culprit.detectors.schema import tool_arg_malformed
from culprit.linearize import linearize
from culprit.schemas import Outcome, Trace
from culprit.synth import successful_run
from culprit.synth_inject import inject
from culprit.vocab import openinference, otel_genai


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def _ctx_from_spans(spans):
    steps = linearize(spans)
    spans_by_id = {s.span_id: s for s in spans}
    trace = Trace(
        trace_id=spans[0].trace_id, source="test", outcome=Outcome.SUCCESS,
        root_span_id=spans[0].span_id, span_count=len(spans), step_count=len(steps),
    )
    return build_context(trace, steps, spans_by_id)


def test_context_overflow_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="context_overflow", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in context_overflow(ctx))


def test_context_overflow_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert context_overflow(_ctx(run)) == []


def test_context_overflow_stays_inert_for_otel_genai_request_max_tokens():
    """`gen_ai.request.max_tokens` is an output cap, not a context window."""
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "chat", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": "openai",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.response.model": "gpt-4",
            "gen_ai.usage.input_tokens": 5000,
            "gen_ai.usage.output_tokens": 500,
            "gen_ai.request.max_tokens": 1000,
        },
    }
    span = otel_genai.to_span(raw, "t1")
    ctx = _ctx_from_spans([span])
    assert context_overflow(ctx) == []


def test_tool_arg_malformed_fires_on_openinference_malformed_tool():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "tool", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {"openinference.span.kind": "TOOL", "tool.name": "x", "input.value": "{not json"},
    }
    span = openinference.to_span(raw, "t1")
    ctx = _ctx_from_spans([span])
    signals = tool_arg_malformed(ctx)
    assert len(signals) == 1
    assert signals[0].detector == "tool_arg_malformed"
    assert signals[0].step_index == 0


def test_tool_arg_malformed_fires_on_otel_genai_malformed_tool():
    raw = {
        "span_id": "s1", "parent_span_id": None, "name": "execute_tool", "start_ns": 0, "end_ns": 1,
        "status_code": "STATUS_CODE_OK", "status_message": None,
        "attributes": {
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "x",
            "gen_ai.tool.call.arguments": "{not json",
        },
    }
    span = otel_genai.to_span(raw, "t1")
    ctx = _ctx_from_spans([span])
    signals = tool_arg_malformed(ctx)
    assert len(signals) == 1
    assert signals[0].detector == "tool_arg_malformed"
    assert signals[0].step_index == 0


def test_step_budget_exhausted_fires_at_the_step_before_the_padding_run():
    """`at_step=3` lands the injector's cut on a TOOL step whose signature
    differs from the appended "plan" padding, so the run-length-encoded
    padding run starts cleanly after it rather than absorbing it."""
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="step_budget_exhausted", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in step_budget_exhausted(ctx))


def test_step_budget_exhausted_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert step_budget_exhausted(_ctx(run)) == []
