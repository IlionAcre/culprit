"""output_schema_violation, tool_arg_malformed, hallucinated_tool."""

from culprit.detectors.base import build_context
from culprit.detectors.schema import hallucinated_tool, output_schema_violation, tool_arg_malformed
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_output_schema_violation_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="output_schema_violation", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in output_schema_violation(ctx))


def test_output_schema_violation_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert output_schema_violation(_ctx(run)) == []


def test_tool_arg_malformed_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="tool_arg_malformed", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in tool_arg_malformed(ctx))


def test_tool_arg_malformed_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert tool_arg_malformed(_ctx(run)) == []


def test_hallucinated_tool_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="hallucinated_tool", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in hallucinated_tool(ctx))


def test_hallucinated_tool_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert hallucinated_tool(_ctx(run)) == []


def test_hallucinated_tool_does_not_flag_a_tool_call_whose_execution_appears_later():
    """`tool_universe` is built from the whole trace, so a tool call is safe
    even though its TOOL execution step comes after the calling LLM step."""
    run = successful_run(seed=1)
    ctx = _ctx(run)
    call_step = next(s for s in run.steps if "call:search_orders" in s.signature)
    assert call_step.step_index < next(
        s.step_index for s in run.steps if "search_orders:ok" in s.signature
    )
    assert hallucinated_tool(ctx) == []
