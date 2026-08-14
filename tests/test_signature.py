from culprit.schemas import SpanKind, Step
from culprit.signature import signature_of, sim


def _step(step_index, kind, actor, signature, depth=0) -> Step:
    return Step(
        trace_id="t", step_index=step_index, span_id=f"sp{step_index}", kind=kind,
        actor=actor, depth=depth, tree_path="0", signature=signature, summary="",
        start_ns=0, end_ns=1, duration_ms=1.0,
    )


def test_signature_of_is_a_passthrough_of_step_signature():
    step = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    assert signature_of(step) == step.signature


def test_identical_steps_score_higher_than_a_totally_different_step():
    a = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    b = _step(1, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    c = _step(2, SpanKind.LLM, "other", "llm:other:answer")

    assert sim(a, b) > sim(a, c)


def test_sim_is_symmetric():
    a = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    b = _step(1, SpanKind.LLM, "agent", "llm:agent:call:search_orders")

    assert sim(a, b) == sim(b, a)


def test_llm_step_signature_encodes_the_decision_not_the_model():
    """The load-bearing property from CLAUDE.md: two LLM steps that call
    different tools must not collapse to the same signature just because
    they share a model."""
    call_a = _step(0, SpanKind.LLM, "agent", "llm:agent:call:search_orders")
    call_b = _step(1, SpanKind.LLM, "agent", "llm:agent:call:process_refund")

    assert signature_of(call_a) != signature_of(call_b)
    assert sim(call_a, call_b) < 1.0


def test_tool_outcome_changes_similarity_even_for_the_same_tool():
    ok = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    empty = _step(1, SpanKind.TOOL, "agent", "tool:agent:search_orders:empty")
    other_tool_ok = _step(2, SpanKind.TOOL, "agent", "tool:agent:get_customer:ok")

    assert sim(ok, empty) > sim(ok, other_tool_ok)  # same tool, different outcome
    assert sim(ok, empty) < sim(ok, ok)


def test_depth_delta_reduces_similarity_gracefully():
    shallow = _step(0, SpanKind.AGENT, "orch", "agent:orch:invoke", depth=0)
    deep = _step(1, SpanKind.AGENT, "orch", "agent:orch:invoke", depth=4)

    assert sim(shallow, deep) < sim(shallow, shallow)
