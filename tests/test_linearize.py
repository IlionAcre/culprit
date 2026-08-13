from culprit.linearize import linearize, orphan_span_ids
from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Message,
    Span,
    SpanKind,
    SpanStatus,
    ToolCallRequest,
    ToolPayload,
)


def _span(
    span_id: str, kind: SpanKind, *, parent: str | None, start: int, end: int | None = None,
    payload=None, name: str | None = None,
) -> Span:
    return Span(
        trace_id="t1", span_id=span_id, parent_span_id=parent, name=name or span_id,
        kind=kind, status=SpanStatus.OK, status_message=None, start_ns=start,
        end_ns=end if end is not None else start + 100, vocabulary="test", attributes={},
        payload=payload,
    )


def _agent(span_id: str, agent_name: str, *, parent: str | None, start: int) -> Span:
    return _span(
        span_id, SpanKind.AGENT, parent=parent, start=start,
        payload=AgentPayload(agent_name=agent_name, role="", input_text="", output_text="", delegated_to=[]),
    )


def _llm(span_id: str, *, parent: str, start: int, tool_calls=None) -> Span:
    return _span(
        span_id, SpanKind.LLM, parent=parent, start=start,
        payload=LlmPayload(
            provider="p", model="m", request_messages=[Message(role="user", content="hi")],
            response_messages=[Message(role="assistant", content="ok")],
            tool_calls=tool_calls or [],
        ),
    )


def _tool(span_id: str, *, parent: str, start: int, result: str = "ok", is_error: bool = False) -> Span:
    return _span(
        span_id, SpanKind.TOOL, parent=parent, start=start,
        payload=ToolPayload(
            tool_name="do_thing", call_id="c1", arguments_json="{}", arguments={},
            result_text=result, result_len=len(result), is_error=is_error,
            error_message="boom" if is_error else None,
        ),
    )


def test_linearize_returns_empty_list_for_no_spans():
    assert linearize([]) == []


def test_linearize_dfs_pre_order_visits_root_then_children_in_start_ns_order():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("second-child", parent="root", start=200),
        _llm("first-child", parent="root", start=100),
    ]

    steps = linearize(spans)

    assert [s.span_id for s in steps] == ["root", "first-child", "second-child"]
    assert [s.step_index for s in steps] == [0, 1, 2]


def test_linearize_is_deterministic_under_sibling_timestamp_ties():
    """Dedicated determinism test (CLAUDE.md linearization rule 1): OTel
    exporters routinely emit siblings with identical millisecond-truncated
    start times. Without the span_id secondary sort key, the same set of
    spans could linearize differently depending on input list order,
    silently poisoning L2 alignment. This must hold regardless of the order
    spans are handed to linearize() in."""
    tied_start = 500
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("bbb", parent="root", start=tied_start),
        _llm("aaa", parent="root", start=tied_start),
        _llm("ccc", parent="root", start=tied_start),
    ]

    order_1 = [s.span_id for s in linearize(spans)]
    order_2 = [s.span_id for s in linearize(list(reversed(spans)))]
    order_3 = [s.span_id for s in linearize([spans[0], spans[2], spans[3], spans[1]])]

    assert order_1 == order_2 == order_3 == ["root", "aaa", "bbb", "ccc"]


def test_linearize_only_semantic_kinds_become_steps():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _span("chain-1", SpanKind.CHAIN, parent="root", start=50),
        _llm("llm-1", parent="root", start=100),
        _span("embed-1", SpanKind.EMBEDDING, parent="root", start=150),
    ]

    steps = linearize(spans)

    assert [s.kind for s in steps] == [SpanKind.AGENT, SpanKind.LLM]
    assert [s.span_id for s in steps] == ["root", "llm-1"]


def test_linearize_collapses_a_non_semantic_span_into_the_next_semantic_step():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _span("chain-1", SpanKind.CHAIN, parent="root", start=50),
        _llm("llm-1", parent="root", start=100),
    ]

    steps = linearize(spans)

    llm_step = next(s for s in steps if s.span_id == "llm-1")
    assert llm_step.collapsed_span_ids == ["chain-1"]


def test_linearize_collapses_a_trailing_non_semantic_span_into_the_last_step():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("llm-1", parent="root", start=100),
        _span("guardrail-1", SpanKind.GUARDRAIL, parent="root", start=200),
    ]

    steps = linearize(spans)

    assert steps[-1].span_id == "llm-1"
    assert steps[-1].collapsed_span_ids == ["guardrail-1"]


def test_linearize_step_index_is_dense_zero_based_across_collapsed_spans():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _span("chain-1", SpanKind.CHAIN, parent="root", start=25),
        _llm("llm-1", parent="root", start=50),
        _span("chain-2", SpanKind.CHAIN, parent="root", start=75),
        _tool("tool-1", parent="root", start=100),
    ]

    steps = linearize(spans)

    # 5 spans, 2 of them (both CHAIN) collapsed - 3 dense, zero-based steps.
    assert [s.step_index for s in steps] == [0, 1, 2]
    assert [s.span_id for s in steps] == ["root", "llm-1", "tool-1"]


def test_linearize_reparents_an_orphan_span_with_a_missing_parent_to_root():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("orphan", parent="does-not-exist", start=100),
    ]

    steps = linearize(spans)

    assert [s.span_id for s in steps] == ["root", "orphan"]
    assert steps[1].depth == 1
    assert orphan_span_ids(spans) == ["orphan"]


def test_linearize_never_drops_a_span_that_is_its_own_parent():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("self-loop", parent="self-loop", start=100),
    ]

    steps = linearize(spans)

    assert "self-loop" in [s.span_id for s in steps]
    assert "self-loop" in orphan_span_ids(spans)


def test_orphan_span_ids_is_empty_when_every_span_resolves_cleanly():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("child", parent="root", start=100),
    ]

    assert orphan_span_ids(spans) == []


def test_linearize_resolves_actor_from_nearest_agent_ancestor():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _agent("sub-agent", "billing_agent", parent="root", start=50),
        _llm("llm-under-sub-agent", parent="sub-agent", start=100),
    ]

    steps = linearize(spans)

    llm_step = next(s for s in steps if s.span_id == "llm-under-sub-agent")
    assert llm_step.actor == "billing_agent"


def test_linearize_signature_distinguishes_tool_call_from_plan_and_answer():
    tool_call = ToolCallRequest(call_id="c1", tool_name="search_orders", arguments={})
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("plan-step", parent="root", start=50),
        _llm("call-step", parent="root", start=100, tool_calls=[tool_call]),
        _tool("tool-step", parent="root", start=150),
        _llm("answer-step", parent="root", start=200),
    ]

    steps = linearize(spans)
    by_id = {s.span_id: s for s in steps}

    assert by_id["plan-step"].signature == "llm:orchestrator:plan"
    assert by_id["call-step"].signature == "llm:orchestrator:call:search_orders"
    assert by_id["answer-step"].signature == "llm:orchestrator:answer"


def test_linearize_tool_signature_reflects_empty_and_error_outcomes():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _tool("empty-result", parent="root", start=50, result=""),
        _tool("error-result", parent="root", start=100, result="", is_error=True),
        _tool("ok-result", parent="root", start=150, result="found it"),
    ]

    steps = linearize(spans)
    by_id = {s.span_id: s for s in steps}

    assert by_id["empty-result"].signature.endswith(":empty")
    assert by_id["error-result"].signature.endswith(":err")
    assert by_id["ok-result"].signature.endswith(":ok")


def test_linearize_tree_path_and_depth_reflect_sibling_position():
    spans = [
        _agent("root", "orchestrator", parent=None, start=0),
        _llm("first", parent="root", start=100),
        _llm("second", parent="root", start=200),
    ]

    steps = linearize(spans)

    assert steps[0].depth == 0 and steps[0].tree_path == "0"
    assert steps[1].depth == 1 and steps[1].tree_path == "0.0"
    assert steps[2].depth == 1 and steps[2].tree_path == "0.1"
