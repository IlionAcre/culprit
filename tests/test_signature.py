from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Message,
    RetrievalPayload,
    RetrievedDoc,
    Span,
    SpanKind,
    SpanStatus,
    Step,
    ToolCallRequest,
    ToolPayload,
)
from culprit.signature import signature_of, sim


def _step(step_index, kind, actor, signature, depth=0) -> Step:
    return Step(
        trace_id="t", step_index=step_index, span_id=f"sp{step_index}", kind=kind,
        actor=actor, depth=depth, tree_path="0", signature=signature, summary="",
        start_ns=0, end_ns=1, duration_ms=1.0,
    )


def _span(step: Step, payload, *, status=SpanStatus.OK) -> Span:
    return Span(
        trace_id="t", span_id=step.span_id, parent_span_id=None, name="x", kind=step.kind,
        status=status, status_message=None, start_ns=0, end_ns=1, vocabulary="synth",
        attributes={}, payload=payload,
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


# --- span-aware features, added after the contract fix (contrast() now
# receives spans_by_id) -----------------------------------------------------

def test_arg_value_jaccard_is_now_real_not_a_fixed_neutral():
    same_args = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    same_args_2 = _step(1, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    diff_args = _step(2, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")

    payload = ToolPayload(tool_name="search_orders", call_id="c1", arguments_json="{}",
                           arguments={"order_id": "A"}, result_text="ok", result_len=2, is_error=False)
    same_payload = ToolPayload(tool_name="search_orders", call_id="c2", arguments_json="{}",
                                arguments={"order_id": "A"}, result_text="ok", result_len=2, is_error=False)
    diff_payload = ToolPayload(tool_name="search_orders", call_id="c3", arguments_json="{}",
                                arguments={"order_id": "Z"}, result_text="ok", result_len=2, is_error=False)

    span_a = _span(same_args, payload)
    span_b = _span(same_args_2, same_payload)
    span_c = _span(diff_args, diff_payload)

    assert sim(same_args, same_args_2, span_a, span_b) > sim(same_args, diff_args, span_a, span_c)


def test_retrieval_docid_overlap_is_now_real():
    a = _step(0, SpanKind.RETRIEVER, "agent", "retr:agent:w2:hit")
    b = _step(1, SpanKind.RETRIEVER, "agent", "retr:agent:w2:hit")
    c = _step(2, SpanKind.RETRIEVER, "agent", "retr:agent:w2:hit")

    docs_shared = [RetrievedDoc(doc_id="d1", content_len=10, content_preview="x", score=0.8)]
    docs_disjoint = [RetrievedDoc(doc_id="d2", content_len=10, content_preview="y", score=0.8)]
    span_a = _span(a, RetrievalPayload(query="refund policy", documents=docs_shared, top_k=1))
    span_b = _span(b, RetrievalPayload(query="refund policy", documents=docs_shared, top_k=1))
    span_c = _span(c, RetrievalPayload(query="refund policy", documents=docs_disjoint, top_k=1))

    assert sim(a, b, span_a, span_b) > sim(a, c, span_a, span_c)


def test_missing_span_on_one_side_is_a_real_disagreement_not_neutral():
    step = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    payload = ToolPayload(tool_name="search_orders", call_id="c1", arguments_json="{}",
                           arguments={"order_id": "A"}, result_text="ok", result_len=2, is_error=False)
    with_span = sim(step, step, _span(step, payload), _span(step, payload))
    one_missing = sim(step, step, _span(step, payload), None)

    assert one_missing < with_span


def test_signature_of_appends_argsmalformed_flag_for_unparseable_tool_arguments():
    step = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    payload = ToolPayload(tool_name="search_orders", call_id="c1", arguments_json='{"order_id": "ORD-1',
                           arguments={}, result_text="ok", result_len=2, is_error=False)

    enriched = signature_of(step, _span(step, payload))

    assert enriched == "tool:agent:search_orders:ok|argsmalformed"


def test_signature_of_appends_neartokenlimit_flag():
    step = _step(0, SpanKind.LLM, "agent", "llm:agent:answer")
    payload = LlmPayload(provider="p", model="m", request_messages=[], response_messages=[],
                          total_tokens=7800, max_tokens=8000)

    enriched = signature_of(step, _span(step, payload))

    assert "neartokenlimit" in enriched


def test_signature_of_appends_delegates_flag_for_duplicate_delegation():
    step = _step(0, SpanKind.AGENT, "orch", "agent:orch:invoke")
    payload = AgentPayload(agent_name="orch", role="router", input_text="x", output_text="",
                            delegated_to=["refund_agent", "refund_agent"])

    enriched = signature_of(step, _span(step, payload))

    assert enriched == "agent:orch:invoke|delegates2"


def test_signature_of_appends_lowscore_flag_for_weak_retrieval():
    step = _step(0, SpanKind.RETRIEVER, "agent", "retr:agent:w2:hit")
    docs = [RetrievedDoc(doc_id="d1", content_len=10, content_preview="x", score=0.1)]
    payload = RetrievalPayload(query="refund policy", documents=docs, top_k=1)

    enriched = signature_of(step, _span(step, payload))

    assert "lowscore" in enriched


def test_signature_of_appends_querydrift_flag_when_query_word_count_disagrees_with_signature():
    """The step's `signature` bakes in `w{n}` from whatever query existed
    when linearize.py ran; if the span's current query has since diverged
    (a mutation that never re-linearized), that mismatch is itself a
    signal - this is how goal_token_drift becomes visible."""
    step = _step(0, SpanKind.RETRIEVER, "agent", "retr:agent:w2:hit")
    docs = [RetrievedDoc(doc_id="d1", content_len=10, content_preview="x", score=0.8)]
    payload = RetrievalPayload(query="unrelated topic about quarterly sales figures", documents=docs, top_k=1)

    enriched = signature_of(step, _span(step, payload))

    assert "querydrift" in enriched


def test_signature_of_appends_schemaviolation_flag_for_unparseable_json_response():
    step = _step(0, SpanKind.LLM, "agent", "llm:agent:answer")
    payload = LlmPayload(
        provider="p", model="m", request_messages=[],
        response_messages=[Message(role="assistant", content="Sure thing, all done!")],
        response_format_schema={"type": "object", "required": ["refund_id"]},
    )

    enriched = signature_of(step, _span(step, payload))

    assert "schemaviolation" in enriched


def test_signature_of_adds_no_flags_for_a_clean_payload():
    step = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")
    payload = ToolPayload(tool_name="search_orders", call_id="c1", arguments_json='{"order_id": "A"}',
                           arguments={"order_id": "A"}, result_text="ok", result_len=2, is_error=False)

    assert signature_of(step, _span(step, payload)) == step.signature


def test_signature_of_falls_back_to_the_bare_signature_with_no_span():
    step = _step(0, SpanKind.TOOL, "agent", "tool:agent:search_orders:ok")

    assert signature_of(step, None) == step.signature
    assert signature_of(step) == step.signature
