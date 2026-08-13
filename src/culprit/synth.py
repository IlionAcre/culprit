"""Synthetic trace generator: an "order refund agent" run, seeded and
reproducible, plus fault injection with recoverable ground truth.

**This is the linchpin module of Phase 0.** WS-C (detectors), WS-D
(contrastive), WS-E (adjudication), and WS-F (clustering) get all their test
data from here rather than waiting on WS-A (ingestion) to exist. Two
properties carry that weight:

1. **Successes vary naturally.** `successful_run` reorders independent steps
   (retrieval / duplicate-check / verification are mutually independent and
   are shuffled), varies retrieval doc counts, includes or omits optional
   steps (length variation), and varies tool arguments per seed. A reference
   pool of identical traces would teach WS-D's profile a single rigid path,
   and every honest difference in a real run would then score as a
   divergence. See CLAUDE.md's L2 section for why that matters.

2. **`inject` returns ground truth.** WS-C and WS-D both assert against the
   returned step index; without it every downstream test would hardcode a
   magic number that silently rots the first time generation changes.

Deliberately no logging (matches `llm.py`'s precedent): this module is a pure,
deterministic generator with no I/O and no partial-failure modes worth a log
line: the one error path (`SynthInjectionError`) is a programmer error a
caller sees immediately as a raised exception.

**Design note on the module-level registry.** `successful_run`/`inject` are
specified to return only a `Trace`, but `Trace` deliberately carries no steps
or spans (see `schemas.py`), and `run_detectors(trace, steps, spans_by_id)` /
`contrast(trace, steps, ...)` both need them. Rather than smuggling them
through `Trace.metadata` (fragile, and pollutes the one field linearize.py
uses for orphan bookkeeping), this module keeps a private `trace_id ->
(spans, steps)` registry and exposes `spans_of`/`steps_of` accessors. A
downstream test does `trace = successful_run(seed); steps = steps_of(trace)`.
"""

import copy
import json
import random
import zlib
from collections.abc import Callable

from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    Message,
    Outcome,
    RetrievalPayload,
    RetrievedDoc,
    Span,
    SpanKind,
    SpanStatus,
    Step,
    Trace,
    ToolCallRequest,
    ToolPayload,
)

ACTOR_ORCH = "orchestrator"
ACTOR_AGENT = "refund_agent"
_START_NS = 1_755_000_000_000_000_000

# One entry per L1 detector in the plan's catalogue (tool_errors, loops,
# context, schema, flow, retrieval, provenance, timing = 20 total), so WS-C's
# definition of done ("every detector fires on its matching injection") has
# exactly one synth case to compile against per detector.
INJECTION_KINDS = frozenset(
    {
        "tool_error",
        "empty_tool_result",
        "error_swallowed",
        "oscillation",
        "repeated_identical_action",
        "retry_storm",
        "context_overflow",
        "silent_history_truncation",
        "step_budget_exhausted",
        "output_schema_violation",
        "tool_arg_malformed",
        "hallucinated_tool",
        "premature_termination",
        "missing_verification",
        "duplicate_delegation",
        "unused_retrieval",
        "low_score_retrieval",
        "goal_token_drift",
        "parameter_drift",
        "stall_timeout",
    }
)


class SynthInjectionError(Exception):
    """Raised for an unknown `kind` or a trace this module did not generate,
    since fabricating a ground-truth index for either would silently corrupt
    every downstream assertion that trusts it."""


_REGISTRY: dict[str, tuple[list[Span], list[Step]]] = {}


# --- payload builders -------------------------------------------------

def _llm_payload(
    text: str, *, tool_calls: list[ToolCallRequest] | None = None, prompt_tokens: int = 600
) -> LlmPayload:
    return LlmPayload(
        provider="synth",
        model="synth-llm-1",
        request_messages=[Message(role="user", content=text)],
        response_messages=[Message(role="assistant", content=None if tool_calls else text)],
        tool_calls=tool_calls or [],
        finish_reason="tool_calls" if tool_calls else "stop",
        prompt_tokens=prompt_tokens,
        completion_tokens=80,
        total_tokens=prompt_tokens + 80,
        max_tokens=8000,
        temperature=0.2,
    )


def _tool_payload(
    tool_name: str, args: dict, result_text: str, *, is_error: bool = False,
    error_message: str | None = None,
) -> ToolPayload:
    return ToolPayload(
        tool_name=tool_name,
        call_id=f"call-{tool_name}",
        arguments_json=json.dumps(args),
        arguments=args,
        result_text=result_text,
        result_len=len(result_text),
        is_error=is_error,
        error_message=error_message,
    )


def _retrieval_payload(query: str, doc_count: int, rng: random.Random) -> RetrievalPayload:
    docs = [
        RetrievedDoc(
            doc_id=f"doc-{n}",
            content_len=rng.randint(200, 900),
            content_preview=f"Refund policy excerpt {n}: items may be returned within 30 days...",
            score=round(rng.uniform(0.55, 0.95), 3),
        )
        for n in range(doc_count)
    ]
    return RetrievalPayload(query=query, documents=docs, top_k=doc_count)


# --- span/step construction --------------------------------------------

def _base_ts(idx: int) -> tuple[int, int]:
    start = _START_NS + idx * 2_000_000_000
    return start, start + 1_500_000_000


def _make(
    trace_id: str, idx: int, kind: SpanKind, actor: str, name: str, payload, signature: str,
    summary: str, *, status: SpanStatus = SpanStatus.OK, status_message: str | None = None,
    parent: str | None = None,
) -> tuple[Span, Step]:
    span_id = f"{trace_id}-sp{idx}"
    start, end = _base_ts(idx)
    span = Span(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent, name=name, kind=kind,
        status=status, status_message=status_message, start_ns=start, end_ns=end,
        vocabulary="synth", attributes={}, payload=payload,
    )
    step = Step(
        trace_id=trace_id, step_index=idx, span_id=span_id, kind=kind, actor=actor,
        depth=0 if parent is None else 1, tree_path="0" if parent is None else f"0.{idx}",
        signature=signature, summary=summary, start_ns=start, end_ns=end,
        duration_ms=(end - start) / 1_000_000,
    )
    return span, step


def _generate(seed: int) -> tuple[str, list[Span], list[Step]]:
    """Build one refund-agent run. `spans[i]` and `steps[i]` always describe
    the same event (synth never collapses framework spans), which is what
    lets every injector below index both lists with one `i`.

    Steps 0-3 (delegate, plan, decide-to-search, search_orders) are fixed
    across every seed so `inject(..., at_step=3)` reliably lands on a TOOL
    step regardless of seed, per the Task 7 test. Everything from step 4
    onward varies: retrieval / duplicate-check / verification are mutually
    independent and shuffled, doc counts and arguments are randomized, and
    the two optional blocks make length vary by seed.
    """
    rng = random.Random(seed)
    trace_id = f"synth-{seed}"
    order_id = f"ORD-{rng.randint(10000, 99999)}"
    goal = f"Refund order {order_id}: customer says the package never arrived"

    spans: list[Span] = []
    steps: list[Step] = []

    def add(kind, actor, name, payload, signature, summary, **kw):
        idx = len(steps)
        parent = spans[0].span_id if spans else None
        span, step = _make(trace_id, idx, kind, actor, name, payload, signature, summary, parent=parent, **kw)
        spans.append(span)
        steps.append(step)

    add(SpanKind.AGENT, ACTOR_ORCH, "invoke_agent",
        AgentPayload(agent_name=ACTOR_ORCH, role="router", input_text=goal, output_text="",
                     delegated_to=[ACTOR_AGENT]),
        f"agent:{ACTOR_ORCH}:invoke", "Orchestrator delegates the refund request")
    add(SpanKind.LLM, ACTOR_AGENT, "plan", _llm_payload(f"Goal: {goal}. Plan the steps."),
        f"llm:{ACTOR_AGENT}:plan", "Agent plans how to resolve the refund")
    add(SpanKind.LLM, ACTOR_AGENT, "call_search_orders",
        _llm_payload("Look up the order.", tool_calls=[
            ToolCallRequest(call_id="call-search_orders", tool_name="search_orders",
                             arguments={"order_id": order_id})]),
        f"llm:{ACTOR_AGENT}:call:search_orders", "Agent decides to search for the order")
    add(SpanKind.TOOL, ACTOR_AGENT, "search_orders",
        _tool_payload("search_orders", {"order_id": order_id},
                      f"Order {order_id}: status=shipped, amount=$42.00"),
        f"tool:{ACTOR_AGENT}:search_orders:ok", "search_orders returns order details")

    doc_count = rng.randint(1, 4)
    blocks = [("retrieval", doc_count)]
    if rng.random() < 0.5:
        blocks.append(("dup_check", None))
    if rng.random() < 0.5:
        blocks.append(("verify", None))
    rng.shuffle(blocks)

    for name, arg in blocks:
        if name == "retrieval":
            add(SpanKind.RETRIEVER, ACTOR_AGENT, "retrieve_policy",
                _retrieval_payload("refund policy", arg, rng),
                f"retr:{ACTOR_AGENT}:policy:{'hit' if arg else 'miss'}",
                f"Retrieves {arg} refund-policy documents")
        elif name == "dup_check":
            add(SpanKind.LLM, ACTOR_AGENT, "call_check_duplicate_refund",
                _llm_payload("Check whether this refund was already issued.", tool_calls=[
                    ToolCallRequest(call_id="call-check_duplicate_refund",
                                     tool_name="check_duplicate_refund", arguments={"order_id": order_id})]),
                f"llm:{ACTOR_AGENT}:call:check_duplicate_refund", "Agent checks for a duplicate refund")
            add(SpanKind.TOOL, ACTOR_AGENT, "check_duplicate_refund",
                _tool_payload("check_duplicate_refund", {"order_id": order_id}, "No prior refund found"),
                f"tool:{ACTOR_AGENT}:check_duplicate_refund:ok", "No duplicate refund on record")
        else:
            add(SpanKind.LLM, ACTOR_AGENT, "call_verify_eligibility",
                _llm_payload("Verify the order is eligible for a refund.", tool_calls=[
                    ToolCallRequest(call_id="call-verify_eligibility", tool_name="verify_eligibility",
                                     arguments={"order_id": order_id})]),
                f"llm:{ACTOR_AGENT}:call:verify_eligibility", "Agent verifies refund eligibility")
            add(SpanKind.TOOL, ACTOR_AGENT, "verify_eligibility",
                _tool_payload("verify_eligibility", {"order_id": order_id}, "eligible=true"),
                f"tool:{ACTOR_AGENT}:verify_eligibility:ok", "Order is eligible for refund")

    add(SpanKind.LLM, ACTOR_AGENT, "call_process_refund",
        _llm_payload("Process the refund.", tool_calls=[
            ToolCallRequest(call_id="call-process_refund", tool_name="process_refund",
                             arguments={"order_id": order_id, "amount": 42.0})]),
        f"llm:{ACTOR_AGENT}:call:process_refund", "Agent decides to process the refund")
    add(SpanKind.TOOL, ACTOR_AGENT, "process_refund",
        _tool_payload("process_refund", {"order_id": order_id, "amount": 42.0},
                      f"Refund issued for {order_id}"),
        f"tool:{ACTOR_AGENT}:process_refund:ok", "Refund is processed")
    add(SpanKind.LLM, ACTOR_AGENT, "answer",
        _llm_payload(f"Your refund for {order_id} has been processed.", prompt_tokens=900),
        f"llm:{ACTOR_AGENT}:answer", "Agent reports the refund was processed")

    return trace_id, spans, steps


def _trace_from(
    trace_id: str, spans: list[Span], steps: list[Step], outcome: Outcome,
    metadata: dict | None = None,
) -> Trace:
    goal = next((s.payload.input_text for s in spans if isinstance(s.payload, AgentPayload)), None)
    return Trace(
        trace_id=trace_id, source="synth", outcome=outcome, agent_key="synth:refund_agent",
        task_key=f"synth-task:{trace_id}", task_goal=goal, framework="synth",
        root_span_id=spans[0].span_id, span_count=len(spans), step_count=len(steps),
        metadata=metadata or {},
    )


def successful_run(seed: int) -> Trace:
    """A seeded, reproducible "order refund agent" success. Call `steps_of`/
    `spans_of` on the result to get the full step and span sequence."""
    trace_id, spans, steps = _generate(seed)
    trace = _trace_from(trace_id, spans, steps, Outcome.SUCCESS)
    _REGISTRY[trace_id] = (spans, steps)
    return trace


def spans_of(trace: Trace) -> list[Span]:
    return list(_REGISTRY[trace.trace_id][0])


def steps_of(trace: Trace) -> list[Step]:
    return list(_REGISTRY[trace.trace_id][1])


# --- injection helpers ---------------------------------------------------

def _is_tool(s: Step) -> bool:
    return s.kind == SpanKind.TOOL


def _is_llm(s: Step) -> bool:
    return s.kind == SpanKind.LLM


def _is_retriever(s: Step) -> bool:
    return s.kind == SpanKind.RETRIEVER


def _is_agent(s: Step) -> bool:
    return s.kind == SpanKind.AGENT


def _nearest(steps: list[Step], at_step: int, predicate: Callable[[Step], bool]) -> int:
    """The step nearest `at_step` (by index distance) satisfying `predicate`.
    Every injector's trigger condition needs a specific kind of step (a TOOL
    step for `tool_error`, an LLM step for `context_overflow`, ...), but
    `at_step` is caller-chosen and may not land on one directly (WS-C's
    reachability test calls every kind with the same `at_step=2`). Falls back
    to `at_step` itself if nothing matches, so a caller always gets an index
    back rather than an exception."""
    at_step = max(0, min(at_step, len(steps) - 1))
    if predicate(steps[at_step]):
        return at_step
    for delta in range(1, len(steps)):
        for cand in (at_step - delta, at_step + delta):
            if 0 <= cand < len(steps) and predicate(steps[cand]):
                return cand
    return at_step


def _insert(spans: list[Span], steps: list[Step], at_idx: int, pairs: list[tuple[Span, Step]]) -> int:
    for offset, (span, step) in enumerate(pairs):
        spans.insert(at_idx + offset, span)
        steps.insert(at_idx + offset, step)
    for k in range(len(steps)):
        steps[k].step_index = k
    return at_idx


def _repeat_tool(spans, steps, i, count, *, same_result, label) -> int:
    """Shared by `repeated_identical_action` and `retry_storm`: both insert
    `count` more calls to the same tool with the same arguments right after
    step `i`; they differ only in whether the result is identical too (zero
    progress, the `retry_storm` trigger) or merely repeated (the weaker
    `repeated_identical_action` trigger)."""
    src = spans[i].payload
    tool, args = src.tool_name, dict(src.arguments)
    actor, trace_id = steps[i].actor, steps[i].trace_id
    pairs = []
    for n in range(count):
        result = src.result_text if same_result else f"Order status=shipped (lookup {n + 2})"
        span, step = _make(trace_id, i + n + 1, SpanKind.TOOL, actor, tool,
                            _tool_payload(tool, args, result),
                            f"tool:{actor}:{tool}:ok", f"{label} {tool}", parent=spans[0].span_id)
        span.span_id = step.span_id = f"{trace_id}-{label}{n}"
        pairs.append((span, step))
    return _insert(spans, steps, i + 1, pairs)


def _inj_tool_error(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    spans[i].status, spans[i].status_message = SpanStatus.ERROR, "upstream tool error"
    p = spans[i].payload
    p.is_error, p.error_message = True, "ConnectionError: upstream service unavailable"
    p.result_text = f"Error: {p.error_message}"
    p.result_len = len(p.result_text)
    steps[i].signature = f"tool:{steps[i].actor}:{p.tool_name}:err"
    steps[i].summary = f"{p.tool_name} raised an error"
    return i


def _inj_empty_tool_result(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    p = spans[i].payload
    p.result_text = rng.choice(["", "[]", "{}", "null", "No results found"])
    p.result_len, p.is_error = len(p.result_text), False
    spans[i].status = SpanStatus.OK
    steps[i].signature = f"tool:{steps[i].actor}:{p.tool_name}:empty"
    steps[i].summary = f"{p.tool_name} returned nothing"
    return i


def _inj_error_swallowed(spans, steps, at_step, rng) -> int:
    i = _inj_empty_tool_result(spans, steps, at_step, rng)
    for j in (i + 1, i + 2):
        if j < len(steps) and _is_llm(steps[j]):
            p = spans[j].payload
            p.response_messages = [Message(
                role="assistant",
                content="Done, everything checked out fine and the refund was issued successfully.")]
            p.finish_reason = "stop"
            steps[j].summary = "Agent reports success despite the empty tool result"
            break
    return i


def _inj_oscillation(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    actor, trace_id = steps[i].actor, steps[i].trace_id
    pairs = []
    for n in range(4):
        tool = "search_orders" if n % 2 == 0 else "check_duplicate_refund"
        span, step = _make(trace_id, i + n + 1, SpanKind.TOOL, actor, tool,
                            _tool_payload(tool, {"order_id": "ORD-OSC"}, "no progress"),
                            f"tool:{actor}:{tool}:ok", f"Repeats {tool} (oscillation)", parent=spans[0].span_id)
        span.span_id = step.span_id = f"{trace_id}-osc{n}"
        pairs.append((span, step))
    return _insert(spans, steps, i, pairs)


def _inj_repeated_identical_action(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    _repeat_tool(spans, steps, i, 2, same_result=False, label="rep")
    return i


def _inj_retry_storm(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    _repeat_tool(spans, steps, i, 3, same_result=True, label="retry")
    return i


def _inj_context_overflow(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    p = spans[i].payload
    p.max_tokens, p.prompt_tokens, p.completion_tokens = 8000, 7600, 200
    p.total_tokens = p.prompt_tokens + p.completion_tokens
    steps[i].summary = "Prompt is near the model's context window limit"
    return i


def _inj_silent_history_truncation(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    actor = steps[i].actor
    j = next((k for k in range(i + 1, len(steps)) if _is_llm(steps[k]) and steps[k].actor == actor), i)
    spans[i].payload.prompt_tokens = 6000
    spans[j].payload.prompt_tokens = 3800
    spans[j].payload.request_messages = spans[i].payload.request_messages + [
        Message(role="assistant", content="continuing")]
    steps[j].summary = "prompt_tokens drops sharply though history kept growing (silent truncation)"
    return j


def _inj_step_budget_exhausted(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    actor, trace_id = steps[i].actor, steps[i].trace_id
    del spans[i + 1:]
    del steps[i + 1:]
    for n in range(40):
        span, step = _make(trace_id, i + n + 1, SpanKind.LLM, actor, "plan",
                            _llm_payload("Still figuring out the next step..."),
                            f"llm:{actor}:plan", "Agent keeps re-planning without resolving",
                            parent=spans[0].span_id)
        span.span_id = step.span_id = f"{trace_id}-budget{n}"
        step.step_index = len(steps)
        spans.append(span)
        steps.append(step)
    return i


def _inj_output_schema_violation(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    p = spans[i].payload
    p.response_format_schema = {"type": "object", "required": ["refund_id"],
                                 "properties": {"refund_id": {"type": "string"}}}
    p.response_messages = [Message(role="assistant", content="Sure thing, all done!")]
    steps[i].summary = "Response does not match the declared output schema"
    return i


def _inj_tool_arg_malformed(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    p = spans[i].payload
    p.arguments_json, p.arguments = '{"order_id": "ORD-1234', {}
    steps[i].summary = "Tool arguments failed to parse as JSON"
    return i


def _inj_hallucinated_tool(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    fake = "reverse_time_machine"
    spans[i].payload.tool_calls = [ToolCallRequest(call_id="call-fake", tool_name=fake, arguments={})]
    steps[i].signature = f"llm:{steps[i].actor}:call:{fake}"
    steps[i].summary = f"Agent calls a tool that was never registered: {fake}"
    return i


def _inj_premature_termination(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_llm)
    actor, trace_id = steps[i].actor, steps[i].trace_id
    del spans[i + 1:]
    del steps[i + 1:]
    span, step = _make(trace_id, i + 1, SpanKind.LLM, actor, "answer", _llm_payload("Done!"),
                        f"llm:{actor}:answer", "Agent terminates without completing the task",
                        parent=spans[0].span_id)
    span.span_id = step.span_id = f"{trace_id}-term"
    spans.append(span)
    steps.append(step)
    return i


def _inj_missing_verification(spans, steps, at_step, rng) -> int:
    keep = [k for k, s in enumerate(steps) if "verify_eligibility" not in s.signature]
    spans[:], steps[:] = [spans[k] for k in keep], [steps[k] for k in keep]
    for k, step in enumerate(steps):
        step.step_index = k
    j = next(k for k, s in enumerate(steps) if s.kind == SpanKind.TOOL and "process_refund" in s.signature)
    steps[j].summary = "Refund processed without a verification step"
    return j


def _inj_duplicate_delegation(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_agent)
    spans[i].payload.delegated_to = [ACTOR_AGENT, ACTOR_AGENT]
    steps[i].summary = "Orchestrator delegates to the same agent twice"
    return i


def _inj_unused_retrieval(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_retriever)
    docs = [RetrievedDoc(doc_id="doc-unrelated", content_len=500,
                          content_preview="Unrelated marketing copy about a different product line.",
                          score=0.6)]
    spans[i].payload = RetrievalPayload(query=spans[i].payload.query, documents=docs, top_k=1)
    steps[i].summary = "Retrieved documents are never referenced afterward"
    return i


def _inj_low_score_retrieval(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_retriever)
    docs = [d.model_copy(update={"score": 0.12}) for d in spans[i].payload.documents] or [
        RetrievedDoc(doc_id="doc-weak", content_len=100, content_preview="tangential", score=0.1)]
    spans[i].payload = RetrievalPayload(query=spans[i].payload.query, documents=docs, top_k=len(docs))
    steps[i].summary = "All retrieved documents score far below the relevance threshold"
    return i


def _inj_goal_token_drift(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_retriever)
    spans[i].payload = RetrievalPayload(query="unrelated topic about quarterly sales figures",
                                         documents=spans[i].payload.documents, top_k=spans[i].payload.top_k)
    steps[i].summary = "Retrieval query has drifted away from the task goal"
    return i


def _inj_parameter_drift(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    fabricated = "ORD-99999-ZZZZ"
    p = spans[i].payload
    p.arguments = {**p.arguments, "order_id": fabricated}
    p.arguments_json = json.dumps(p.arguments)
    steps[i].summary = f"Tool argument references {fabricated}, absent from the goal or any prior step"
    return i


def _inj_stall_timeout(spans, steps, at_step, rng) -> int:
    i = _nearest(steps, at_step, _is_tool)
    spans[i].end_ns = spans[i].start_ns + 45_000_000_000
    steps[i].end_ns = spans[i].end_ns
    steps[i].duration_ms = (steps[i].end_ns - steps[i].start_ns) / 1_000_000
    steps[i].summary = f"{spans[i].payload.tool_name} stalls far longer than similar calls"
    return i


_INJECTORS: dict[str, Callable[[list[Span], list[Step], int, random.Random], int]] = {
    "tool_error": _inj_tool_error,
    "empty_tool_result": _inj_empty_tool_result,
    "error_swallowed": _inj_error_swallowed,
    "oscillation": _inj_oscillation,
    "repeated_identical_action": _inj_repeated_identical_action,
    "retry_storm": _inj_retry_storm,
    "context_overflow": _inj_context_overflow,
    "silent_history_truncation": _inj_silent_history_truncation,
    "step_budget_exhausted": _inj_step_budget_exhausted,
    "output_schema_violation": _inj_output_schema_violation,
    "tool_arg_malformed": _inj_tool_arg_malformed,
    "hallucinated_tool": _inj_hallucinated_tool,
    "premature_termination": _inj_premature_termination,
    "missing_verification": _inj_missing_verification,
    "duplicate_delegation": _inj_duplicate_delegation,
    "unused_retrieval": _inj_unused_retrieval,
    "low_score_retrieval": _inj_low_score_retrieval,
    "goal_token_drift": _inj_goal_token_drift,
    "parameter_drift": _inj_parameter_drift,
    "stall_timeout": _inj_stall_timeout,
}


def inject(trace: Trace, kind: str, at_step: int) -> tuple[Trace, int]:
    """Mutate a copy of `trace` (never the original) to trigger the `kind`
    fault near `at_step`, returning the mutated `Trace` plus the ground-truth
    step index the mutation actually landed on. `at_step` is a hint, not a
    guarantee: each injector locates the nearest step matching its own
    trigger precondition (see `_nearest`), so the same `at_step` is valid for
    every kind in `INJECTION_KINDS`.
    """
    if kind not in INJECTION_KINDS:
        raise SynthInjectionError(f"unknown injection kind {kind!r}; available: {sorted(INJECTION_KINDS)}")
    if trace.trace_id not in _REGISTRY:
        raise SynthInjectionError(
            f"trace {trace.trace_id!r} was not produced by this module; "
            "generate it with successful_run() first"
        )
    spans, steps = _REGISTRY[trace.trace_id]
    spans, steps = copy.deepcopy(spans), copy.deepcopy(steps)
    # Deterministic per (trace, kind, at_step): str hashing is randomized
    # per-process in CPython, so a plain hash() would break reproducibility.
    rng = random.Random(zlib.crc32(f"{trace.trace_id}|{kind}|{at_step}".encode()))
    ground_truth = _INJECTORS[kind](spans, steps, at_step, rng)

    new_trace_id = f"{trace.trace_id}--{kind}@{at_step}"
    for span in spans:
        span.trace_id = new_trace_id
    for step in steps:
        step.trace_id = new_trace_id
    mutated = _trace_from(new_trace_id, spans, steps, Outcome.FAILURE,
                           metadata={"injected_kind": kind, "injected_step_index": ground_truth})
    _REGISTRY[new_trace_id] = (spans, steps)
    return mutated, ground_truth
