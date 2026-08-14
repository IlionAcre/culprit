"""Synthetic "order refund agent" run generator: seeded, reproducible, and
built so successes vary naturally across seeds rather than being identical.

**Why this exists.** WS-C (detectors), WS-D (contrastive), WS-E
(adjudication), and WS-F (clustering) get all their test data from
`synth.py` + `synth_inject.py` rather than waiting on WS-A (ingestion) to
exist.

**Why successes vary.** `successful_run` shuffles independent steps
(retrieval / duplicate-check / verification are mutually independent),
varies retrieval doc counts, includes or omits the two optional steps
(length variation), and varies tool arguments per seed. A reference pool of
identical traces would teach WS-D's profile a single rigid path, and every
honest difference in a real run would then score as a divergence. See
CLAUDE.md's L2 section for why that matters.

**Fault injection lives in `synth_inject.py`.** The run generator and the
twenty fault injectors are different concerns: this module only ever builds
a healthy run, `synth_inject.py` only ever mutates one. Splitting them is
what keeps either file small enough to hold in your head; combined they ran
past 600 lines. `synth_inject.py` imports the handful of internals below it
needs (`SynthRun`, `_make`, `_llm_payload`, `_tool_payload`, `ACTOR_AGENT`,
`_trace_from`) to fabricate or rewrite steps and to build its own mutated
`Trace`.

**RAG context injection (Foundation amendment, post-WS-D handoff).** Every
`_generate` run now folds the retriever step's documents into the request
message of the LLM step immediately following it, via
`_inject_retrieved_context`. Before this, every LLM call was an independent
hand-authored instruction string that never incorporated
`RetrievalPayload.documents`, so a retrieval step had no causal path to
anything downstream: L1's `unused_retrieval` detector could not implement its
specified "<15% token overlap with the next prompt" test (there was nothing
in the next prompt to overlap with, clean or faulty alike), and WS-D's
contrastive layer measured `unused_retrieval` at literally zero recall for
the same reason (see `AI_docs/PHASES.md`'s WS-C open item and
`tests/test_contrast.py`'s `_NO_MECHANISM_KINDS`). Deliberately deferred
until WS-D landed its reference model against the old generator shape, per
the same file's note, so the reference model would not shift underneath a
workstream still learning from it.

**Why `successful_run` returns a `SynthRun` bundle, not a bare `Trace`.**
`Trace` deliberately carries no spans or steps (see `schemas.py`), but every
real consumer - `run_detectors(trace, steps, spans_by_id)`,
`contrast(trace, steps, ...)` - needs all three. An earlier version of this
module solved that with a module-level `trace_id -> (spans, steps)`
registry populated as a side effect of generation. That had four problems:
it never got cleaned up (a single test session accumulates thousands of
entries), it made `successful_run` look like a pure constructor while
secretly mutating module state, it let state leak between tests sharing a
process, and worst, it turned any `Trace` that did not come from this
module (a real one from WS-A, say) into a bare `KeyError` on a lookup call
with nothing explaining why. Returning `SynthRun(trace, spans, steps)` by
value has none of those problems: nothing to clean up, no side effects, no
cross-test state, and passing the wrong kind of object to `synth_inject.py`
fails with a message that names the mistake instead of an opaque `KeyError`.
"""

import json
import random
from dataclasses import dataclass

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


@dataclass
class SynthRun:
    """A generated or mutated run: the canonical `Trace` plus its full span
    and step sequence, bundled together because `Trace` alone is never
    enough to run a detector or the contrastive layer against. See the
    module docstring for why this replaced a module-level registry."""

    trace: Trace
    spans: list[Span]
    steps: list[Step]


# --- payload builders (also used by synth_inject.py) -----------------------

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


def _render_retrieved_context(docs: list[RetrievedDoc]) -> str:
    """The block folded into the next prompt. Built from `content_preview`,
    the only document text `RetrievedDoc` carries (see `schemas.py`'s Task 2
    amendment: length plus a bounded preview, not full text) - a real RAG
    prompt would use the full chunk, but the preview is what this fixture
    has, and it is enough for a token-overlap test to have something real to
    measure against."""
    excerpts = "\n".join(f"- {d.doc_id}: {d.content_preview}" for d in docs)
    return f"Retrieved context:\n{excerpts}"


def _inject_retrieved_context(spans: list[Span], steps: list[Step]) -> None:
    """Real RAG: the step after a retrieval carries what was retrieved. Folds
    the retriever step's documents into the immediately following LLM step's
    request message, mutating `spans` in place. `_generate`'s retrieval
    block is always immediately followed by exactly one LLM step (the next
    shuffled block's call, or `call_process_refund` if retrieval landed
    last), so one linear scan covers every seed; a step with no LLM
    successor (there is currently none) is simply left alone rather than
    raising, so this stays safe if the run shape ever changes.

    Deliberately runs inside `_generate`, before `synth_inject.py` ever sees
    the run: the 20 injectors in that module mutate a deep copy of an
    already-healthy run, so `unused_retrieval`/`low_score_retrieval`/
    `goal_token_drift` correctly rewrite *only* the retrieved documents,
    leaving the already-baked next prompt untouched - exactly the "documents
    retrieved don't match what the prompt actually used" shape those faults
    are meant to represent.
    """
    for i in range(len(steps) - 1):
        if steps[i].kind != SpanKind.RETRIEVER or steps[i + 1].kind != SpanKind.LLM:
            continue
        docs = spans[i].payload.documents
        if not docs:
            continue
        next_payload = spans[i + 1].payload
        if not next_payload.request_messages:
            continue
        original = next_payload.request_messages[0]
        context = _render_retrieved_context(docs)
        next_payload.request_messages[0] = original.model_copy(
            update={"content": f"{context}\n\n{original.content or ''}"}
        )


# --- span/step construction (also used by synth_inject.py) -----------------

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
    lets every injector in `synth_inject.py` index both lists with one `i`.

    Steps 0-3 (delegate, plan, decide-to-search, search_orders) are fixed
    across every seed so `inject(..., at_step=3)` reliably lands on a TOOL
    step regardless of seed. Everything from step 4 onward varies: retrieval
    / duplicate-check / verification are mutually independent and shuffled,
    doc counts and arguments are randomized, and the two optional blocks
    make length vary by seed.
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

    _inject_retrieved_context(spans, steps)

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


def successful_run(seed: int) -> SynthRun:
    """A seeded, reproducible "order refund agent" success, bundled with its
    full span and step sequence (`.trace`, `.spans`, `.steps`)."""
    trace_id, spans, steps = _generate(seed)
    trace = _trace_from(trace_id, spans, steps, Outcome.SUCCESS)
    return SynthRun(trace=trace, spans=spans, steps=steps)
