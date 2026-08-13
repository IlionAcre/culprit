"""Fault injection for `synth.py` runs: twenty mutation functions, one per
L1 detector in the plan's catalogue (tool_errors, loops, context, schema,
flow, retrieval, provenance, timing = 20 total), each returning the
ground-truth step index it mutated.

Split from `synth.py` (the run generator) because they are different
concerns that got hard to hold in one file together: `synth.py` only ever
builds a healthy run, this module only ever mutates one. It imports the
handful of `synth.py` internals it needs to fabricate or rewrite steps
(`_make`, `_llm_payload`, `_tool_payload`, `ACTOR_AGENT`, `_trace_from`) plus
the `SynthRun` bundle both modules pass around.

**Injection realism for the four detectors the product exists to catch**
(`empty_tool_result`, `silent_history_truncation`, `parameter_drift`,
`error_swallowed`): the trace must look entirely healthy right where the
damage happens (status OK, no exception), with the corruption only
surfacing later. Those four are tested directly for exactly that property in
`tests/test_synth_inject.py`.

**`inject` takes and returns a `SynthRun`, not a bare `Trace`.** A caller
who mistakenly passes a real ingested `Trace` (from WS-A, say) gets a
`SynthInjectionError` naming the mistake, not a mysterious `KeyError` from a
lookup table keyed by trace id. See `synth.py`'s module docstring for the
registry design this replaced and why.
"""

import copy
import json
import random
import zlib
from collections.abc import Callable

from culprit.schemas import (
    Message,
    Outcome,
    RetrievalPayload,
    RetrievedDoc,
    Span,
    SpanKind,
    SpanStatus,
    Step,
    ToolCallRequest,
)
from culprit.synth import ACTOR_AGENT, SynthRun, _llm_payload, _make, _tool_payload, _trace_from

# One entry per L1 detector in the plan's catalogue, so WS-C's definition of
# done ("every detector fires on its matching injection") has exactly one
# synth case to compile against per detector.
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
    """Raised for an unknown `kind` or a non-`SynthRun` argument, since
    fabricating a ground-truth index for either would silently corrupt every
    downstream assertion that trusts it."""


# --- shared injection helpers -------------------------------------------

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


# --- injectors: tool_errors.py's family ------------------------------------

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


# --- injectors: loops.py's family -------------------------------------------

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


# --- injectors: context.py's family -----------------------------------------

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


# --- injectors: schema.py's family ------------------------------------------

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


# --- injectors: flow.py's family --------------------------------------------

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


# --- injectors: retrieval.py's family ---------------------------------------

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


# --- injectors: provenance.py's and timing.py's families -------------------

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


def inject(run: SynthRun, kind: str, at_step: int) -> tuple[SynthRun, int]:
    """Mutate a copy of `run` (never the original) to trigger the `kind`
    fault near `at_step`, returning the mutated `SynthRun` plus the
    ground-truth step index the mutation actually landed on. `at_step` is a
    hint, not a guarantee: each injector locates the nearest step matching
    its own trigger precondition (see `_nearest`), so the same `at_step` is
    valid for every kind in `INJECTION_KINDS`.
    """
    if not isinstance(run, SynthRun):
        raise SynthInjectionError(
            f"inject() expects a SynthRun from successful_run() or a prior "
            f"inject() call, got {type(run).__name__}"
        )
    if kind not in INJECTION_KINDS:
        raise SynthInjectionError(f"unknown injection kind {kind!r}; available: {sorted(INJECTION_KINDS)}")

    spans, steps = copy.deepcopy(run.spans), copy.deepcopy(run.steps)
    # Deterministic per (trace, kind, at_step): str hashing is randomized
    # per-process in CPython, so a plain hash() would break reproducibility.
    rng = random.Random(zlib.crc32(f"{run.trace.trace_id}|{kind}|{at_step}".encode()))
    ground_truth = _INJECTORS[kind](spans, steps, at_step, rng)

    new_trace_id = f"{run.trace.trace_id}--{kind}@{at_step}"
    for span in spans:
        span.trace_id = new_trace_id
    for step in steps:
        step.trace_id = new_trace_id
    mutated_trace = _trace_from(new_trace_id, spans, steps, Outcome.FAILURE,
                                 metadata={"injected_kind": kind, "injected_step_index": ground_truth})
    return SynthRun(trace=mutated_trace, spans=spans, steps=steps), ground_truth
