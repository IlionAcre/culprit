"""Span tree -> linear `Step` sequence. Three load-bearing rules (CLAUDE.md
"Linearization" section): (1) DFS pre-order, siblings sorted by `(start_ns,
span_id)` - the span_id tie-break matters because OTel exporters routinely
emit siblings with identical millisecond-truncated start times, and without
it the same trace linearizes differently on two runs, poisoning L2
alignment; (2) only LLM/TOOL/RETRIEVER/AGENT spans become steps, everything
else folds into the *next* semantic step's `collapsed_span_ids` (or the
last step's, if none follows), so framework noise never dominates "step 47";
(3) orphan spans (broken/missing parent references - sampled and
partially-exported traces are common) re-parent to root, never dropped.

`actor` is resolved here, not carried on `Span`: the nearest AGENT
ancestor's `agent_name`, or the span's own name at the root if the root
itself is not an AGENT span. `signature` follows the coarse alignment-token
format `synth.py` already establishes as canonical
(`llm:{actor}:call:{tool}` / `:plan` / `:answer`,
`tool:{actor}:{tool}:{ok|err|empty}`, `retr:{actor}:{shape}:{hit|miss}`,
`agent:{actor}:invoke`).

`orphan_span_ids()` is exposed separately since `linearize()`'s own contract
is `list[Span] -> list[Step]`; a caller assembling `Trace` stashes the
result into `metadata["orphan_span_ids"]`.
"""

from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    RetrievalPayload,
    Span,
    SpanKind,
    Step,
    ToolPayload,
)

_STEP_KINDS = frozenset({SpanKind.LLM, SpanKind.TOOL, SpanKind.RETRIEVER, SpanKind.AGENT})
_EMPTY_RESULTS = frozenset({"", "[]", "{}", "null", "no results found"})


def _true_root(spans: list[Span]) -> Span:
    declared_roots = [s for s in spans if s.parent_span_id is None]
    candidates = declared_roots or spans
    return min(candidates, key=lambda s: (s.start_ns, s.span_id))


def _effective_parents(spans: list[Span]) -> tuple[str, dict[str, str], list[str]]:
    """(root_span_id, {span_id: effective_parent_span_id}, orphan_span_ids).

    A span is an orphan if its declared `parent_span_id` does not resolve to
    a real, root-reachable ancestor within this list: missing, self-
    referential, one of several `parent_span_id=None` "roots" that lost the
    tie-break, or part of a parent cycle that never reaches the root.
    Orphans are re-parented directly under the true root rather than
    dropped."""
    spans_by_id = {s.span_id: s for s in spans}
    root = _true_root(spans)

    effective_parent: dict[str, str] = {}
    orphans: list[str] = []
    for span in spans:
        if span.span_id == root.span_id:
            continue
        parent_id = span.parent_span_id
        if parent_id is not None and parent_id in spans_by_id and parent_id != span.span_id:
            effective_parent[span.span_id] = parent_id
        else:
            effective_parent[span.span_id] = root.span_id
            orphans.append(span.span_id)

    # Break any parent cycle that never reaches the root: walk each span's
    # chain and, on a repeat, re-parent that span directly under root. Rare
    # in practice (a genuinely cyclic parent chain), but must never hang.
    for span in spans:
        seen: set[str] = set()
        cur = span.span_id
        while cur != root.span_id:
            if cur in seen:
                effective_parent[span.span_id] = root.span_id
                if span.span_id not in orphans:
                    orphans.append(span.span_id)
                break
            seen.add(cur)
            cur = effective_parent.get(cur, root.span_id)

    return root.span_id, effective_parent, orphans


def orphan_span_ids(spans: list[Span]) -> list[str]:
    """Span ids re-parented under the trace root because their declared
    parent did not resolve. See module docstring for why this is separate
    from `linearize()`."""
    if not spans:
        return []
    _, _, orphans = _effective_parents(spans)
    return sorted(orphans)


def _dfs_order(
    root_id: str, children: dict[str, list[Span]], spans_by_id: dict[str, Span]
) -> list[tuple[Span, int, str]]:
    order: list[tuple[Span, int, str]] = []

    def visit(span_id: str, depth: int, path: str) -> None:
        span = spans_by_id[span_id]
        order.append((span, depth, path))
        kids = sorted(children.get(span_id, []), key=lambda s: (s.start_ns, s.span_id))
        for i, kid in enumerate(kids):
            visit(kid.span_id, depth + 1, f"{path}.{i}")

    visit(root_id, 0, "0")
    return order


def _tool_outcome(payload: ToolPayload) -> str:
    if payload.is_error:
        return "err"
    if payload.result_len == 0 or payload.result_text.strip().lower() in _EMPTY_RESULTS:
        return "empty"
    return "ok"


def _signature_and_summary(span: Span, actor: str, seen_llm_actors: set[str]) -> tuple[str, str]:
    payload = span.payload

    if span.kind == SpanKind.AGENT:
        return f"agent:{actor}:invoke", f"{actor} is invoked"

    if span.kind == SpanKind.LLM:
        is_first_for_actor = actor not in seen_llm_actors
        seen_llm_actors.add(actor)
        tool_calls = payload.tool_calls if isinstance(payload, LlmPayload) else []
        if tool_calls:
            tool_name = tool_calls[0].tool_name
            return f"llm:{actor}:call:{tool_name}", f"{actor} decides to call {tool_name}"
        if is_first_for_actor:
            return f"llm:{actor}:plan", f"{actor} plans its next step"
        return f"llm:{actor}:answer", f"{actor} responds"

    if span.kind == SpanKind.TOOL:
        if isinstance(payload, ToolPayload):
            tool_name, outcome = payload.tool_name, _tool_outcome(payload)
        else:
            tool_name, outcome = span.name, "ok"
        return f"tool:{actor}:{tool_name}:{outcome}", f"{tool_name} returns {outcome}"

    if span.kind == SpanKind.RETRIEVER:
        query = payload.query if isinstance(payload, RetrievalPayload) else ""
        hit = "hit" if isinstance(payload, RetrievalPayload) and payload.documents else "miss"
        return f"retr:{actor}:w{len(query.split())}:{hit}", f"{actor} retrieves documents ({hit})"

    return f"unknown:{actor}", f"{actor} performs an unclassified step"


def _resolve_actor(
    span: Span, root_id: str, effective_parent: dict[str, str], actor_by_span: dict[str, str],
) -> str:
    if span.kind == SpanKind.AGENT and isinstance(span.payload, AgentPayload):
        return span.payload.agent_name
    if span.span_id == root_id:
        return span.name or "unknown"
    parent_id = effective_parent.get(span.span_id)
    return actor_by_span.get(parent_id, "unknown")


def linearize(spans: list[Span]) -> list[Step]:
    if not spans:
        return []

    root_id, effective_parent, _orphans = _effective_parents(spans)
    spans_by_id = {s.span_id: s for s in spans}
    children: dict[str, list[Span]] = {}
    for span in spans:
        if span.span_id == root_id:
            continue
        children.setdefault(effective_parent[span.span_id], []).append(span)

    ordered = _dfs_order(root_id, children, spans_by_id)

    steps: list[Step] = []
    actor_by_span: dict[str, str] = {}
    seen_llm_actors: set[str] = set()
    pending_collapsed: list[str] = []

    for span, depth, path in ordered:
        actor = _resolve_actor(span, root_id, effective_parent, actor_by_span)
        actor_by_span[span.span_id] = actor

        if span.kind not in _STEP_KINDS:
            pending_collapsed.append(span.span_id)
            continue

        signature, summary = _signature_and_summary(span, actor, seen_llm_actors)
        steps.append(Step(
            trace_id=span.trace_id, step_index=len(steps), span_id=span.span_id,
            kind=span.kind, actor=actor, depth=depth, tree_path=path,
            signature=signature, summary=summary, start_ns=span.start_ns,
            end_ns=span.end_ns, duration_ms=(span.end_ns - span.start_ns) / 1_000_000,
            collapsed_span_ids=pending_collapsed,
        ))
        pending_collapsed = []

    # Trailing non-semantic spans after the last semantic step (e.g. a
    # closing GUARDRAIL span with no further real action) have no "next"
    # step to attach to - fold them into the last step instead of losing
    # the attribution entirely.
    if pending_collapsed and steps:
        last = steps[-1]
        steps[-1] = last.model_copy(
            update={"collapsed_span_ids": last.collapsed_span_ids + pending_collapsed}
        )

    return steps
