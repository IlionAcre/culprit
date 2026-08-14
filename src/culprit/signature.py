"""The alignment alphabet (coarse signature token) and the fine-grained
similarity function `align.py` uses for substitution scoring.

**Load-bearing property, inherited from `linearize.py` and restated here
because getting it wrong collapses the whole layer**: the coarse token for
an LLM step encodes the decision the step made (`llm:{actor}:call:{tool}` /
`:plan` / `:answer`), never the model name. Signed by model name instead,
every LLM step becomes an identical token and alignment degenerates to
"how many steps are LLM calls," which carries no positional information at
all.

**Contract fix, authorized after the WS-D handoff report.** `contrast()`
originally received only `steps: list[Step]`, so three of the plan's eight
fine features (arg_key_jaccard, arg_value_jaccard, retrieval_docid_overlap)
were structurally unreachable and degraded to a fixed neutral value. The
frozen signature now mirrors `run_detectors(trace, steps, spans_by_id)` and
passes spans through, so `sim()` takes optional `Span`s and those three
features are real. A caller with no span for a step (or none at all) still
gets a sane fallback, not a crash - see `_jaccard`'s docstring.

`signature_of` also now folds a small set of payload-derived facts into the
coarse token itself when a span is available, extending the existing
`{ok|err|empty}` convention rather than departing from it (malformed tool
arguments, a near-full context window, more than one delegation target, a
low mean retrieval score, and a retrieval query that drifted from what the
signature originally recorded). Each is a single deterministic fact read
off one span's own payload - no cross-step lookback, no task-specific
knowledge of what an "order_id" is. `parameter_drift` and
`silent_history_truncation` are deliberately not attempted here: the first
would need excluding task-specific argument keys to avoid drowning in
noise from synth's per-trace-random order_id (every tool call's arguments
differ from every reference's for that reason regardless of any injected
fault), and the second is a genuinely cross-step pattern already owned by
an L1 detector of the same name. Reported as open gaps rather than forced.
"""

import json

from culprit.schemas import AgentPayload, LlmPayload, RetrievalPayload, Span, Step, ToolPayload

# Feature order and weights from the plan's L2 spec, `sim(a, b)` in [0, 1]:
# [kind_match, actor_match, tool_match, outcome_match, arg_key_jaccard,
#  arg_value_jaccard, 1-|depth_delta|/4, retrieval_docid_overlap]
_WEIGHTS = (0.25, 0.15, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03)

_LOW_SCORE_THRESHOLD = 0.3
_NEAR_LIMIT_RATIO = 0.9  # matches L1 context_overflow's own threshold


def _payload_flags(step: Step, span: Span | None) -> list[str]:
    """Deterministic facts read off one span's payload, appended to the
    coarse token (see module docstring for why these five and not more)."""
    if span is None or span.payload is None:
        return []
    p = span.payload
    flags: list[str] = []

    if isinstance(p, ToolPayload):
        if not p.is_error and p.arguments == {} and p.arguments_json not in (None, "", "{}"):
            flags.append("argsmalformed")
    elif isinstance(p, LlmPayload):
        if p.response_format_schema is not None:
            content = p.response_messages[0].content if p.response_messages else None
            try:
                if content is None:
                    raise ValueError
                json.loads(content)
            except (ValueError, TypeError, json.JSONDecodeError):
                flags.append("schemaviolation")
        if p.total_tokens is not None and p.max_tokens:
            if p.total_tokens >= _NEAR_LIMIT_RATIO * p.max_tokens:
                flags.append("neartokenlimit")
    elif isinstance(p, RetrievalPayload):
        actual_words = len(p.query.split())
        if f"w{actual_words}:" not in step.signature:
            flags.append("querydrift")
        scores = [d.score for d in p.documents if d.score is not None]
        if scores and (sum(scores) / len(scores)) < _LOW_SCORE_THRESHOLD:
            flags.append("lowscore")
    elif isinstance(p, AgentPayload):
        if len(p.delegated_to) > 1:
            flags.append(f"delegates{len(p.delegated_to)}")

    return flags


def signature_of(step: Step, span: Span | None = None) -> str:
    """The coarse alignment token for one step: `Step.signature` (computed
    at ingestion by `linearize.py` using the `kind:actor:...` convention
    this module documents) plus any payload-derived flags, appended after
    a `|` so the base token's colon-delimited structure stays parseable
    (`_tool_token`/`_outcome_token` split on `|` first). No flags means no
    `|` at all, so the large majority of tokens are byte-identical to the
    pre-contract-fix behavior."""
    flags = _payload_flags(step, span)
    if not flags:
        return step.signature
    return f"{step.signature}|{','.join(sorted(flags))}"


def _tool_token(signature: str) -> str:
    base = signature.split("|", 1)[0]
    parts = base.split(":")
    if parts[0] == "llm" and len(parts) >= 4 and parts[2] == "call":
        return parts[3]
    if parts[0] == "tool" and len(parts) >= 3:
        return parts[2]
    return ""


def _outcome_token(signature: str) -> str:
    """The trailing ok/err/empty/hit/miss suffix plus any payload flags: a
    normal tool call and one whose arguments failed to parse are not the
    same outcome even when they share a tool name, so the flags belong in
    this feature rather than a new weighted term the spec didn't ask for."""
    base, _, flags = signature.partition("|")
    parts = base.split(":")
    tail = parts[-1] if len(parts) >= 3 else ""
    return f"{tail}|{flags}" if flags else tail


def _arg_items(span: Span | None) -> dict[str, object] | None:
    if span is None or span.payload is None:
        return None
    p = span.payload
    if isinstance(p, ToolPayload):
        return p.arguments
    if isinstance(p, LlmPayload) and p.tool_calls:
        return p.tool_calls[0].arguments
    return None


def _doc_ids(span: Span | None) -> set[str] | None:
    if span is None or not isinstance(span.payload, RetrievalPayload):
        return None
    return {d.doc_id for d in span.payload.documents}


def _jaccard(a: set | None, b: set | None) -> float:
    """1.0 when neither side carries the feature (two steps that both have
    no arguments, or both have no retrieved docs, trivially agree on that),
    0.0 when only one side does (a real disagreement), and plain Jaccard
    when both do. This is the same "both empty agrees" reasoning that fixed
    `tool_match` for two non-tool-calling steps - see the git history for
    that bug."""
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def sim(a: Step, b: Step, span_a: Span | None = None, span_b: Span | None = None) -> float:
    """Weighted 8-feature similarity in [0, 1]. `align.py`'s substitution
    score is `2 * sim(a, b) - 1`, so identical steps score close to +1 and
    thoroughly unrelated steps score close to -1. `span_a`/`span_b` are
    optional so this stays callable with just `Step`s (tests, or a future
    caller with no span for one side); omitting them falls the three
    payload features back to "both absent" (1.0), not the old fixed 0.5 -
    see `_jaccard`."""
    sig_a, sig_b = signature_of(a, span_a), signature_of(b, span_b)
    tool_a, tool_b = _tool_token(sig_a), _tool_token(sig_b)
    out_a, out_b = _outcome_token(sig_a), _outcome_token(sig_b)

    arg_a, arg_b = _arg_items(span_a), _arg_items(span_b)
    arg_keys = _jaccard(set(arg_a) if arg_a is not None else None, set(arg_b) if arg_b is not None else None)
    arg_values = _jaccard(
        {f"{k}={v}" for k, v in arg_a.items()} if arg_a is not None else None,
        {f"{k}={v}" for k, v in arg_b.items()} if arg_b is not None else None,
    )
    retrieval_overlap = _jaccard(_doc_ids(span_a), _doc_ids(span_b))

    features = (
        1.0 if a.kind == b.kind else 0.0,
        1.0 if a.actor == b.actor else 0.0,
        1.0 if tool_a == tool_b else 0.0,
        1.0 if out_a and out_a == out_b else 0.0,
        arg_keys,
        arg_values,
        1.0 - min(abs(a.depth - b.depth), 4) / 4,
        retrieval_overlap,
    )
    return sum(w * f for w, f in zip(_WEIGHTS, features))
