"""unused_retrieval, low_score_retrieval, goal_token_drift.

The plan's catalogue wording for `unused_retrieval` is "<15% token overlap
with next prompt". This synth generator never actually stitches retrieved
content into any later prompt (every LLM call is an independent,
hand-authored instruction string, not an accumulating conversation), so a
literal next-prompt overlap check would fire on every retrieval in every
trace, clean or not: there would be nothing to compare against. Both
retrieval detectors below instead measure the retrieval *query*'s own
coverage against, respectively, the retrieved documents and the task goal,
which is the same "was this retrieval actually about what we were doing"
question, answered against data this fixture actually populates. See the
final report for this called out explicitly as a spec/fixture mismatch.
"""

from culprit.detectors.base import DetectorContext, tokens, truncate
from culprit.schemas import RetrievalPayload, SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_COVERAGE_THRESHOLD = 0.15
_LOW_SCORE_THRESHOLD = 0.3


def _coverage(reference: frozenset[str], candidate: frozenset[str]) -> float:
    """Fraction of `reference`'s tokens also present in `candidate`. Using
    the (small, focused) query as `reference` avoids the dilution a large
    bag of boilerplate document words would otherwise cause."""
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def unused_retrieval(ctx: DetectorContext) -> list[Signal]:
    """The retrieval query's own vocabulary barely appears anywhere in what
    was retrieved: the documents are not actually about the query."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.RETRIEVER:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, RetrievalPayload):
            continue
        p = span.payload
        if not p.documents:
            continue
        doc_terms = tokens(" ".join(d.content_preview for d in p.documents))
        if _coverage(tokens(p.query), doc_terms) < _COVERAGE_THRESHOLD:
            signals.append(Signal(
                detector="unused_retrieval", step_index=step.step_index, span_id=step.span_id,
                severity=0.45, category=FailureClass.RETRIEVAL_MISS.value,
                message="Retrieved documents share almost no vocabulary with the query that fetched them",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.documents",
                    excerpt=truncate(" | ".join(d.content_preview for d in p.documents)),
                )],
            ))
    return signals


def low_score_retrieval(ctx: DetectorContext) -> list[Signal]:
    """Every retrieved document scores below the relevance threshold."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.RETRIEVER:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, RetrievalPayload):
            continue
        scores = [d.score for d in span.payload.documents if d.score is not None]
        if scores and max(scores) < _LOW_SCORE_THRESHOLD:
            signals.append(Signal(
                detector="low_score_retrieval", step_index=step.step_index, span_id=step.span_id,
                severity=0.5, category=FailureClass.RETRIEVAL_MISS.value,
                message=f"Best retrieval score {max(scores):.2f} is far below the relevance threshold",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.documents",
                    excerpt=truncate(str(scores)), numeric=max(scores),
                )],
            ))
    return signals


def goal_token_drift(ctx: DetectorContext) -> list[Signal]:
    """The retrieval query's own vocabulary barely appears in the task
    goal: the agent is searching for something unrelated to what it was
    asked to do."""
    signals = []
    if not ctx.goal_terms:
        return signals
    for step in ctx.steps:
        if step.kind != SpanKind.RETRIEVER:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, RetrievalPayload):
            continue
        query = span.payload.query
        if _coverage(tokens(query), ctx.goal_terms) < _COVERAGE_THRESHOLD:
            signals.append(Signal(
                detector="goal_token_drift", step_index=step.step_index, span_id=step.span_id,
                severity=0.45, category=FailureClass.TASK_MISINTERPRETATION.value,
                message=f"Retrieval query {query!r} shares little vocabulary with the task goal",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.query",
                    excerpt=truncate(query),
                )],
            ))
    return signals
