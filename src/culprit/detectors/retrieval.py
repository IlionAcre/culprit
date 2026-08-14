"""unused_retrieval, low_score_retrieval, goal_token_drift.

**`unused_retrieval` restored to the catalogue's specified overlap test
(Foundation amendment, post-WS-D handoff).** The plan's wording is "<15%
token overlap with next prompt". This detector originally could not
implement that literally: `synth.py`'s LLM prompts were independent
hand-authored strings that never incorporated retrieved content, so a
literal next-prompt check would have fired on every retrieval, clean or
not - there was nothing in the next prompt to compare against. It was
reimplemented as query-vs-document coverage instead (the same "was this
retrieval actually about what we were doing" question, answered against
data the fixture actually populated). Now that `synth.py` folds a
retriever step's documents into the immediately following LLM step's
request message (see `synth.py`'s module docstring), the next prompt has
real content to compare against, so this reverts to comparing retrieved
documents against `DetectorContext.step_text` of the next step - the
literal "next prompt" the spec asks for.
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
    """The retrieved documents' own vocabulary barely appears in the next
    prompt: the documents were fetched but never actually referenced in
    what the agent went on to ask the model. Catalogue spec: <15% token
    overlap with the next prompt. A retrieval step with no following LLM
    step (nothing left to check) is skipped rather than flagged."""
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
        next_step = next(
            (s for s in ctx.steps if s.step_index > step.step_index and s.kind == SpanKind.LLM),
            None,
        )
        if next_step is None:
            continue
        doc_terms = tokens(" ".join(d.content_preview for d in p.documents))
        next_prompt_terms = tokens(ctx.step_text[next_step.step_index])
        if _coverage(doc_terms, next_prompt_terms) < _COVERAGE_THRESHOLD:
            signals.append(Signal(
                detector="unused_retrieval", step_index=step.step_index, span_id=step.span_id,
                severity=0.45, category=FailureClass.RETRIEVAL_MISS.value,
                message="Retrieved documents share almost no vocabulary with the next prompt",
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
