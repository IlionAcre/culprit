"""context_overflow, silent_history_truncation, step_budget_exhausted.

`silent_history_truncation` is one of the four detectors the product exists
to catch (CLAUDE.md's "L1 detectors" section): a framework silently
dropping conversation history is invisible in every existing observability
tool, and the trace looks entirely healthy at the point it happens.
"""

from culprit.detectors.base import DetectorContext, truncate
from culprit.schemas import SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_OVERFLOW_RATIO = 0.9
_TRUNCATION_DROP_RATIO = 0.7  # curr <= 0.7 * prev means more than a 30% drop
_BUDGET_RUN_LENGTH = 10


def context_overflow(ctx: DetectorContext) -> list[Signal]:
    """`total_tokens >= 0.9 * max_tokens` on an LLM step, per the plan's
    catalogue entry, which is deliberately model-agnostic: it reads the
    reported context window rather than hardcoding a per-model table."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.LLM:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        _prompt, _completion, total = ctx.token_series.get(step.step_index, (None, None, None))
        max_tokens = getattr(span.payload, "max_tokens", None) if span is not None else None
        if total is None or max_tokens is None or max_tokens <= 0:
            continue
        if total >= _OVERFLOW_RATIO * max_tokens:
            signals.append(Signal(
                detector="context_overflow", step_index=step.step_index, span_id=step.span_id,
                severity=0.6, category=FailureClass.CONTEXT_LOSS.value,
                message=f"total_tokens {total} is within 10% of max_tokens {max_tokens}",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.total_tokens",
                    excerpt=str(total), numeric=float(total),
                )],
            ))
    return signals


def silent_history_truncation(ctx: DetectorContext) -> list[Signal]:
    """`prompt_tokens` drops more than 30% between consecutive LLM steps of
    the same actor while the request message count keeps growing (a
    framework silently dropped history rather than the conversation
    genuinely shrinking). The message-count check is what tells a real
    truncation apart from the conversation legitimately being shorter."""
    signals = []
    for actor, indices in ctx.llm_steps_by_actor.items():
        for prev_i, curr_i in zip(indices, indices[1:]):
            prev_prompt = ctx.token_series.get(prev_i, (None, None, None))[0]
            curr_prompt = ctx.token_series.get(curr_i, (None, None, None))[0]
            if prev_prompt is None or curr_prompt is None or prev_prompt <= 0:
                continue
            prev_span = ctx.spans_by_id.get(ctx.steps[prev_i].span_id)
            curr_span = ctx.spans_by_id.get(ctx.steps[curr_i].span_id)
            prev_msgs = len(getattr(prev_span.payload, "request_messages", None) or [])
            curr_msgs = len(getattr(curr_span.payload, "request_messages", None) or [])
            if curr_prompt <= prev_prompt * _TRUNCATION_DROP_RATIO and curr_msgs >= prev_msgs:
                signals.append(Signal(
                    detector="silent_history_truncation", step_index=curr_i, span_id=ctx.steps[curr_i].span_id,
                    severity=0.75, category=FailureClass.CONTEXT_LOSS.value,
                    message=(
                        f"prompt_tokens dropped from {prev_prompt} to {curr_prompt} for {actor} "
                        "while the conversation kept growing"
                    ),
                    evidence=[Evidence(
                        span_id=ctx.steps[curr_i].span_id, step_index=curr_i, field="payload.prompt_tokens",
                        excerpt=f"{prev_prompt} -> {curr_prompt}", numeric=float(curr_prompt),
                    )],
                ))
    return signals


def step_budget_exhausted(ctx: DetectorContext) -> list[Signal]:
    """A run of `_BUDGET_RUN_LENGTH`+ consecutive steps sharing one
    signature that never resolves. The flagged step is the one immediately
    before the run starts (`start - 1`), not the run itself: that is the
    point where the agent's decision to keep repeating was made, matching
    `synth_inject.py`'s `_inj_step_budget_exhausted`, which returns the last
    normal step before the padding, not one of the padding steps."""
    signals = []
    for signature, start, length in ctx.signature_runs:
        if length < _BUDGET_RUN_LENGTH:
            continue
        cause_index = max(start - 1, 0)
        signals.append(Signal(
            detector="step_budget_exhausted", step_index=cause_index, span_id=ctx.steps[cause_index].span_id,
            severity=0.7, category=FailureClass.STEP_BUDGET_EXHAUSTED.value,
            message=f"{length} consecutive steps with signature {signature!r} never resolved",
            evidence=[Evidence(
                span_id=ctx.steps[cause_index].span_id, step_index=cause_index, field="signature",
                excerpt=truncate(signature),
            )],
        ))
    return signals
