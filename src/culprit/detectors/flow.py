"""premature_termination, missing_verification, duplicate_delegation.

`missing_verification` is gated on `trace.outcome == FAILURE`; see its
docstring below for why that gate is load-bearing rather than decorative,
and the module's final report entry for the same point.
"""

from culprit.detectors.base import DetectorContext, truncate
from culprit.schemas import AgentPayload, Outcome, SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass


def premature_termination(ctx: DetectorContext) -> list[Signal]:
    """The run ends on an "answer" LLM step with no completing TOOL action
    immediately before it. In this domain a healthy run always ends
    `TOOL(process_refund) -> LLM(answer)`; a run that jumps straight from
    planning or a tool *request* to a final answer, with the corresponding
    tool execution missing, skipped the step that would have completed the
    task."""
    steps = ctx.steps
    if len(steps) < 2:
        return []
    last, prior = steps[-1], steps[-2]
    if last.kind != SpanKind.LLM or "answer" not in last.signature:
        return []
    if prior.kind == SpanKind.TOOL:
        return []
    return [Signal(
        detector="premature_termination", step_index=prior.step_index, span_id=prior.span_id,
        severity=0.65, category=FailureClass.PREMATURE_TERMINATION.value,
        message="Run answers without a completing tool action immediately before it",
        evidence=[Evidence(
            span_id=prior.span_id, step_index=prior.step_index, field="signature",
            excerpt=truncate(prior.signature),
        )],
    )]


def missing_verification(ctx: DetectorContext) -> list[Signal]:
    """`process_refund` executes with no `verify_eligibility` step anywhere
    in the trace.

    This synth generator makes verification genuinely optional: roughly
    half of clean successful runs skip it too (see the final report), so
    presence/absence alone cannot tell "verification was not needed" apart
    from "verification was skipped and that is why this run failed" without
    also knowing the run failed. Gating on `trace.outcome == Outcome.FAILURE`
    resolves that: it costs nothing in real use, since `run_detectors` is
    only ever invoked on a failed trace in production (diagnosing failures
    is the whole product), and it is what keeps this detector silent on
    every synthetic success regardless of that run's own verification
    choice.

    String-matches the synthetic tool names `verify_eligibility` and
    `process_refund`, binding this detector to the synth domain; it is inert
    on real traces.
    """
    if ctx.trace.outcome != Outcome.FAILURE:
        return []
    if any("verify_eligibility" in s.signature for s in ctx.steps):
        return []
    process_step = next(
        (s for s in ctx.steps if s.kind == SpanKind.TOOL and "process_refund" in s.signature), None
    )
    if process_step is None:
        return []
    return [Signal(
        detector="missing_verification", step_index=process_step.step_index, span_id=process_step.span_id,
        severity=0.5, category=FailureClass.MISSING_VERIFICATION.value,
        message="process_refund executed without any eligibility verification step in this failed run",
        evidence=[Evidence(
            span_id=process_step.span_id, step_index=process_step.step_index, field="signature",
            excerpt=truncate(process_step.signature),
        )],
    )]


def duplicate_delegation(ctx: DetectorContext) -> list[Signal]:
    """An AGENT step delegates to the same agent name more than once in one
    delegation call. Clean runs always delegate to a single agent, so a
    repeat in the list can only come from a genuine duplicate.
    No ratified delegation attribute exists in either vocabulary, and
    `_agent_payload` hardcodes `delegated_to=[]`, so this detector is inert
    on real traces."""
    signals = []
    for step in ctx.steps:
        if step.kind != SpanKind.AGENT:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, AgentPayload):
            continue
        delegated = span.payload.delegated_to
        if len(delegated) != len(set(delegated)):
            signals.append(Signal(
                detector="duplicate_delegation", step_index=step.step_index, span_id=step.span_id,
                severity=0.55, category=FailureClass.CONFLICTING_SUBRESULTS_UNRECONCILED.value,
                message=f"Delegates to the same agent more than once: {delegated}",
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="payload.delegated_to",
                    excerpt=truncate(str(delegated)),
                )],
            ))
    return signals
