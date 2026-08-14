"""oscillation, repeated_identical_action, retry_storm: three flavors of
"the agent is not making progress," differing only in how the repetition is
shaped. All three are scoped to TOOL steps, because a repeated *decision* to
call a tool is generic (an LLM step can say the same thing twice for
legitimate reasons), but repeating the tool call itself against the
environment is what actually costs nothing to happen and everything to miss.
"""

from culprit.detectors.base import DetectorContext
from culprit.schemas import SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_MIN_RUN = 3  # the original call plus at least 2 repeats


def _tool_name(ctx: DetectorContext, step_index: int) -> str | None:
    span = ctx.spans_by_id.get(ctx.steps[step_index].span_id)
    return getattr(span.payload, "tool_name", None) if span is not None else None


def oscillation(ctx: DetectorContext) -> list[Signal]:
    """A B A B (period 2-4, at least 2 full cycles) among TOOL steps. Clean
    runs call every tool exactly once, so a genuine 2-distinct-tool
    alternating pattern cannot occur by construction; only a real loop
    produces one."""
    tool_steps = [s.step_index for s in ctx.steps if s.kind == SpanKind.TOOL]
    names = [_tool_name(ctx, i) for i in tool_steps]
    for period in (2, 3, 4):
        window = period * 2
        for start in range(len(names) - window + 1):
            block = names[start:start + period]
            if len(set(block)) < 2:
                continue  # a constant block is repeated_identical_action's / retry_storm's territory
            if names[start:start + window] == block * 2:
                i = tool_steps[start]
                return [Signal(
                    detector="oscillation", step_index=i, span_id=ctx.steps[i].span_id,
                    severity=0.75, category=FailureClass.INFINITE_LOOP_OR_OSCILLATION.value,
                    message=f"Tool calls oscillate between {sorted(set(block))} with period {period}",
                    evidence=[Evidence(
                        span_id=ctx.steps[i].span_id, step_index=i, field="signature",
                        excerpt=" -> ".join(block * 2),
                    )],
                )]
    return []


def _consecutive_same_args_runs(ctx: DetectorContext) -> list[tuple[int, int]]:
    """(start_step_index, run_length) for maximal runs of consecutive TOOL
    steps sharing the same tool name and the same argument hash."""
    runs: list[tuple[int, int]] = []
    steps = ctx.steps
    i = 0
    while i < len(steps):
        if steps[i].kind != SpanKind.TOOL or ctx.args_hash_by_step.get(i) is None:
            i += 1
            continue
        name_i = _tool_name(ctx, i)
        j = i + 1
        while (
            j < len(steps)
            and steps[j].kind == SpanKind.TOOL
            and _tool_name(ctx, j) == name_i
            and ctx.args_hash_by_step.get(j) == ctx.args_hash_by_step.get(i)
        ):
            j += 1
        runs.append((i, j - i))
        i = j
    return runs


def repeated_identical_action(ctx: DetectorContext) -> list[Signal]:
    """The same tool called back-to-back with identical arguments, but the
    results differ call to call: some progress is happening even though the
    action is being repeated. Zero-progress repetition (identical results
    too) is `retry_storm`'s stronger case, not this one's, so the two never
    double-fire on the same run."""
    signals = []
    for start, length in _consecutive_same_args_runs(ctx):
        if length < _MIN_RUN:
            continue
        results = {ctx.result_hash_by_step.get(k) for k in range(start, start + length)}
        if len(results) == 1:
            continue
        name = _tool_name(ctx, start)
        signals.append(Signal(
            detector="repeated_identical_action", step_index=start, span_id=ctx.steps[start].span_id,
            severity=0.5, category=FailureClass.INFINITE_LOOP_OR_OSCILLATION.value,
            message=f"{name} called {length} times in a row with identical arguments",
            evidence=[Evidence(
                span_id=ctx.steps[start].span_id, step_index=start, field="payload.arguments",
                excerpt=f"{name} repeated {length} times with identical arguments",
            )],
        ))
    return signals


def retry_storm(ctx: DetectorContext) -> list[Signal]:
    """The same tool called back-to-back with identical arguments *and* an
    identical result every time: zero progress, per the plan's catalogue
    entry for this detector."""
    signals = []
    for start, length in _consecutive_same_args_runs(ctx):
        if length < _MIN_RUN:
            continue
        results = {ctx.result_hash_by_step.get(k) for k in range(start, start + length)}
        if len(results) != 1:
            continue
        name = _tool_name(ctx, start)
        signals.append(Signal(
            detector="retry_storm", step_index=start, span_id=ctx.steps[start].span_id,
            severity=0.8, category=FailureClass.INFINITE_LOOP_OR_OSCILLATION.value,
            message=f"{name} retried {length} times with an identical result each time (zero progress)",
            evidence=[Evidence(
                span_id=ctx.steps[start].span_id, step_index=start, field="payload.result_text",
                excerpt=f"{name} produced the identical result across {length} consecutive calls",
            )],
        ))
    return signals
