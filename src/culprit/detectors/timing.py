"""stall_timeout: a step whose duration is far outside the norm for this
trace.

The plan's catalogue entry reads ">5x median for that signature", which is
a cross-trace historical statistic L2's reference pool would supply, not
something available at L1 (usually one occurrence per signature per trace).
This uses an in-trace proxy instead: the median duration of steps of the
same *kind* in this trace, which is enough to catch one step stalling for
many times longer than everything around it without needing history this
layer does not have.
"""

from statistics import median

from culprit.detectors.base import DetectorContext
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

_STALL_RATIO = 5.0
_MIN_MEDIAN_MS = 1.0


def stall_timeout(ctx: DetectorContext) -> list[Signal]:
    by_kind: dict[str, list[float]] = {}
    for step in ctx.steps:
        by_kind.setdefault(step.kind.value, []).append(step.duration_ms)
    medians = {kind: median(durations) for kind, durations in by_kind.items()}

    signals = []
    for step in ctx.steps:
        m = medians.get(step.kind.value, 0.0)
        if m < _MIN_MEDIAN_MS:
            continue
        if step.duration_ms >= _STALL_RATIO * m:
            signals.append(Signal(
                detector="stall_timeout", step_index=step.step_index, span_id=step.span_id,
                severity=0.55, category=FailureClass.UNKNOWN.value,
                message=(
                    f"{step.signature} took {step.duration_ms:.0f}ms, over {_STALL_RATIO:.0f}x "
                    f"the {m:.0f}ms median for {step.kind.value} steps in this trace"
                ),
                evidence=[Evidence(
                    span_id=step.span_id, step_index=step.step_index, field="duration_ms",
                    excerpt=f"{step.duration_ms:.0f}ms vs {m:.0f}ms median", numeric=step.duration_ms,
                )],
            ))
    return signals
