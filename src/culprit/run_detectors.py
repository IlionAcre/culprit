"""L1 deterministic detector entrypoint: runs the full detector catalogue
over one trace's steps and returns their signals.

Owned by WS-C, which additionally owns `detectors/base.py`,
`detectors/registry.py`, and one module per detector family (`tool_errors.py`,
`loops.py`, `context.py`, `schema.py`, `flow.py`, `retrieval.py`,
`provenance.py`, `timing.py`).
"""

from culprit.detectors.base import build_context
from culprit.detectors.registry import DETECTORS
from culprit.schemas import Span, Step, Trace
from culprit.signals import Signal
from culprit.taxonomy import FailureClass


def run_detectors(
    trace: Trace, steps: list[Step], spans_by_id: dict[str, Span]
) -> list[Signal]:
    """Build the shared `DetectorContext` once, then run every registered
    detector against it, returning one `Signal` list.

    Per the per-item error isolation rule, a detector that raises produces
    exactly one sentinel `Signal` with `error` set; the other nineteen still
    run and return normally. Detectors run sequentially (not fanned out with
    `ThreadPoolExecutor`): the catalogue is cheap pure-Python work over an
    already-built `DetectorContext` (the actual 5ms-vs-200ms cost this
    module cares about is building that context once, not per-detector
    threading overhead), so a plain loop keeps the isolation logic and stack
    traces easy to follow.
    """
    ctx = build_context(trace, steps, spans_by_id)
    signals: list[Signal] = []
    for name, detector in DETECTORS.items():
        try:
            signals.extend(detector(ctx))
        except Exception as exc:  # noqa: BLE001 - per-detector isolation is the point
            signals.append(Signal(
                detector=name,
                step_index=-1,
                span_id=None,
                severity=0.0,
                category=FailureClass.UNKNOWN.value,
                message=f"detector {name!r} raised {type(exc).__name__}: {exc}",
                evidence=[],
                error=str(exc),
            ))
    return signals
