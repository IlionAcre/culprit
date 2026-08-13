"""L1 deterministic detector entrypoint: runs the full detector catalogue
over one trace's steps and returns their signals.

Stubbed here at its real signature during Phase 0 so downstream workstreams
(WS-E adjudication via `culprit.synth_results`, WS-G service surface via
monkeypatching) can be written and tested against it before WS-C exists.
WS-C fills the body and additionally owns `detectors/base.py`,
`detectors/registry.py`, and one module per detector family (`tool_errors.py`,
`loops.py`, `context.py`, `schema.py`, `flow.py`, `retrieval.py`,
`provenance.py`, `timing.py`).
"""

from culprit.schemas import Span, Step, Trace
from culprit.signals import Signal


def run_detectors(
    trace: Trace, steps: list[Step], spans_by_id: dict[str, Span]
) -> list[Signal]:
    """Build the shared `DetectorContext` once, then run every registered
    detector against it, returning one `Signal` list.

    Owned by WS-C. Per the per-item error isolation rule, a detector that
    raises must produce exactly one sentinel `Signal` with `error` set and
    be excluded from downstream ranking; the other detectors still run.
    """
    raise NotImplementedError("owned by WS-C")
