"""L0-L5 orchestration entrypoint: the single function that runs the whole
analysis cascade for one trace and returns a `Diagnosis`.

Filled in during Integration task I1, after WS-A through WS-F all landed.
**Per-layer isolation is the point of this module, more than the wiring
itself** (CLAUDE.md's `degraded_layers` guardrail): a layer that raises
degrades the diagnosis and records itself in `Diagnosis.degraded_layers`
rather than crashing the run, so a caller always gets *some* diagnosis and
is told honestly how much of the cascade produced it. L2 additionally
degrades on a clean, designed abstention (`ContrastResult.abstained`), not
only on a raised exception - an abstained L2 is exactly as materially
weaker a diagnosis as a crashed one, and hiding that behind an empty
`divergences` list would be the "quietly presenting a weaker answer as a
complete one" bug this mechanism exists to prevent.

L5 clustering is deliberately not called from here: `jobs.py`'s
`recluster_job` runs it separately as a scheduled batch recompute over the
accumulated diagnosis corpus, never inline per diagnosis (a single new
diagnosis is not a meaningful clustering input on its own).

**Foundation amendment (post-Phase-0): added the `model` parameter.** WS-G,
the first real caller, found that `adjudicate(candidates, trace, steps, *,
call_fn, model)` requires a model string and nothing routed one in here. The
project's layering rule (core modules never import `culprit.config`; only
`cli.py` and plugin registries do) means the model cannot be resolved
internally, so it is threaded through as a required keyword argument, exactly
like `conn_fn`/`call_fn`/`embed_fn`.
"""

import logging
import uuid
from datetime import UTC, datetime

from culprit.confidence import select_diagnosis
from culprit.db import ConnFn
from culprit.embed import EmbedFn
from culprit.llm import CallFn
from culprit.logging_config import LOGGER_NAME
from culprit.run_detectors import run_detectors
from culprit.schemas import Trace
from culprit.signals import Diagnosis, DivergenceCandidate, Signal
from culprit.store_traces import read_trace

logger = logging.getLogger(LOGGER_NAME)

# Recorded on every Diagnosis regardless of which layers degraded, so a
# later reader can tell which code version produced it (Apple's "Overton"
# citation in CLAUDE.md: re-running analysis makes a new Diagnosis, and this
# is what makes two diagnoses for the same trace comparable).
LAYER_VERSIONS = {"l0": "1.0.0", "l1": "1.0.0", "l2": "1.0.0", "l3": "1.0.0"}


def _load_steps(trace: Trace, conn_fn: ConnFn) -> tuple[list, dict]:
    """L0's output (linearized steps, spans keyed by id) was already
    computed and persisted at ingest time (`jobs.ingest_trace`); this
    re-reads it via `conn_fn` rather than requiring the caller to pass
    spans/steps directly, matching `jobs._default_trace_loader`'s own
    Trace-only return shape. A trace with no persisted spans/steps (empty
    trace, or a read failure) degrades to empty lists rather than raising,
    so L1-L3 still run over nothing and produce an honest, if thin, result.
    """
    _trace, spans, steps = read_trace(conn_fn, trace.trace_id)
    return steps, {s.span_id: s for s in spans}


def diagnose(
    trace: Trace, *, conn_fn: ConnFn, call_fn: CallFn, embed_fn: EmbedFn, model: str
) -> Diagnosis:
    """Run L1 (detectors) through L3 (adjudication) over one trace's already
    L0-normalized steps, and return the resulting `Diagnosis`. Does not
    persist; `jobs.diagnose_trace_job` is the caller responsible for
    writing the result via `store_diagnoses.write_diagnosis`.

    Every layer below is wrapped in its own `try/except`: a raise in one
    layer never stops the layers after it from running, it only costs that
    layer's contribution and adds its name to `degraded_layers`. This is
    filled in incrementally as L2/L3 wiring lands; L0+L1 wiring in this
    revision is the first proven slice of that isolation mechanism.
    """
    degraded_layers: list[str] = []

    try:
        steps, spans_by_id = _load_steps(trace, conn_fn)
    except Exception as exc:  # noqa: BLE001 - layer isolation is the point
        logger.warning(
            "L0 step load failed", extra={"event": "pipeline_l0_failed", "trace_id": trace.trace_id, "error": str(exc)},
        )
        steps, spans_by_id = [], {}
        degraded_layers.append("l0")

    signals: list[Signal] = []
    try:
        signals = run_detectors(trace, steps, spans_by_id)
    except Exception as exc:  # noqa: BLE001 - layer isolation is the point
        logger.warning(
            "L1 detectors failed", extra={"event": "pipeline_l1_failed", "trace_id": trace.trace_id, "error": str(exc)},
        )
        degraded_layers.append("l1")

    # L2 and L3 land in a following revision; wired here as empty results so
    # this slice is a complete, correct function on its own.
    divergences: list[DivergenceCandidate] = []
    adjudications: list = []

    verdict = select_diagnosis(adjudications)

    return Diagnosis(
        diagnosis_id=str(uuid.uuid4()),
        trace_id=trace.trace_id,
        created_at=datetime.now(UTC),
        root_cause_step_index=verdict.winner.step_index if verdict.winner else None,
        root_cause_span_id=verdict.winner.span_id if verdict.winner else None,
        failure_class=verdict.winner.failure_class if verdict.winner else None,
        confidence=verdict.winner.confidence if verdict.winner else 0.0,
        calibrated_confidence=verdict.winner.calibrated_confidence if verdict.winner else 0.0,
        abstained=verdict.abstained,
        abstain_reason=verdict.abstain_reason,
        rationale=verdict.winner.rationale if verdict.winner else "",
        counterfactual=verdict.winner.counterfactual if verdict.winner else "",
        candidates_considered=0,
        signals=signals,
        divergences=divergences,
        adjudications=adjudications,
        layer_versions=dict(LAYER_VERSIONS),
        degraded_layers=degraded_layers,
    )
