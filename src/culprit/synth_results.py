"""Synthetic `Signal` / `DivergenceCandidate` / `Diagnosis` factories.

WS-E (adjudication) and WS-F (clustering) consume analysis results, they do
not produce them: WS-E turns `Candidate`s into `Adjudication`s, WS-F turns a
list of `Diagnosis` into cluster ids. Without these factories, both
workstreams would have to wait on WS-C (detectors) and WS-D (contrastive) to
exist before they could build even one test fixture, for no reason tied to
what they are actually testing. These factories let a test construct a
valid, contract-shaped result directly.

Every factory takes keyword-only arguments, one per field, each defaulted to
a sensible synthetic value. A caller overrides only the field its test cares
about, e.g. `make_signal(step_index=3)`, and gets a fully valid object back
for every other field. This is the same reasoning as Litmus's fixture
helpers, just formalized as named factories here because `Signal` and
`Diagnosis` have enough required fields that hand-writing them per test
would bury the one field a test is actually about.
"""

from datetime import UTC, datetime
from uuid import uuid4

from culprit.signals import (
    Adjudication,
    Diagnosis,
    DivergenceCandidate,
    Evidence,
    Signal,
)
from culprit.taxonomy import FailureClass


def make_signal(
    *,
    detector: str = "synthetic_detector",
    step_index: int = 0,
    span_id: str | None = "span-0",
    severity: float = 0.5,
    category: str = FailureClass.UNKNOWN.value,
    message: str = "synthetic signal for testing",
    evidence: list[Evidence] | None = None,
    error: str | None = None,
) -> Signal:
    """Build a valid `Signal`, the L1 detector output. `evidence` defaults to
    an empty list rather than a shared mutable default, so two calls never
    accidentally alias the same list object."""
    return Signal(
        detector=detector,
        step_index=step_index,
        span_id=span_id,
        severity=severity,
        category=category,
        message=message,
        evidence=list(evidence) if evidence is not None else [],
        error=error,
    )


def make_divergence(
    *,
    step_index: int = 0,
    span_id: str = "span-0",
    divergence_score: float = 0.5,
    cliff_delta: float = 0.3,
    surprisal: float = 1.0,
    profile_surprisal: float = 1.0,
    reference_count: int = 5,
    observed_signature: str = "tool:agent:some_tool:ok",
    expected_signatures: list[tuple[str, float]] | None = None,
    nearest_reference_trace_ids: list[str] | None = None,
    alignment_op: str = "mismatch",
    error: str | None = None,
) -> DivergenceCandidate:
    """Build a valid `DivergenceCandidate`, the L2 contrastive output."""
    return DivergenceCandidate(
        step_index=step_index,
        span_id=span_id,
        divergence_score=divergence_score,
        cliff_delta=cliff_delta,
        surprisal=surprisal,
        profile_surprisal=profile_surprisal,
        reference_count=reference_count,
        observed_signature=observed_signature,
        expected_signatures=(
            list(expected_signatures)
            if expected_signatures is not None
            else [("tool:agent:expected_tool:ok", 0.8)]
        ),
        nearest_reference_trace_ids=(
            list(nearest_reference_trace_ids)
            if nearest_reference_trace_ids is not None
            else ["trace-reference-1"]
        ),
        alignment_op=alignment_op,
        error=error,
    )


def make_diagnosis(
    *,
    diagnosis_id: str | None = None,
    trace_id: str = "trace-synthetic",
    created_at: datetime | None = None,
    root_cause_step_index: int | None = 0,
    root_cause_span_id: str | None = "span-0",
    failure_class: str | None = FailureClass.UNKNOWN.value,
    confidence: float = 0.5,
    calibrated_confidence: float = 0.5,
    abstained: bool = False,
    abstain_reason: str | None = None,
    rationale: str = "synthetic rationale for testing",
    counterfactual: str = "synthetic counterfactual for testing",
    candidates_considered: int = 1,
    signals: list[Signal] | None = None,
    divergences: list[DivergenceCandidate] | None = None,
    adjudications: list[Adjudication] | None = None,
    layer_versions: dict[str, str] | None = None,
    degraded_layers: list[str] | None = None,
    error: str | None = None,
) -> Diagnosis:
    """Build a valid `Diagnosis`. `diagnosis_id` defaults to a fresh uuid4
    per call rather than a fixed string, so a test building several
    diagnoses (e.g. for L5 clustering) gets distinct ids without having to
    invent them."""
    return Diagnosis(
        diagnosis_id=diagnosis_id if diagnosis_id is not None else str(uuid4()),
        trace_id=trace_id,
        created_at=created_at if created_at is not None else datetime.now(UTC),
        root_cause_step_index=root_cause_step_index,
        root_cause_span_id=root_cause_span_id,
        failure_class=failure_class,
        confidence=confidence,
        calibrated_confidence=calibrated_confidence,
        abstained=abstained,
        abstain_reason=abstain_reason,
        rationale=rationale,
        counterfactual=counterfactual,
        candidates_considered=candidates_considered,
        signals=list(signals) if signals is not None else [],
        divergences=list(divergences) if divergences is not None else [],
        adjudications=list(adjudications) if adjudications is not None else [],
        layer_versions=(
            dict(layer_versions)
            if layer_versions is not None
            else {"l1": "0.1.0", "l2": "0.1.0", "l3": "0.1.0"}
        ),
        degraded_layers=list(degraded_layers) if degraded_layers is not None else [],
        error=error,
    )
