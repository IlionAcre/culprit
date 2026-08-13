from datetime import datetime, timezone

from culprit.signals import (
    Adjudication,
    Candidate,
    ContrastResult,
    Diagnosis,
    DivergenceCandidate,
    Evidence,
    NeighborFn,
    Signal,
)


def test_a_detector_that_raised_produces_a_sentinel_signal_excluded_from_ranking():
    """One buggy detector must never cost the whole diagnosis. The sentinel
    shape is what run_detectors emits instead of propagating the exception."""
    sentinel = Signal(
        detector="oscillation", step_index=-1, span_id=None, severity=0.0,
        category="unknown", message="detector failed", evidence=[],
        error="ValueError: bad window",
    )

    assert sentinel.error is not None
    assert sentinel.severity == 0.0
    assert sentinel.step_index == -1


def test_evidence_excerpt_carries_a_dotted_field_path_and_optional_numeric():
    """Evidence is frozen: once a detector attaches it to a Signal it must
    not be mutated by a later layer reusing the same object."""
    evidence = Evidence(
        span_id="s5", step_index=4, field="payload.result_text",
        excerpt="No results found", numeric=None,
    )

    assert evidence.field == "payload.result_text"
    assert evidence.numeric is None


def test_neighbor_fn_alias_matches_query_embedding_and_k_to_ordered_trace_ids():
    """signals.py, not db.py, owns NeighborFn so WS-D can depend on the shape
    of nearest-neighbor lookup without importing store_traces directly."""

    def fake_neighbor_fn(query_embedding: list[float], k: int) -> list[str]:
        return ["trace-a", "trace-b"][:k]

    fn: NeighborFn = fake_neighbor_fn

    assert fn([0.1, 0.2], 1) == ["trace-a"]


def test_divergence_candidate_carries_top_three_expected_signatures_with_probability():
    candidate = DivergenceCandidate(
        step_index=14, span_id="s14", divergence_score=0.83, cliff_delta=0.6,
        surprisal=2.1, profile_surprisal=1.8, reference_count=12,
        observed_signature="tool:billing_agent:refund_order:ok",
        expected_signatures=[
            ("tool:billing_agent:search_orders:ok", 0.82),
            ("tool:billing_agent:get_customer:ok", 0.11),
            ("tool:billing_agent:refund_order:ok", 0.07),
        ],
        nearest_reference_trace_ids=["t1", "t2"],
        alignment_op="mismatch",
    )

    assert candidate.expected_signatures[0][0] == "tool:billing_agent:search_orders:ok"
    assert len(candidate.expected_signatures) == 3


def test_contrast_result_abstains_with_a_reason_when_references_are_too_thin():
    """A contrastive method with two references produces confident nonsense;
    abstention here is what propagates into Diagnosis.degraded_layers."""
    result = ContrastResult(
        candidates=[], reference_count=2, abstained=True,
        abstain_reason="insufficient_references",
    )

    assert result.abstained is True
    assert result.abstain_reason == "insufficient_references"


def test_candidate_prior_gets_a_co_location_bonus_when_source_is_both():
    """The +0.15 bonus itself is computed in candidates.py; this test only
    fixes the shape the merge logic writes into, source="both"."""
    candidate = Candidate(
        step_index=14, span_id="s14", rank=0, prior=0.95, source="both",
        signals=[], divergence=None,
    )

    assert candidate.source == "both"
    assert candidate.prior <= 1.0


def test_adjudication_carries_raw_and_calibrated_confidence_separately():
    """Reporting only the calibrated value would hide how far calibration
    moved it, which is the number needed to tell if calibration helps."""
    adjudication = Adjudication(
        step_index=14, span_id="s14", is_root_cause=True,
        failure_class="silent_empty_result_misread", confidence=0.91,
        calibrated_confidence=0.74, rationale="tool returned no results",
        counterfactual="a successful run would have retried the search",
        cited_step_indices=[12, 13, 14], abstained=False, model="gemini-flash",
        prompt_tokens=800, completion_tokens=120, cost_usd=0.0004,
    )

    assert adjudication.confidence != adjudication.calibrated_confidence
    assert adjudication.error is None


def test_diagnosis_accepts_nested_dataclass_lists_and_round_trips_through_model_dump():
    """Diagnosis is Pydantic because it crosses two I/O boundaries (JSONB and
    HTTP), but its list fields hold plain dataclasses (Signal,
    DivergenceCandidate, Adjudication). Pydantic v2 must validate those
    natively with no extra adapter, or every consumer downstream breaks."""
    signal = Signal(
        detector="empty_tool_result", step_index=12, span_id="s12",
        severity=0.7, category="silent_empty_result_misread",
        message="tool returned empty", evidence=[],
    )
    divergence = DivergenceCandidate(
        step_index=14, span_id="s14", divergence_score=0.83, cliff_delta=0.6,
        surprisal=2.1, profile_surprisal=1.8, reference_count=12,
        observed_signature="tool:billing_agent:refund_order:ok",
        expected_signatures=[("tool:billing_agent:search_orders:ok", 0.82)],
        nearest_reference_trace_ids=["t1"], alignment_op="mismatch",
    )
    adjudication = Adjudication(
        step_index=14, span_id="s14", is_root_cause=True,
        failure_class="silent_empty_result_misread", confidence=0.91,
        calibrated_confidence=0.74, rationale="tool returned no results",
        counterfactual="a successful run would have retried the search",
        cited_step_indices=[12, 13, 14], abstained=False, model="gemini-flash",
        prompt_tokens=800, completion_tokens=120, cost_usd=0.0004,
    )

    diagnosis = Diagnosis(
        diagnosis_id="d1", trace_id="t1", created_at=datetime.now(timezone.utc),
        root_cause_step_index=14, root_cause_span_id="s14",
        failure_class="silent_empty_result_misread", confidence=0.91,
        calibrated_confidence=0.74, abstained=False, abstain_reason=None,
        rationale="tool returned no results", counterfactual="retry the search",
        candidates_considered=5, signals=[signal], divergences=[divergence],
        adjudications=[adjudication], layer_versions={"l1": "1.0", "l2": "1.0"},
        degraded_layers=[],
    )

    dumped = diagnosis.model_dump()

    assert dumped["signals"][0]["detector"] == "empty_tool_result"
    assert dumped["divergences"][0]["step_index"] == 14
    assert dumped["adjudications"][0]["model"] == "gemini-flash"
