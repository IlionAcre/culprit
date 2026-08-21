"""Output shaping for the FastAPI and Typer surfaces: hand-picked field
selection per response shape, not a blanket `model_dump()`. Litmus's own
convention (see Litmus/CLAUDE.md) is deliberate here too: a field added to
`Diagnosis`, `Signal`, `DivergenceCandidate`, or `Adjudication` later must
not silently change what an API response or CLI print statement contains
until someone decides it should. Exists as its own module so `api.py` and
`cli.py` stay under the project's module size ceiling (see CLAUDE.md).
"""

from culprit.signals import Adjudication, Diagnosis, DivergenceCandidate, Evidence, Signal


def evidence_view(evidence: Evidence) -> dict:
    return {
        "span_id": evidence.span_id,
        "step_index": evidence.step_index,
        "field": evidence.field,
        "excerpt": evidence.excerpt,
        "numeric": evidence.numeric,
    }


def signal_view(signal: Signal) -> dict:
    return {
        "detector": signal.detector,
        "step_index": signal.step_index,
        "span_id": signal.span_id,
        "severity": signal.severity,
        "category": signal.category,
        "message": signal.message,
        "evidence": [evidence_view(e) for e in signal.evidence],
        "error": signal.error,
    }


def divergence_view(divergence: DivergenceCandidate) -> dict:
    return {
        "step_index": divergence.step_index,
        "span_id": divergence.span_id,
        "divergence_score": divergence.divergence_score,
        "cliff_delta": divergence.cliff_delta,
        "surprisal": divergence.surprisal,
        "profile_surprisal": divergence.profile_surprisal,
        "reference_count": divergence.reference_count,
        "observed_signature": divergence.observed_signature,
        "expected_signatures": [list(pair) for pair in divergence.expected_signatures],
        "nearest_reference_trace_ids": list(divergence.nearest_reference_trace_ids),
        "alignment_op": divergence.alignment_op,
        "error": divergence.error,
    }


def adjudication_view(adjudication: Adjudication) -> dict:
    return {
        "step_index": adjudication.step_index,
        "span_id": adjudication.span_id,
        "source": adjudication.source,
        "is_root_cause": adjudication.is_root_cause,
        "failure_class": adjudication.failure_class,
        "confidence": adjudication.confidence,
        "calibrated_confidence": adjudication.calibrated_confidence,
        "rationale": adjudication.rationale,
        "counterfactual": adjudication.counterfactual,
        "cited_step_indices": list(adjudication.cited_step_indices),
        "abstained": adjudication.abstained,
        "model": adjudication.model,
        "cost_usd": adjudication.cost_usd,
        "error": adjudication.error,
    }


def diagnosis_summary_view(diagnosis: Diagnosis) -> dict:
    """List-view shape: enough to identify and triage a diagnosis without
    its full nested evidence trail. Used by `GET /traces/{id}/diagnoses`."""
    return {
        "diagnosis_id": diagnosis.diagnosis_id,
        "trace_id": diagnosis.trace_id,
        "created_at": diagnosis.created_at.isoformat(),
        "root_cause_step_index": diagnosis.root_cause_step_index,
        "failure_class": diagnosis.failure_class,
        "calibrated_confidence": diagnosis.calibrated_confidence,
        "abstained": diagnosis.abstained,
        "candidates_considered": diagnosis.candidates_considered,
        "degraded_layers": list(diagnosis.degraded_layers),
    }


def diagnosis_detail_view(diagnosis: Diagnosis) -> dict:
    """Full single-diagnosis shape: every field plus each nested signal,
    divergence, and adjudication with its own evidence."""
    return {
        **diagnosis_summary_view(diagnosis),
        "root_cause_span_id": diagnosis.root_cause_span_id,
        "confidence": diagnosis.confidence,
        "abstain_reason": diagnosis.abstain_reason,
        "rationale": diagnosis.rationale,
        "counterfactual": diagnosis.counterfactual,
        "layer_versions": dict(diagnosis.layer_versions),
        "signals": [signal_view(s) for s in diagnosis.signals],
        "divergences": [divergence_view(d) for d in diagnosis.divergences],
        "adjudications": [adjudication_view(a) for a in diagnosis.adjudications],
        "error": diagnosis.error,
    }


def job_status_view(status: dict) -> dict:
    """Passthrough shaping for `queue.fetch_job_status`'s dict, kept as its
    own function (rather than inlined at each call site) so a field added
    to that dict later goes through one reviewed place, same as every
    other view here."""
    return {
        "job_id": status["job_id"],
        "status": status["status"],
        "result": status["result"],
        "error": status["error"],
    }
