from datetime import UTC, datetime

from culprit.synth_results import make_diagnosis, make_divergence, make_signal
from culprit.taxonomy import FailureClass
from culprit.views import (
    adjudication_view,
    diagnosis_detail_view,
    diagnosis_summary_view,
    divergence_view,
    evidence_view,
    job_status_view,
    signal_view,
)
from culprit.signals import Adjudication, Evidence


def test_signal_view_selects_exactly_the_documented_fields():
    """Hand-picked shaping, not asdict(): asserting the exact key set is
    what catches a field silently leaking into (or vanishing from) the API
    response if Signal's field list changes later."""
    evidence = Evidence(span_id="s1", step_index=2, field="payload.x", excerpt="e", numeric=1.0)
    signal = make_signal(step_index=3, evidence=[evidence])

    view = signal_view(signal)

    assert set(view.keys()) == {
        "detector", "step_index", "span_id", "severity", "category",
        "message", "evidence", "error",
    }
    assert view["step_index"] == 3
    assert view["evidence"] == [evidence_view(evidence)]


def test_divergence_view_selects_exactly_the_documented_fields():
    divergence = make_divergence(step_index=7)

    view = divergence_view(divergence)

    assert set(view.keys()) == {
        "step_index", "span_id", "divergence_score", "cliff_delta", "surprisal",
        "profile_surprisal", "reference_count", "observed_signature",
        "expected_signatures", "nearest_reference_trace_ids", "alignment_op", "error",
    }
    assert view["step_index"] == 7


def test_adjudication_view_selects_exactly_the_documented_fields():
    adjudication = Adjudication(
        step_index=1, span_id="s1", source="filler", is_root_cause=True,
        failure_class=FailureClass.TOOL_FAILURE_UNHANDLED.value, confidence=0.9,
        calibrated_confidence=0.7, rationale="r", counterfactual="c",
        cited_step_indices=[1, 2], abstained=False, model="m",
        prompt_tokens=10, completion_tokens=5, cost_usd=0.001,
    )

    view = adjudication_view(adjudication)

    assert set(view.keys()) == {
        "step_index", "span_id", "source", "is_root_cause", "failure_class",
        "confidence", "calibrated_confidence", "rationale", "counterfactual",
        "cited_step_indices", "abstained", "model", "cost_usd", "error",
    }
    assert view["source"] == "filler"
    # prompt_tokens/completion_tokens are deliberately excluded from the
    # response shape (internal cost accounting detail), proving this is a
    # hand-picked view rather than every Adjudication field.
    assert "prompt_tokens" not in view


def test_diagnosis_summary_view_omits_nested_signals_divergences_adjudications():
    """The list-view shape must stay small: no nested evidence trail."""
    diagnosis = make_diagnosis(signals=[make_signal()], divergences=[make_divergence()])

    view = diagnosis_summary_view(diagnosis)

    assert "signals" not in view
    assert "divergences" not in view
    assert "adjudications" not in view
    assert view["diagnosis_id"] == diagnosis.diagnosis_id


def test_diagnosis_detail_view_includes_every_nested_signal_and_divergence():
    signal = make_signal(step_index=1)
    divergence = make_divergence(step_index=1)
    diagnosis = make_diagnosis(signals=[signal], divergences=[divergence])

    view = diagnosis_detail_view(diagnosis)

    assert view["signals"] == [signal_view(signal)]
    assert view["divergences"] == [divergence_view(divergence)]
    assert view["rationale"] == diagnosis.rationale


def test_two_diagnoses_differing_only_by_id_still_differ_only_in_id_fields():
    """A sanity check that the view is a pure projection, not accidentally
    stable across genuinely different diagnoses.

    created_at is pinned to one shared timestamp rather than each
    make_diagnosis call defaulting its own `datetime.now(UTC)`: two
    sequential calls almost never land on the exact same microsecond, so
    the un-pinned version of this test was flaky - it failed on the
    (usual) case where the timestamps differed, and only passed on the
    rare tick-collision where they didn't. Pinning it makes the test
    assert what it actually means: id fields are the only thing that
    should differ here."""
    created_at = datetime.now(UTC)
    d1 = make_diagnosis(diagnosis_id="d1", trace_id="t1", created_at=created_at)
    d2 = make_diagnosis(diagnosis_id="d2", trace_id="t1", created_at=created_at)

    v1, v2 = diagnosis_summary_view(d1), diagnosis_summary_view(d2)

    assert v1["diagnosis_id"] != v2["diagnosis_id"]
    assert {k: v for k, v in v1.items() if k != "diagnosis_id"} == {
        k: v for k, v in v2.items() if k != "diagnosis_id"
    }


def test_diagnosis_detail_view_surfaces_candidate_source_on_every_adjudication():
    """Done-when 5: CLI/API consumers can tell which adjudications came from
    evidence-free fillers versus real L1/L2 signals."""
    l1 = Adjudication(
        step_index=0, span_id="s0", source="l1", is_root_cause=False,
        failure_class=FailureClass.UNKNOWN.value, confidence=0.5,
        calibrated_confidence=0.3, rationale="r", counterfactual="c",
        cited_step_indices=[], abstained=True, model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.0001,
    )
    filler = Adjudication(
        step_index=1, span_id="s1", source="filler", is_root_cause=False,
        failure_class=FailureClass.UNKNOWN.value, confidence=0.5,
        calibrated_confidence=0.1, rationale="r", counterfactual="c",
        cited_step_indices=[], abstained=True, model="m",
        prompt_tokens=1, completion_tokens=1, cost_usd=0.0001,
    )
    diagnosis = make_diagnosis(adjudications=[l1, filler])

    view = diagnosis_detail_view(diagnosis)

    sources = [a["source"] for a in view["adjudications"]]
    assert sources == ["l1", "filler"]


def test_job_status_view_passes_through_the_documented_fields():
    status = {"job_id": "j1", "status": "finished", "result": "d1", "error": None}

    view = job_status_view(status)

    assert view == status
