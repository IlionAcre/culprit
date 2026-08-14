import pytest

from culprit.adjudicate import adjudicate
from culprit.cluster import cluster_diagnoses
from culprit.contrast import contrast
from culprit.pipeline import diagnose
from culprit.run_detectors import run_detectors
from culprit.synth_results import make_diagnosis, make_divergence, make_signal


def test_factories_produce_valid_contract_objects_without_running_any_layer():
    """WS-E and WS-F build entirely against these, so they never wait on
    WS-C or WS-D to exist."""
    signal = make_signal(detector="empty_tool_result", step_index=3)
    diagnosis = make_diagnosis(root_cause_step_index=3)

    assert signal.step_index == 3
    assert diagnosis.root_cause_step_index == 3


def test_make_signal_defaults_every_field_a_caller_does_not_override():
    """A test concerned with one field should not have to construct a fully
    populated Signal to get one."""
    signal = make_signal()

    assert signal.detector == "synthetic_detector"
    assert signal.severity == 0.5
    assert signal.evidence == []
    assert signal.error is None


def test_make_signal_evidence_default_is_not_a_shared_mutable_list():
    """Two independent calls must never end up aliasing the same evidence
    list, or a test mutating one signal's evidence would corrupt another."""
    first = make_signal()
    second = make_signal()

    first.evidence.append("not-real-evidence")

    assert second.evidence == []


def test_make_divergence_overrides_only_the_field_the_caller_names():
    divergence = make_divergence(step_index=14, divergence_score=0.83)

    assert divergence.step_index == 14
    assert divergence.divergence_score == 0.83
    assert divergence.reference_count == 5  # untouched default
    assert len(divergence.expected_signatures) >= 1


def test_make_diagnosis_defaults_give_a_fully_valid_object():
    diagnosis = make_diagnosis()

    assert diagnosis.diagnosis_id
    assert diagnosis.abstained is False
    assert diagnosis.signals == []
    assert diagnosis.divergences == []
    assert diagnosis.adjudications == []
    assert diagnosis.layer_versions


def test_make_diagnosis_generates_a_distinct_id_per_call_unless_overridden():
    """Building several diagnoses for a clustering test (WS-F) must not
    require the caller to invent unique ids by hand."""
    first = make_diagnosis()
    second = make_diagnosis()

    assert first.diagnosis_id != second.diagnosis_id


def test_make_diagnosis_accepts_nested_signal_and_divergence_overrides():
    signal = make_signal(step_index=3)
    divergence = make_divergence(step_index=3)

    diagnosis = make_diagnosis(
        root_cause_step_index=3, signals=[signal], divergences=[divergence]
    )

    assert diagnosis.signals[0].step_index == 3
    assert diagnosis.divergences[0].step_index == 3


def test_contrast_is_implemented_and_abstains_cleanly_with_no_references():
    """WS-D landed (unlike run_detectors/adjudicate/cluster below, still
    Phase 1 stubs): `contrast()` no longer raises `NotImplementedError`.
    An empty neighbor pool is exactly the "insufficient_references"
    abstention path, not a crash - `trace=None` is safe here only because
    zero steps short-circuits before `trace` is ever read."""
    result = contrast(None, [], neighbor_fn=lambda v, k: [], conn_fn=lambda: None)

    assert result.abstained
    assert result.abstain_reason == "empty_trace"


def test_run_detectors_returns_empty_on_an_empty_trace():
    """WS-C's stub raised NotImplementedError here during Phase 0; now that
    WS-C is implemented, this just proves the entrypoint handles the
    trivial empty-trace case gracefully. Real detector coverage lives in
    tests/test_run_detectors.py and tests/test_detectors_*.py."""
    from culprit.synth import successful_run

    trace = successful_run(seed=0).trace
    assert run_detectors(trace, [], {}) == []


def test_adjudicate_stub_names_ws_e():
    with pytest.raises(NotImplementedError, match="WS-E"):
        adjudicate([], None, [], call_fn=lambda model, prompt: ("", 0.0, 0.0), model="m")


def test_cluster_diagnoses_stub_names_ws_f():
    with pytest.raises(NotImplementedError, match="WS-F"):
        cluster_diagnoses([], embed_fn=lambda texts: [])


def test_pipeline_diagnose_stub_names_integration_task():
    """pipeline.py is Integration-owned, not a WS stub, so its message names
    the integration task rather than a workstream letter."""
    with pytest.raises(NotImplementedError, match="Integration"):
        diagnose(
            None,
            conn_fn=lambda: None,
            call_fn=lambda model, prompt: ("", 0.0, 0.0),
            embed_fn=lambda texts: [],
        )
