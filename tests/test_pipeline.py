"""Integration task I1: proves pipeline.diagnose actually wires layers
together rather than only compiling against their signatures, and proves
the per-layer isolation mechanism CLAUDE.md calls the honesty mechanism of
the whole system - a layer that raises degrades the diagnosis into
`degraded_layers` instead of crashing the run. Wired incrementally
(L0+L1, then L2, then L3); each layer's own tests monkeypatch the layers
below it to a clean result so a given test only exercises the layer it
names, matching the isolation the production code itself provides.
"""

import json

from culprit.pipeline import diagnose
from culprit.signals import ContrastResult, Diagnosis, DivergenceCandidate
from culprit.synth import successful_run
from culprit.synth_inject import inject
from culprit.taxonomy import FailureClass


def _fake_call_fn(model, prompt):
    return ("{}", 0.0, 0.0)


def _fake_embed_fn(texts):
    return [[0.0] * 384 for _ in texts]


def _clean_contrast(*args, **kwargs) -> ContrastResult:
    """A non-abstained, zero-candidate L2 result: used to isolate tests that
    are about L0/L1 wiring from L2's own real execution, which needs a real
    Postgres connection this test suite never has (conn_fn=lambda: None).
    L2's own wiring/isolation gets its own tests below."""
    return ContrastResult(candidates=[], reference_count=5, abstained=False, abstain_reason=None)


def test_diagnose_wires_l0_and_l1_and_finds_the_injected_signal(monkeypatch):
    """empty_tool_result is one of the four detectors CLAUDE.md calls "the
    point of the product": the trace looks healthy at that step and only
    surfaces the damage later. Injecting it and getting a matching Signal
    back out of `diagnose` proves L0 (read_trace) and L1 (run_detectors)
    are really wired, not just individually testable in isolation."""
    base = successful_run(seed=1)
    run, step_index = inject(base, "empty_tool_result", at_step=1)

    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (run.trace, run.spans, run.steps)
    )
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    diagnosis = diagnose(
        run.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.trace_id == run.trace.trace_id
    assert diagnosis.degraded_layers == []
    assert any(
        s.detector == "empty_tool_result" and s.step_index == step_index for s in diagnosis.signals
    )


def test_diagnose_degrades_l0_instead_of_crashing_when_read_trace_raises(monkeypatch):
    """The isolation contract at the trace level: a failure loading L0's
    persisted output must still produce a Diagnosis, with 'l0' recorded in
    degraded_layers, never an exception escaping to the caller."""
    base = successful_run(seed=2)

    def _boom(conn_fn, trace_id):
        raise RuntimeError("connection pool exhausted")

    monkeypatch.setattr("culprit.pipeline.read_trace", _boom)
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.degraded_layers == ["l0"]
    assert diagnosis.signals == []


def test_diagnose_degrades_l1_instead_of_crashing_when_run_detectors_raises(monkeypatch):
    """Same isolation contract, one layer down: L0 succeeds, L1 raises, the
    diagnosis still comes back with 'l1' in degraded_layers rather than an
    exception costing the whole run."""
    base = successful_run(seed=3)

    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )

    def _boom(trace, steps, spans_by_id):
        raise ValueError("detector catalogue exploded")

    monkeypatch.setattr("culprit.pipeline.run_detectors", _boom)
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.degraded_layers == ["l1"]
    assert diagnosis.signals == []


def _fake_divergence(step_index: int = 5) -> DivergenceCandidate:
    return DivergenceCandidate(
        step_index=step_index, span_id=f"s{step_index}", divergence_score=0.7,
        cliff_delta=0.5, surprisal=1.0, profile_surprisal=1.0, reference_count=10,
        observed_signature="tool:agent:refund_order:ok",
        expected_signatures=[("tool:agent:search_orders:ok", 0.8)],
        nearest_reference_trace_ids=["t1"], alignment_op="mismatch",
    )


def test_diagnose_wires_l2_and_carries_divergences_through_when_contrast_succeeds(monkeypatch):
    """Proves contrast() is really called (not skipped) and its candidates
    flow into Diagnosis.divergences untouched."""
    base = successful_run(seed=5)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )
    divergence = _fake_divergence()
    monkeypatch.setattr(
        "culprit.pipeline.contrast",
        lambda *a, **kw: ContrastResult(candidates=[divergence], reference_count=10, abstained=False, abstain_reason=None),
    )

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.degraded_layers == []
    assert diagnosis.divergences == [divergence]


def test_diagnose_degrades_l2_with_the_abstain_reason_when_contrast_abstains(monkeypatch):
    """A designed abstention (too few references) is exactly as materially
    weaker a diagnosis as a crash - CLAUDE.md's honesty mechanism - so it
    must land in degraded_layers too, carrying the reason, not just an
    empty divergences list with no explanation."""
    base = successful_run(seed=6)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )
    monkeypatch.setattr(
        "culprit.pipeline.contrast",
        lambda *a, **kw: ContrastResult(candidates=[], reference_count=1, abstained=True, abstain_reason="insufficient_references"),
    )

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.degraded_layers == ["l2:insufficient_references"]
    assert diagnosis.divergences == []


def test_diagnose_degrades_l2_instead_of_crashing_when_contrast_raises(monkeypatch):
    base = successful_run(seed=7)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )

    def _boom(*a, **kw):
        raise RuntimeError("alignment blew up")

    monkeypatch.setattr("culprit.pipeline.contrast", _boom)

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.degraded_layers == ["l2"]
    assert diagnosis.divergences == []


def _verdict_call_fn(step_index: int):
    def _call(model, prompt):
        raw = json.dumps({
            "is_root_cause": True,
            "failure_class": FailureClass.SILENT_EMPTY_RESULT_MISREAD.value,
            "confidence": 0.95,
            "rationale": f"step {step_index} returned an empty result the agent treated as success",
            "counterfactual": "a successful run would have retried or surfaced the empty result",
            "cited_step_indices": [step_index],
        })
        return raw, 1.0, 0.001
    return _call


def test_diagnose_wires_l3_and_surfaces_a_committed_root_cause(monkeypatch):
    """End-to-end proof that L1's signal, L2's clean-empty result, and L3's
    adjudication are really chained through merge_candidates -> adjudicate
    -> select_diagnosis, landing a real, non-abstained root cause."""
    base = successful_run(seed=8)
    run, step_index = inject(base, "empty_tool_result", at_step=1)

    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (run.trace, run.spans, run.steps)
    )
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    diagnosis = diagnose(
        run.trace, conn_fn=lambda: None, call_fn=_verdict_call_fn(step_index), embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.degraded_layers == []
    assert diagnosis.abstained is False
    assert diagnosis.root_cause_step_index == step_index
    assert diagnosis.failure_class == FailureClass.SILENT_EMPTY_RESULT_MISREAD.value
    assert diagnosis.candidates_considered >= 1
    assert len(diagnosis.adjudications) >= 1


def test_diagnose_degrades_l3_instead_of_crashing_when_merge_candidates_raises(monkeypatch):
    base = successful_run(seed=9)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    def _boom(*a, **kw):
        raise ValueError("candidate merge exploded")

    monkeypatch.setattr("culprit.pipeline.merge_candidates", _boom)

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.degraded_layers == ["l3"]
    assert diagnosis.adjudications == []
    assert diagnosis.abstained is True


def test_diagnose_records_layer_versions_and_abstains_with_no_adjudications_yet(monkeypatch):
    """L2/L3 are not wired in this revision, so with zero adjudications the
    trace-level verdict must abstain rather than fabricate a root cause -
    proves select_diagnosis is really consulted, not bypassed."""
    base = successful_run(seed=4)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )
    monkeypatch.setattr("culprit.pipeline.contrast", _clean_contrast)

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.abstained is True
    assert diagnosis.root_cause_step_index is None
    assert diagnosis.layer_versions["l1"]
