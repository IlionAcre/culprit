"""Integration task I1: proves pipeline.diagnose actually wires layers
together rather than only compiling against their signatures, and proves
the per-layer isolation mechanism CLAUDE.md calls the honesty mechanism of
the whole system - a layer that raises degrades the diagnosis into
`degraded_layers` instead of crashing the run. Wired incrementally (L0+L1
first); tests for L2/L3 land alongside their own wiring.
"""

from culprit.pipeline import diagnose
from culprit.signals import Diagnosis
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _fake_call_fn(model, prompt):
    return ("{}", 0.0, 0.0)


def _fake_embed_fn(texts):
    return [[0.0] * 384 for _ in texts]


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

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert isinstance(diagnosis, Diagnosis)
    assert diagnosis.degraded_layers == ["l1"]
    assert diagnosis.signals == []


def test_diagnose_records_layer_versions_and_abstains_with_no_adjudications_yet(monkeypatch):
    """L2/L3 are not wired in this revision, so with zero adjudications the
    trace-level verdict must abstain rather than fabricate a root cause -
    proves select_diagnosis is really consulted, not bypassed."""
    base = successful_run(seed=4)
    monkeypatch.setattr(
        "culprit.pipeline.read_trace", lambda conn_fn, trace_id: (base.trace, base.spans, base.steps)
    )

    diagnosis = diagnose(
        base.trace, conn_fn=lambda: None, call_fn=_fake_call_fn, embed_fn=_fake_embed_fn, model="m",
    )

    assert diagnosis.abstained is True
    assert diagnosis.root_cause_step_index is None
    assert diagnosis.layer_versions["l1"]
