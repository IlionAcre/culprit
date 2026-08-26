"""Persistence wiring for `run_benchmark` (Integration task I7): before this
module existed, `culprit bench` only ever persisted the trace, never the
`Diagnosis` or the `BenchmarkCase` ground truth, so a bench run left nothing
in Postgres for a later calibration fit to join against (see CLAUDE.md's "L3
adjudication" section). These tests prove the three writes happen, using
fakes only - no real Postgres, matching this project's existing offline
test style for store-layer calls (`tests/test_jobs.py`)."""

import pytest

from culprit import bench
from culprit.benchmarks.base import BenchmarkCase
from culprit.schemas import Outcome, Trace
from culprit.signals import Adjudication
from culprit.synth_results import make_diagnosis


def _trace(trace_id: str) -> Trace:
    return Trace(
        trace_id=trace_id, source="benchmark:fake", outcome=Outcome.FAILURE,
        span_count=0, step_count=0,
    )


def _cases(trace_id: str) -> list[BenchmarkCase]:
    trace = _trace(trace_id)
    return [
        BenchmarkCase(
            case_id=f"{trace_id}-0", trace=trace, spans=[], steps=[],
            ground_truth_step_index=0, ground_truth_span_id=None,
            ground_truth_failure_class="unknown",
        ),
        BenchmarkCase(
            case_id=f"{trace_id}-1", trace=trace, spans=[], steps=[],
            ground_truth_step_index=2, ground_truth_span_id=None,
            ground_truth_failure_class="retrieval_miss", is_primary=False,
        ),
    ]


def _patch_common(monkeypatch, cases, diagnose_fn):
    monkeypatch.setitem(bench.BENCHMARKS, "fake", lambda data: cases)
    monkeypatch.setattr(bench.pipeline_mod, "diagnose", diagnose_fn)


def test_run_benchmark_persists_trace_cases_and_diagnosis_per_trace(monkeypatch):
    """The real gap this task closes: a bench run must call `write_trace`,
    `write_benchmark_cases`, and `write_diagnosis` once per trace, not just
    `write_trace` as the pre-I7 code did."""
    cases = _cases("t1")
    diagnosis = make_diagnosis(trace_id="t1")

    written_traces = []
    written_cases = []
    written_diagnoses = []

    monkeypatch.setattr(bench, "write_trace", lambda conn_fn, trace, spans, steps: written_traces.append(trace.trace_id))
    monkeypatch.setattr(bench, "write_benchmark_cases", lambda conn_fn, benchmark, cs: written_cases.append((benchmark, [c.case_id for c in cs])))
    monkeypatch.setattr(bench, "write_diagnosis", lambda conn_fn, d: written_diagnoses.append(d))
    _patch_common(monkeypatch, cases, lambda trace, **kw: diagnosis)

    run = bench.run_benchmark(
        "fake", data=None, conn_fn=lambda: None,
        call_fn=lambda *a, **kw: ("out", 1.0, 0.0, 10, 5),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        model="fake-model",
    )

    assert written_traces == ["t1"]
    assert written_cases == [("fake", ["t1-0", "t1-1"])]
    assert written_diagnoses == [diagnosis]
    assert run.n_traces == 1


def test_run_benchmark_still_persists_cases_when_diagnose_raises(monkeypatch):
    """Per-trace isolation (bench.py's docstring): a trace whose
    `pipeline.diagnose` call raises must still have had its trace and
    ground-truth cases persisted before the failure, and must not call
    `write_diagnosis` at all for that trace."""
    cases = _cases("t2")

    written_cases = []
    written_diagnoses = []

    monkeypatch.setattr(bench, "write_trace", lambda conn_fn, trace, spans, steps: None)
    monkeypatch.setattr(bench, "write_benchmark_cases", lambda conn_fn, benchmark, cs: written_cases.append([c.case_id for c in cs]))
    monkeypatch.setattr(bench, "write_diagnosis", lambda conn_fn, d: written_diagnoses.append(d))

    def raising_diagnose(trace, **kw):
        raise RuntimeError("boom")

    _patch_common(monkeypatch, cases, raising_diagnose)

    run = bench.run_benchmark(
        "fake", data=None, conn_fn=lambda: None,
        call_fn=lambda *a, **kw: ("out", 1.0, 0.0, 10, 5),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        model="fake-model",
    )

    assert written_cases == [["t2-0", "t2-1"]]
    assert written_diagnoses == []
    assert "t2" in run.per_trace_errors


def test_abstained_diagnosis_with_adjudications_yields_candidate_steps(monkeypatch):
    """Task A's abstention fix: a diagnosis that abstained still ran L3 on a
    real shortlist, so its cases must contribute to candidate recall rather
    than being dropped from the denominator."""
    cases = _cases("t4")
    adjudications = [
        Adjudication(
            step_index=0, span_id="span-0", is_root_cause=False,
            failure_class="unknown", confidence=0.1, calibrated_confidence=0.1,
            rationale="", counterfactual="", cited_step_indices=[],
            abstained=True, model="m", prompt_tokens=None,
            completion_tokens=None, cost_usd=None, error=None,
            source="l1",
        ),
        Adjudication(
            step_index=2, span_id="span-2", is_root_cause=False,
            failure_class="unknown", confidence=0.1, calibrated_confidence=0.1,
            rationale="", counterfactual="", cited_step_indices=[],
            abstained=True, model="m", prompt_tokens=None,
            completion_tokens=None, cost_usd=None, error=None,
            source="l2",
        ),
    ]
    diagnosis = make_diagnosis(
        trace_id="t4", abstained=True, abstain_reason="l3_abstained",
        root_cause_step_index=None, root_cause_span_id=None,
        failure_class=None, adjudications=adjudications,
    )

    monkeypatch.setattr(bench, "write_trace", lambda conn_fn, trace, spans, steps: None)
    monkeypatch.setattr(bench, "write_benchmark_cases", lambda conn_fn, benchmark, cs: None)
    monkeypatch.setattr(bench, "write_diagnosis", lambda conn_fn, d: None)
    monkeypatch.setitem(bench.BENCHMARKS, "fake", lambda data: cases)
    monkeypatch.setattr(bench.pipeline_mod, "diagnose", lambda trace, **kw: diagnosis)

    run = bench.run_benchmark(
        "fake", data=None, conn_fn=lambda: None,
        call_fn=lambda *a, **kw: ("out", 1.0, 0.0, 10, 5),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        model="fake-model",
    )

    assert run.report_all.n_abstained == 2
    assert run.report_all.candidate_recall_at_k[5] == pytest.approx(1.0)
    # No fillers yet, so the evidenced series must agree with the combined one.
    assert run.report_all.evidenced_candidate_recall_at_k == run.report_all.candidate_recall_at_k


def test_run_benchmark_calls_recycle_fn_after_writes_not_before(monkeypatch):
    """`recycle_fn` (see `pooled_conn_fn`'s docstring) must fire only after a
    trace's writes and reads are all finished, so it must still be called
    exactly once per trace even with the new writes added."""
    cases = _cases("t3")
    diagnosis = make_diagnosis(trace_id="t3")
    recycled = []

    monkeypatch.setattr(bench, "write_trace", lambda conn_fn, trace, spans, steps: None)
    monkeypatch.setattr(bench, "write_benchmark_cases", lambda conn_fn, benchmark, cs: None)
    monkeypatch.setattr(bench, "write_diagnosis", lambda conn_fn, d: None)
    _patch_common(monkeypatch, cases, lambda trace, **kw: diagnosis)

    bench.run_benchmark(
        "fake", data=None, conn_fn=lambda: None,
        call_fn=lambda *a, **kw: ("out", 1.0, 0.0, 10, 5),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        model="fake-model",
        recycle_fn=lambda: recycled.append(1),
    )

    assert recycled == [1]


def test_fallback_is_not_counted_as_evidence(monkeypatch):
    """Task E done-when 3: the fallback candidate has no L1/L2 evidence and
    must be excluded from evidenced_candidate_steps, just like filler."""
    cases = _cases("t5")
    adjudications = [
        Adjudication(
            step_index=2, span_id="span-2", is_root_cause=True,
            failure_class="unknown", confidence=0.1, calibrated_confidence=0.1,
            rationale="", counterfactual="", cited_step_indices=[],
            abstained=False, model="m", prompt_tokens=None,
            completion_tokens=None, cost_usd=None, error=None,
            source="fallback",
        ),
        Adjudication(
            step_index=0, span_id="span-0", is_root_cause=False,
            failure_class="unknown", confidence=0.1, calibrated_confidence=0.1,
            rationale="", counterfactual="", cited_step_indices=[],
            abstained=False, model="m", prompt_tokens=None,
            completion_tokens=None, cost_usd=None, error=None,
            source="filler",
        ),
        Adjudication(
            step_index=1, span_id="span-1", is_root_cause=False,
            failure_class="unknown", confidence=0.1, calibrated_confidence=0.1,
            rationale="", counterfactual="", cited_step_indices=[],
            abstained=False, model="m", prompt_tokens=None,
            completion_tokens=None, cost_usd=None, error=None,
            source="filler",
        ),
    ]
    diagnosis = make_diagnosis(
        trace_id="t5", abstained=False,
        root_cause_step_index=2, root_cause_span_id="span-2",
        failure_class="unknown", adjudications=adjudications,
    )

    monkeypatch.setattr(bench, "write_trace", lambda conn_fn, trace, spans, steps: None)
    monkeypatch.setattr(bench, "write_benchmark_cases", lambda conn_fn, benchmark, cs: None)
    monkeypatch.setattr(bench, "write_diagnosis", lambda conn_fn, d: None)
    monkeypatch.setitem(bench.BENCHMARKS, "fake", lambda data: cases)
    monkeypatch.setattr(bench.pipeline_mod, "diagnose", lambda trace, **kw: diagnosis)

    run = bench.run_benchmark(
        "fake", data=None, conn_fn=lambda: None,
        call_fn=lambda *a, **kw: ("out", 1.0, 0.0, 10, 5),
        embed_fn=lambda texts: [[0.0] * 384 for _ in texts],
        model="fake-model",
    )

    assert run.report_all.candidate_recall_at_k[5] == pytest.approx(1.0)
    assert run.report_all.evidenced_candidate_recall_at_k[5] == pytest.approx(0.0)
    assert run.report_all.evidenced_candidate_recall_at_k == {1: 0.0, 3: 0.0, 5: 0.0}
