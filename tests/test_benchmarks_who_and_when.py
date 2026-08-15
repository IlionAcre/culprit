from pathlib import Path

from culprit.benchmarks.who_and_when import load_cases
from culprit.schemas import SpanKind
from culprit.taxonomy import FailureClass

_FIXTURE = Path(__file__).parent / "fixtures" / "benchmarks" / "who_and_when_sample.json"


def test_load_cases_returns_one_case_per_record_for_the_3_record_sample():
    cases = load_cases(_FIXTURE)
    assert len(cases) == 3
    assert {c.trace.trace_id for c in cases} == {
        "benchmark:who_and_when:ww-001", "benchmark:who_and_when:ww-002", "benchmark:who_and_when:ww-003",
    }


def test_cases_carry_canonical_trace_spans_and_steps_with_benchmark_source():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:who_and_when:ww-001")
    assert case.trace.source == "benchmark:who_and_when"
    # 5 messages + 1 synthetic CHAIN root = 6 spans, but the root folds into
    # step 0's collapsed_span_ids rather than getting its own step.
    assert case.trace.span_count == 6
    assert case.trace.step_count == 5
    assert len(case.steps) == 5
    assert case.is_primary is True


def test_mistake_step_maps_directly_to_ground_truth_step_index_because_already_linear():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:who_and_when:ww-001")
    assert case.ground_truth_step_index == 2
    assert case.ground_truth_span_id == case.steps[2].span_id


def test_synthetic_root_is_folded_into_step_zero_not_its_own_step():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:who_and_when:ww-001")
    root_id = f"ww-001-root"
    assert root_id in case.steps[0].collapsed_span_ids
    assert case.steps[0].step_index == 0


def test_kind_refinement_from_content_markers():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:who_and_when:ww-001")
    # message 0: plain reasoning -> AGENT
    assert case.steps[0].kind == SpanKind.AGENT
    # message 1: search_web(...) -> RETRIEVER
    assert case.steps[1].kind == SpanKind.RETRIEVER
    # message 3: send_email(...) -> TOOL
    assert case.steps[3].kind == SpanKind.TOOL


def test_ground_truth_failure_class_is_always_unknown_no_labeled_taxonomy():
    cases = load_cases(_FIXTURE)
    assert all(c.ground_truth_failure_class == FailureClass.UNKNOWN.value for c in cases)


def test_malformed_record_is_skipped_without_crashing_the_whole_run(tmp_path):
    import json

    good = json.loads(_FIXTURE.read_text())[0]
    bad = {"case_id": "ww-broken", "messages": [{"agent": "a", "content": "hi"}], "mistake_step": 99}
    broken_fixture = tmp_path / "broken.json"
    broken_fixture.write_text(json.dumps([good, bad]))

    cases = load_cases(broken_fixture)

    assert len(cases) == 1
    assert cases[0].trace.trace_id == "benchmark:who_and_when:ww-001"
