import logging
from pathlib import Path

from culprit.benchmarks.trail import TRAIL_CATEGORY_MAP, load_cases
from culprit.taxonomy import FailureClass

_FIXTURE = Path(__file__).parent / "fixtures" / "benchmarks" / "trail_sample.json"


def test_load_cases_returns_one_case_per_record_for_the_3_record_sample():
    cases = load_cases(_FIXTURE)
    assert len(cases) == 3
    assert {c.trace.trace_id for c in cases} == {
        "benchmark:trail:trail-001", "benchmark:trail:trail-002", "benchmark:trail:trail-003",
    }


def test_cases_carry_canonical_trace_spans_and_steps_with_benchmark_source():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-001")
    assert case.trace.source == "benchmark:trail"
    assert case.trace.span_count == 3
    assert case.trace.step_count == 3
    assert len(case.spans) == 3
    assert len(case.steps) == 3
    assert case.is_primary is True


def test_error_span_maps_to_correct_step_index_and_known_category():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-001")
    # steps: 0 = invoke_agent, 1 = chat, 2 = execute_tool (the annotated error span)
    assert case.ground_truth_step_index == 2
    assert case.ground_truth_span_id == "tr001a3"
    assert case.ground_truth_failure_class == FailureClass.WRONG_TOOL_SELECTED.value


def test_error_on_the_llm_span_maps_to_the_malformed_tool_input_class():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-002")
    assert case.ground_truth_step_index == 1
    assert case.ground_truth_failure_class == FailureClass.MALFORMED_TOOL_INPUT.value


def test_unmapped_external_category_becomes_unknown_with_logged_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="culprit"):
        cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-003")
    assert case.ground_truth_failure_class == FailureClass.UNKNOWN.value
    assert any("unmapped TRAIL error category" in r.message for r in caplog.records)


def test_category_map_is_an_explicit_reviewable_dict_not_computed():
    assert isinstance(TRAIL_CATEGORY_MAP, dict)
    assert TRAIL_CATEGORY_MAP["wrong_tool_call"] == FailureClass.WRONG_TOOL_SELECTED
    assert all(isinstance(v, FailureClass) for v in TRAIL_CATEGORY_MAP.values())


def test_malformed_record_is_skipped_without_crashing_the_whole_run(tmp_path):
    import json

    good = json.loads(_FIXTURE.read_text())[0]
    bad = {"trace_id": "trail-broken", "resourceSpans": "not a list", "errors": []}
    broken_fixture = tmp_path / "broken.json"
    broken_fixture.write_text(json.dumps([good, bad]))

    cases = load_cases(broken_fixture)

    assert len(cases) == 1
    assert cases[0].trace.trace_id == "benchmark:trail:trail-001"
