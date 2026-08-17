import logging
from pathlib import Path

from culprit.benchmarks.trail import TRAIL_CATEGORY_MAP, load_cases
from culprit.taxonomy import FailureClass

_FIXTURE = Path(__file__).parent / "fixtures" / "benchmarks" / "trail_sample.json"


def test_load_cases_returns_one_case_per_annotation():
    # trail-001 and trail-002 carry one error each; trail-003 carries two,
    # so three records become four cases.
    cases = load_cases(_FIXTURE)
    assert len(cases) == 4
    assert {c.trace.trace_id for c in cases} == {
        "benchmark:trail:trail-001", "benchmark:trail:trail-002", "benchmark:trail:trail-003",
    }


def test_cases_carry_canonical_trace_spans_and_steps_with_benchmark_source():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-001")
    assert case.trace.source == "benchmark:trail"
    # 4 spans incl. the non-semantic root; 3 steps after linearize folds it.
    assert case.trace.span_count == 4
    assert case.trace.step_count == 3
    assert len(case.spans) == 4
    assert len(case.steps) == 3
    assert case.is_primary is True


def test_error_span_maps_to_correct_step_index_and_known_category():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-001")
    # steps: 0 = CodeAgent.run, 1 = LiteLLMModel.__call__, 2 = web_search (annotated)
    assert case.ground_truth_step_index == 2
    assert case.ground_truth_span_id == "aa00000000000004"
    assert case.ground_truth_failure_class == FailureClass.WRONG_TOOL_SELECTED.value


def test_error_on_the_llm_span_maps_to_the_malformed_tool_input_class():
    cases = load_cases(_FIXTURE)
    case = next(c for c in cases if c.trace.trace_id == "benchmark:trail:trail-002")
    # the non-semantic process_item root folds into the single LLM step
    assert case.ground_truth_step_index == 0
    assert case.ground_truth_failure_class == FailureClass.MALFORMED_TOOL_INPUT.value


def test_multi_error_trace_becomes_one_case_per_annotation_sorted_earliest_first():
    cases = load_cases(_FIXTURE)
    own = [c for c in cases if c.trace.trace_id == "benchmark:trail:trail-003"]
    assert len(own) == 2
    assert own[0].is_primary is True
    assert own[1].is_primary is False
    # category normalization: strip + case + hyphen/space collapse
    assert own[0].ground_truth_failure_class == FailureClass.PLAN_OMISSION.value


def test_unmapped_external_category_becomes_unknown_with_logged_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="culprit"):
        cases = load_cases(_FIXTURE)
    case = next(
        c for c in cases
        if c.trace.trace_id == "benchmark:trail:trail-003" and not c.is_primary
    )
    assert case.ground_truth_failure_class == FailureClass.UNKNOWN.value
    assert any("unmapped TRAIL error category" in r.message for r in caplog.records)


def test_category_map_is_an_explicit_reviewable_dict_not_computed():
    assert isinstance(TRAIL_CATEGORY_MAP, dict)
    assert TRAIL_CATEGORY_MAP["tool selection errors"] == FailureClass.WRONG_TOOL_SELECTED
    assert all(isinstance(v, FailureClass) for v in TRAIL_CATEGORY_MAP.values())


def test_malformed_record_is_skipped_without_crashing_the_whole_run(tmp_path):
    import json

    good = json.loads(_FIXTURE.read_text(encoding="utf-8"))[0]
    bad = {"trace_id": "trail-broken", "spans": "not a list", "errors": []}
    broken_fixture = tmp_path / "broken.json"
    broken_fixture.write_text(json.dumps([good, bad]), encoding="utf-8")

    cases = load_cases(broken_fixture)

    assert len(cases) == 1
    assert cases[0].trace.trace_id == "benchmark:trail:trail-001"
