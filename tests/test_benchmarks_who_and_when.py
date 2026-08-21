import json
from pathlib import Path

from culprit.benchmarks.who_and_when import load_cases
from culprit.schemas import SpanKind, SpanStatus, ToolPayload
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


def _single_message_case(tmp_path, content: str):
    """Build a one-message Who&When fixture and return the lone case."""
    record = {
        "case_id": "ww-exitcode",
        "task_goal": "test exitcode classification",
        "messages": [{"agent": "terminal", "content": content}],
        "mistake_agent": "terminal",
        "mistake_step": 0,
    }
    fixture = tmp_path / "exitcode.json"
    fixture.write_text(json.dumps([record]))
    cases = load_cases(fixture)
    assert len(cases) == 1
    return cases[0]


def test_exitcode_success_turn_is_classified_as_tool(tmp_path):
    case = _single_message_case(tmp_path, "exitcode: 0 (execution succeeded)\nCode output: 42")
    step = case.steps[0]
    span = case.spans[1]
    assert step.kind == SpanKind.TOOL
    payload = span.payload
    assert isinstance(payload, ToolPayload)
    assert payload.is_error is False
    assert payload.result_text == "42"
    assert payload.error_message is None
    assert payload.result_len == len(payload.result_text)
    assert span.status == SpanStatus.OK


def test_exitcode_failure_turn_captures_traceback_as_error_message(tmp_path):
    content = (
        "exitcode: 1 (execution failed)\n"
        "Code output: Traceback (most recent call last):\n"
        '  File "x.py", line 1, in <module>\n'
        "    raise ValueError('bad')\n"
        "ValueError: bad"
    )
    case = _single_message_case(tmp_path, content)
    step = case.steps[0]
    span = case.spans[1]
    assert step.kind == SpanKind.TOOL
    payload = span.payload
    assert isinstance(payload, ToolPayload)
    assert payload.is_error is True
    assert "Traceback" in payload.error_message
    assert payload.result_len == len(payload.result_text)
    assert span.status == SpanStatus.ERROR


def test_exitcode_124_timeout_turn_is_classified_as_tool_error(tmp_path):
    content = "exitcode: 124 (execution failed)\nCode output: Command timed out after 30s"
    case = _single_message_case(tmp_path, content)
    step = case.steps[0]
    span = case.spans[1]
    assert step.kind == SpanKind.TOOL
    payload = span.payload
    assert isinstance(payload, ToolPayload)
    assert payload.is_error is True
    assert "timed out" in payload.error_message.lower()
    assert payload.result_len == len(payload.result_text)
    assert span.status == SpanStatus.ERROR


def test_ordinary_reasoning_turn_still_classifies_as_agent(tmp_path):
    case = _single_message_case(tmp_path, "I will think about this.")
    assert case.steps[0].kind == SpanKind.AGENT
