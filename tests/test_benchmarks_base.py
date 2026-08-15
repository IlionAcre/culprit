from culprit.benchmarks.base import BenchmarkCase
from culprit.schemas import Outcome, Trace


def _trace() -> Trace:
    return Trace(
        trace_id="t1", source="benchmark:trail", outcome=Outcome.FAILURE,
        span_count=0, step_count=0,
    )


def test_benchmark_case_defaults_is_primary_true():
    case = BenchmarkCase(
        case_id="c1", trace=_trace(), spans=[], steps=[],
        ground_truth_step_index=0, ground_truth_span_id=None,
        ground_truth_failure_class="unknown",
    )
    assert case.is_primary is True


def test_benchmark_case_can_mark_a_non_primary_row_for_multi_error_traces():
    case = BenchmarkCase(
        case_id="c2", trace=_trace(), spans=[], steps=[],
        ground_truth_step_index=5, ground_truth_span_id="s5",
        ground_truth_failure_class="retrieval_miss", is_primary=False,
    )
    assert case.is_primary is False
