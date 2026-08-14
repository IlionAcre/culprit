"""tool_error, the loud sibling of the tool_errors.py family.
`empty_tool_result` and `error_swallowed` get heavy flagship coverage in
test_detectors_flagship.py."""

from culprit.detectors.base import build_context
from culprit.detectors.tool_errors import tool_error
from culprit.schemas import SpanStatus
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_tool_error_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="tool_error", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in tool_error(ctx))


def test_tool_error_flagged_span_actually_has_error_status():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="tool_error", at_step=3)
    assert mutated.spans[gt].status == SpanStatus.ERROR
    assert mutated.spans[gt].payload.is_error is True


def test_tool_error_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert tool_error(_ctx(run)) == []
