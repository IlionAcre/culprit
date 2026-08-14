"""context_overflow, step_budget_exhausted. `silent_history_truncation`
(same module) gets its own heavy coverage in test_detectors_flagship.py."""

from culprit.detectors.base import build_context
from culprit.detectors.context import context_overflow, step_budget_exhausted
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_context_overflow_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="context_overflow", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in context_overflow(ctx))


def test_context_overflow_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert context_overflow(_ctx(run)) == []


def test_step_budget_exhausted_fires_at_the_step_before_the_padding_run():
    """`at_step=3` lands the injector's cut on a TOOL step whose signature
    differs from the appended "plan" padding, so the run-length-encoded
    padding run starts cleanly after it rather than absorbing it."""
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="step_budget_exhausted", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in step_budget_exhausted(ctx))


def test_step_budget_exhausted_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert step_budget_exhausted(_ctx(run)) == []
