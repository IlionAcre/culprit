"""oscillation, repeated_identical_action, retry_storm."""

from culprit.detectors.base import build_context
from culprit.detectors.loops import oscillation, repeated_identical_action, retry_storm
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_oscillation_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="oscillation", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in oscillation(ctx))


def test_oscillation_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert oscillation(_ctx(run)) == []


def test_repeated_identical_action_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="repeated_identical_action", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in repeated_identical_action(ctx))


def test_repeated_identical_action_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert repeated_identical_action(_ctx(run)) == []


def test_repeated_identical_action_does_not_fire_on_a_zero_progress_retry_storm():
    """A run with identical results too is retry_storm's stronger case; the
    two detectors must not double-fire on the same steps."""
    base = successful_run(seed=1)
    mutated, _gt = inject(base, kind="retry_storm", at_step=3)
    ctx = _ctx(mutated)
    assert repeated_identical_action(ctx) == []


def test_retry_storm_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="retry_storm", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in retry_storm(ctx))


def test_retry_storm_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert retry_storm(_ctx(run)) == []


def test_retry_storm_does_not_fire_on_a_progressing_repeated_action():
    """A run whose repeats produce different results each time is
    repeated_identical_action's weaker case, not zero-progress."""
    base = successful_run(seed=1)
    mutated, _gt = inject(base, kind="repeated_identical_action", at_step=3)
    ctx = _ctx(mutated)
    assert retry_storm(ctx) == []
