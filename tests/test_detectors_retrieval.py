"""unused_retrieval, low_score_retrieval, goal_token_drift."""

from culprit.detectors.base import build_context
from culprit.detectors.retrieval import goal_token_drift, low_score_retrieval, unused_retrieval
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_unused_retrieval_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="unused_retrieval", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in unused_retrieval(ctx))


def test_unused_retrieval_never_fires_on_twenty_clean_runs():
    """Clean retrieval doc counts vary 1-4 per seed; this proves the query
    coverage measure stays robust to that variation rather than only
    working at one doc count."""
    for seed in range(20):
        run = successful_run(seed)
        assert unused_retrieval(_ctx(run)) == []


def test_low_score_retrieval_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="low_score_retrieval", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in low_score_retrieval(ctx))


def test_low_score_retrieval_never_fires_on_twenty_clean_runs():
    """Clean scores are drawn from `rng.uniform(0.55, 0.95)`, comfortably
    above the 0.3 threshold for every seed."""
    for seed in range(20):
        run = successful_run(seed)
        assert low_score_retrieval(_ctx(run)) == []


def test_goal_token_drift_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="goal_token_drift", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in goal_token_drift(ctx))


def test_goal_token_drift_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert goal_token_drift(_ctx(run)) == []
