"""stall_timeout."""

from culprit.detectors.base import build_context
from culprit.detectors.timing import stall_timeout
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_stall_timeout_fires_at_ground_truth():
    """Uses seed=1 (4 TOOL steps: search_orders, check_duplicate_refund,
    verify_eligibility, process_refund) rather than a thinner seed. The
    in-trace median this detector falls back to (no cross-trace history is
    available at L1; see timing.py's docstring) needs at least a few
    same-kind samples so a single stalled step cannot drag the median up
    high enough to hide itself - a real limitation of the in-trace proxy,
    called out in the final report."""
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="stall_timeout", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in stall_timeout(ctx))


def test_stall_timeout_never_fires_on_twenty_clean_runs():
    """Every clean step has an identical fixed duration (1500ms) by
    construction, so the median-ratio check is safe regardless of how many
    TOOL steps a given seed happens to produce."""
    for seed in range(20):
        run = successful_run(seed)
        assert stall_timeout(_ctx(run)) == []
