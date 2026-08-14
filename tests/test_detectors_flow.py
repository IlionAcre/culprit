"""premature_termination, missing_verification, duplicate_delegation."""

from culprit.detectors.base import build_context
from culprit.detectors.flow import duplicate_delegation, missing_verification, premature_termination
from culprit.schemas import Outcome
from culprit.synth import successful_run
from culprit.synth_inject import inject


def _ctx(mutated):
    spans_by_id = {s.span_id: s for s in mutated.spans}
    return build_context(mutated.trace, mutated.steps, spans_by_id)


def test_premature_termination_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="premature_termination", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in premature_termination(ctx))


def test_premature_termination_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert premature_termination(_ctx(run)) == []


def test_missing_verification_fires_at_ground_truth():
    base = successful_run(seed=1)  # has verify_eligibility to begin with
    mutated, gt = inject(base, kind="missing_verification", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in missing_verification(ctx))


def test_missing_verification_never_fires_on_twenty_clean_runs():
    """Roughly half of clean successful runs legitimately skip verification
    too (see synth.py's independent 50% blocks), so this sweeps all twenty
    to prove the outcome gate, not the tool-presence check alone, is what
    keeps it silent."""
    for seed in range(20):
        run = successful_run(seed)
        assert missing_verification(_ctx(run)) == []


def test_missing_verification_requires_the_failure_outcome_gate():
    """Structurally identical to a legitimate clean run that never included
    verification (see the final report): only `trace.outcome == FAILURE`
    distinguishes "verification not needed" from "verification skipped and
    that broke the run". Forging a SUCCESS outcome onto the same mutated
    steps must not fire."""
    base = successful_run(seed=1)
    mutated, _gt = inject(base, kind="missing_verification", at_step=3)
    success_trace = mutated.trace.model_copy(update={"outcome": Outcome.SUCCESS})
    ctx = build_context(success_trace, mutated.steps, {s.span_id: s for s in mutated.spans})
    assert missing_verification(ctx) == []


def test_duplicate_delegation_fires_at_ground_truth():
    base = successful_run(seed=1)
    mutated, gt = inject(base, kind="duplicate_delegation", at_step=3)
    ctx = _ctx(mutated)
    assert any(s.step_index == gt for s in duplicate_delegation(ctx))


def test_duplicate_delegation_never_fires_on_twenty_clean_runs():
    for seed in range(20):
        run = successful_run(seed)
        assert duplicate_delegation(_ctx(run)) == []
