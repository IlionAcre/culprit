"""The WS-C definition of done, proven directly: every one of the twenty L1
detectors fires on its matching `synth_inject` injection, and none of them
fire on any of twenty clean synthetic successful runs. CLAUDE.md's "L1
detectors" section is explicit that the negative result matters as much as
the positive one, so both live in this one file rather than being scattered
per detector family.

`seed=1` is used as the "rich" base run throughout: it is the smallest seed
whose run includes both optional blocks (`check_duplicate_refund` and
`verify_eligibility`), giving every injector the most step diversity to work
with (4 TOOL steps, 2 optional LLM/TOOL pairs) so a true-positive test never
accidentally depends on a thin trace.
"""

import pytest

from culprit.run_detectors import run_detectors
from culprit.schemas import Span
from culprit.synth import SynthRun, successful_run
from culprit.synth_inject import INJECTION_KINDS, inject
from culprit.detectors.registry import DETECTORS

_CLEAN_SEEDS = range(20)
_RICH_SEED = 1  # has both check_duplicate_refund and verify_eligibility


def _spans_by_id(run: SynthRun) -> dict[str, Span]:
    return {s.span_id: s for s in run.spans}


def test_every_injection_kind_has_a_detector():
    """The correspondence the rest of this file leans on: a detector name and
    its `synth_inject` kind are the same string, so the loop below can check
    "did the matching detector fire"."""
    assert INJECTION_KINDS <= set(DETECTORS)


def test_detectors_without_a_synthetic_injection_are_declared():
    """A detector with no synth injection gets no automatic false-positive
    check here, so each one is listed deliberately and carries that check in
    its own test module. `instruction_noncompliance` is validated against
    real TRAIL shapes in `tests/test_detectors_compliance.py`."""
    assert set(DETECTORS) - INJECTION_KINDS == {"instruction_noncompliance", "reasoning_turn_defect"}


@pytest.mark.parametrize("kind", sorted(INJECTION_KINDS))
def test_every_detector_fires_on_its_matching_injection_at_the_ground_truth_step(kind):
    base = successful_run(seed=_RICH_SEED)
    mutated, ground_truth = inject(base, kind=kind, at_step=3)

    signals = run_detectors(mutated.trace, mutated.steps, _spans_by_id(mutated))

    matches = [s for s in signals if s.detector == kind and s.step_index == ground_truth]
    assert matches, (
        f"{kind} did not fire at ground-truth step {ground_truth}; "
        f"got {[(s.detector, s.step_index) for s in signals]}"
    )
    assert all(s.error is None for s in matches)


def test_no_detector_fires_on_twenty_clean_synthetic_successful_runs():
    """The negative result CLAUDE.md calls the load-bearing one: a noisy
    detector costs a candidate slot that should have gone to the real
    cause. Every detector must stay silent across a whole sweep of clean
    seeds, not just avoid firing on the one seed its true-positive test
    happens to use."""
    offenders = []
    for seed in _CLEAN_SEEDS:
        run = successful_run(seed)
        signals = run_detectors(run.trace, run.steps, _spans_by_id(run))
        offenders.extend((seed, s.detector, s.step_index, s.message) for s in signals)

    assert offenders == []


def test_run_detectors_covers_all_twenty_registered_detectors():
    base = successful_run(seed=_RICH_SEED)
    mutated, _gt = inject(base, kind="tool_error", at_step=3)
    signals = run_detectors(mutated.trace, mutated.steps, _spans_by_id(mutated))
    # every signal's detector name must be a real registered detector
    assert {s.detector for s in signals} <= set(DETECTORS)


def test_a_raising_detector_produces_exactly_one_sentinel_signal_and_the_rest_still_run(monkeypatch):
    """Per-detector error isolation: one buggy detector must never cost the
    whole diagnosis. Monkeypatch a single registry entry to raise, and
    confirm the other nineteen still ran and returned real signals for an
    injected trace, while the broken one shows up as exactly one sentinel
    `Signal` with `error` set."""
    def _boom(ctx):
        raise ValueError("synthetic detector failure for isolation testing")

    monkeypatch.setitem(DETECTORS, "tool_error", _boom)

    base = successful_run(seed=_RICH_SEED)
    mutated, ground_truth = inject(base, kind="empty_tool_result", at_step=3)
    signals = run_detectors(mutated.trace, mutated.steps, _spans_by_id(mutated))

    sentinels = [s for s in signals if s.detector == "tool_error"]
    assert len(sentinels) == 1
    assert sentinels[0].error is not None
    assert "ValueError" in sentinels[0].message
    assert sentinels[0].step_index == -1
    assert sentinels[0].span_id is None

    # the other nineteen detectors still ran: empty_tool_result still fires
    healthy = [s for s in signals if s.detector == "empty_tool_result" and s.step_index == ground_truth]
    assert healthy
    assert all(s.error is None for s in healthy)

    # only the monkeypatched detector produced an error sentinel
    assert [s.detector for s in signals if s.error is not None] == ["tool_error"]
