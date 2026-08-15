from culprit.confidence import (
    agrees_with_l1,
    calibrate_confidence,
    cite_check,
    select_diagnosis,
)
from culprit.signals import Adjudication


def _adjudication(
    step_index: int, *, is_root_cause: bool = True, calibrated_confidence: float = 0.8,
    failure_class: str = "tool_failure_unhandled", abstained: bool = False, error: str | None = None,
) -> Adjudication:
    return Adjudication(
        step_index=step_index, span_id=f"span-{step_index}", is_root_cause=is_root_cause,
        failure_class=failure_class, confidence=calibrated_confidence,
        calibrated_confidence=calibrated_confidence, rationale="r", counterfactual="c",
        cited_step_indices=[step_index], abstained=abstained, model="m",
        prompt_tokens=None, completion_tokens=None, cost_usd=None, error=error,
    )


# --- cite_check --------------------------------------------------------

def test_cite_check_returns_one_when_every_citation_is_visible():
    assert cite_check([3, 4], frozenset({1, 2, 3, 4, 5})) == 1.0


def test_cite_check_returns_the_grounded_fraction_when_some_citations_are_out_of_window():
    """Direct evidence of confabulation: citing step 47 when only 12-16 were
    shown must measurably reduce the density, not just flag it binary."""
    assert cite_check([3, 47], frozenset({1, 2, 3, 4, 5})) == 0.5


def test_cite_check_returns_zero_when_no_citation_is_visible():
    assert cite_check([47, 99], frozenset({1, 2, 3})) == 0.0


def test_cite_check_returns_zero_for_an_empty_citation_list():
    """An unsupported rationale earns no credit, the same as a confabulated
    one - the term rewards checkable evidence, and there is none to check."""
    assert cite_check([], frozenset({1, 2, 3})) == 0.0


# --- calibrate_confidence -----------------------------------------------

def test_calibrated_confidence_increases_with_higher_evidence_density():
    low = calibrate_confidence(0.7, prior=0.5, agreement=False, evidence_density=0.0)
    high = calibrate_confidence(0.7, prior=0.5, agreement=False, evidence_density=1.0)

    assert high > low


def test_calibrated_confidence_increases_with_agreement():
    without = calibrate_confidence(0.7, prior=0.5, agreement=False, evidence_density=0.5)
    with_agreement = calibrate_confidence(0.7, prior=0.5, agreement=True, evidence_density=0.5)

    assert with_agreement > without


def test_calibrate_confidence_never_raises_at_model_confidence_boundaries():
    """A model reporting exactly 0.0 or 1.0 confidence (some models saturate)
    must not blow up the logit transform at the domain edge."""
    low = calibrate_confidence(0.0, prior=0.0, agreement=False, evidence_density=0.0)
    high = calibrate_confidence(1.0, prior=1.0, agreement=True, evidence_density=1.0)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_calibrate_confidence_stays_within_unit_interval():
    for conf in (0.1, 0.4, 0.6, 0.9):
        value = calibrate_confidence(conf, prior=0.9, agreement=True, evidence_density=1.0)
        assert 0.0 <= value <= 1.0


# --- agrees_with_l1 -------------------------------------------------------

def test_agrees_with_l1_true_when_the_chosen_class_matches_a_signal_category():
    assert agrees_with_l1(["tool_failure_unhandled", "unknown"], "tool_failure_unhandled") is True


def test_agrees_with_l1_false_when_no_category_matches():
    assert agrees_with_l1(["unknown"], "tool_failure_unhandled") is False


def test_agrees_with_l1_false_when_there_are_no_l1_categories_at_all():
    assert agrees_with_l1([], "tool_failure_unhandled") is False


# --- select_diagnosis: the three abstention gates, one test each ----------

def test_gate_one_self_abstention_blocks_a_candidate_that_denies_being_root_cause():
    """is_root_cause=False is a correct, expected answer, not a failure to
    find one - it must never get silently promoted to a runner-up verdict."""
    adjudications = [_adjudication(3, is_root_cause=False, calibrated_confidence=0.95)]

    verdict = select_diagnosis(adjudications)

    assert verdict.abstained is True
    assert verdict.winner is None
    assert "root cause" in verdict.abstain_reason


def test_gate_two_confidence_floor_blocks_a_low_confidence_verdict():
    adjudications = [_adjudication(3, calibrated_confidence=0.40)]

    verdict = select_diagnosis(adjudications, min_confidence=0.55)

    assert verdict.abstained is True
    assert verdict.winner is None
    assert "floor" in verdict.abstain_reason


def test_gate_three_ambiguity_margin_blocks_two_close_candidates_of_different_classes():
    adjudications = [
        _adjudication(3, calibrated_confidence=0.80, failure_class="tool_failure_unhandled"),
        _adjudication(7, calibrated_confidence=0.76, failure_class="context_loss"),
    ]

    verdict = select_diagnosis(adjudications, ambiguity_margin=0.08)

    assert verdict.abstained is True
    assert verdict.winner is None
    assert "ambiguous" in verdict.abstain_reason


def test_gate_three_does_not_fire_when_close_candidates_agree_on_failure_class():
    """A close margin alone is not ambiguity - the gate only fires when the
    top two also disagree on what kind of failure it was."""
    adjudications = [
        _adjudication(3, calibrated_confidence=0.80, failure_class="tool_failure_unhandled"),
        _adjudication(7, calibrated_confidence=0.76, failure_class="tool_failure_unhandled"),
    ]

    verdict = select_diagnosis(adjudications, ambiguity_margin=0.08)

    assert verdict.abstained is False
    assert verdict.winner.step_index == 3


def test_select_diagnosis_returns_the_winner_when_all_gates_pass():
    adjudications = [
        _adjudication(3, calibrated_confidence=0.90, failure_class="tool_failure_unhandled"),
        _adjudication(7, calibrated_confidence=0.50, failure_class="context_loss"),
    ]

    verdict = select_diagnosis(adjudications)

    assert verdict.abstained is False
    assert verdict.abstain_reason is None
    assert verdict.winner.step_index == 3


def test_select_diagnosis_excludes_sentinel_and_error_adjudications_from_eligibility():
    adjudications = [
        _adjudication(3, abstained=True, error="boom", calibrated_confidence=0.99),
        _adjudication(7, calibrated_confidence=0.80),
    ]

    verdict = select_diagnosis(adjudications)

    assert verdict.winner.step_index == 7
