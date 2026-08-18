import pytest

from culprit.confidence import (
    DEFAULT_A0,
    DEFAULT_A1,
    DEFAULT_A2,
    DEFAULT_A3,
    DEFAULT_A4,
    DEFAULT_A5,
    DEFAULT_A6,
    DEFAULT_A7,
    DEFAULT_A8,
    DEFAULT_A9,
    agrees_with_l1,
    calibrate_confidence,
    cite_check,
    depth_norm,
    select_diagnosis,
    step_position,
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


# --- fitted default coefficients (richer-feature refit, 2026-08-17) -------

def test_default_coefficients_are_the_richer_feature_refit_values():
    """Fitted by logistic regression against the same 447-row TRAIL/Who&When
    population as I7 (46 positive), on the richer production-viable feature
    set (CLAUDE.md's "L3 adjudication" section has the full fit: sample
    size, cross-validation AUC, and the precision-at-threshold table the
    floor was chosen from). Regression-pins the values so a future edit
    cannot silently drift without a test failure calling it out."""
    assert DEFAULT_A0 == pytest.approx(-4.3970)
    assert DEFAULT_A1 == pytest.approx(-0.0482)
    assert DEFAULT_A2 == pytest.approx(0.2273)
    assert DEFAULT_A3 == pytest.approx(-0.0892)
    assert DEFAULT_A4 == pytest.approx(0.0020)
    assert DEFAULT_A5 == pytest.approx(0.1206)
    assert DEFAULT_A6 == pytest.approx(2.1086)
    assert DEFAULT_A7 == pytest.approx(1.1989)
    assert DEFAULT_A8 == pytest.approx(0.2925)
    assert DEFAULT_A9 == pytest.approx(-1.5935)


def test_default_coefficients_now_weight_step_position_and_depth_over_the_original_four_features():
    """The richer refit's headline finding: once rank, step position, step
    depth, and L1 signal count are in the model, they carry more weight
    than the original four features - including agreement (a3), which came
    back small and slightly negative here, a genuinely different (and less
    flattering) result than I7's fit, where agreement was one of the two
    dominant terms."""
    assert abs(DEFAULT_A6) > abs(DEFAULT_A2)
    assert abs(DEFAULT_A6) > abs(DEFAULT_A3)
    assert abs(DEFAULT_A7) > abs(DEFAULT_A3)


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
# Fixed baseline args for the new terms so each test below isolates the one
# feature it names. rank=1 (top), step_position=0.5, depth_norm=0.3,
# n_l1_signals=1, is_fallback=False are all mid-range/typical values.

_BASE_KW = dict(rank=1, step_position=0.5, depth_norm=0.3, n_l1_signals=1, is_fallback=False)


def test_calibrated_confidence_increases_with_higher_evidence_density():
    low = calibrate_confidence(0.7, 0.5, False, 0.0, **_BASE_KW)
    high = calibrate_confidence(0.7, 0.5, False, 1.0, **_BASE_KW)

    assert high > low


def test_calibrated_confidence_no_longer_increases_with_agreement_under_the_richer_model():
    """The richer refit's a3 (agreement) came back small and slightly
    negative (see the coefficient tests above) - the opposite of I7's fit,
    where agreement was one of the two dominant terms. Once rank, step
    position, depth, and L1 signal count are in the model, matching an L1
    category hint no longer independently helps; this pins that this is a
    real, measured change in direction, not an oversight."""
    without = calibrate_confidence(0.7, 0.5, False, 0.5, **_BASE_KW)
    with_agreement = calibrate_confidence(0.7, 0.5, True, 0.5, **_BASE_KW)

    assert with_agreement < without


def test_calibrated_confidence_increases_with_a_better_rank():
    worse_rank = calibrate_confidence(0.7, 0.5, False, 0.5, rank=5, step_position=0.5,
                                       depth_norm=0.3, n_l1_signals=1, is_fallback=False)
    better_rank = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                        depth_norm=0.3, n_l1_signals=1, is_fallback=False)

    assert better_rank > worse_rank


def test_calibrated_confidence_increases_with_later_step_position():
    earlier = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.0,
                                    depth_norm=0.3, n_l1_signals=1, is_fallback=False)
    later = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=1.0,
                                  depth_norm=0.3, n_l1_signals=1, is_fallback=False)

    assert later > earlier


def test_calibrated_confidence_increases_with_step_depth():
    shallow = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                    depth_norm=0.0, n_l1_signals=1, is_fallback=False)
    deep = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                 depth_norm=1.0, n_l1_signals=1, is_fallback=False)

    assert deep > shallow


def test_calibrated_confidence_increases_with_more_l1_signals():
    fewer = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                  depth_norm=0.3, n_l1_signals=0, is_fallback=False)
    more = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                 depth_norm=0.3, n_l1_signals=5, is_fallback=False)

    assert more > fewer


def test_calibrated_confidence_decreases_for_a_fallback_candidate():
    """A fallback candidate carries zero real evidence by construction
    (candidates.py: no L1 signal, no L2 divergence, prior=0.0) - the fit
    found this is worth flagging on its own, distinctly from prior alone."""
    real = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                 depth_norm=0.3, n_l1_signals=1, is_fallback=False)
    fallback = calibrate_confidence(0.7, 0.5, False, 0.5, rank=1, step_position=0.5,
                                     depth_norm=0.3, n_l1_signals=1, is_fallback=True)

    assert fallback < real


def test_calibrate_confidence_never_raises_at_model_confidence_boundaries():
    """A model reporting exactly 0.0 or 1.0 confidence (some models saturate)
    must not blow up the logit transform at the domain edge."""
    low = calibrate_confidence(0.0, 0.0, False, 0.0, rank=5, step_position=0.0,
                                depth_norm=0.0, n_l1_signals=0, is_fallback=True)
    high = calibrate_confidence(1.0, 1.0, True, 1.0, rank=1, step_position=1.0,
                                 depth_norm=1.0, n_l1_signals=5, is_fallback=False)

    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_calibrate_confidence_stays_within_unit_interval():
    for conf in (0.1, 0.4, 0.6, 0.9):
        value = calibrate_confidence(conf, 0.9, True, 1.0, **_BASE_KW)
        assert 0.0 <= value <= 1.0


# --- step_position / depth_norm -------------------------------------------

def test_step_position_is_zero_at_the_first_step_and_one_at_the_last():
    assert step_position(0, 10) == 0.0
    assert step_position(9, 10) == 1.0
    assert step_position(4, 9) == pytest.approx(0.5)


def test_step_position_defaults_to_zero_for_a_trace_with_no_meaningful_length():
    """A trace of 0 or 1 steps has no position to normalize against - this
    must return 0.0 rather than raise a ZeroDivisionError."""
    assert step_position(3, 1) == 0.0
    assert step_position(3, 0) == 0.0


def test_depth_norm_scales_against_the_deepest_step_in_the_trace():
    assert depth_norm(2, 4) == pytest.approx(0.5)
    assert depth_norm(4, 4) == pytest.approx(1.0)


def test_depth_norm_defaults_to_zero_for_a_flat_trace():
    """A trace with no nesting anywhere (max_depth == 0) has nothing to
    normalize against - this must return 0.0 rather than divide by zero."""
    assert depth_norm(0, 0) == 0.0


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
