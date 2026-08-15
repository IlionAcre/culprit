import pytest

from culprit.bench_score import CaseResult, layer_ablation_delta, score


def test_score_empty_results_returns_zeroed_report_not_a_crash():
    report = score([])
    assert report.n_cases == 0
    assert report.exact_accuracy == 0.0
    assert report.tolerance_accuracy == {0: 0.0, 1: 0.0, 3: 0.0}
    assert report.candidate_recall_at_k == {1: 0.0, 3: 0.0, 5: 0.0}


def test_score_hand_worked_example_from_the_plan():
    """The literal example fixed in the plan's definition of done: predictions
    [3, 7, 2] against truth [3, 8, 2]. Exact accuracy 2/3, tolerance@1 3/3,
    and earliness error ((3-3) + (7-8) + (2-2)) / 3 == -0.333... The negative
    sign is the point: it proves predicted-minus-truth (blaming a symptom
    downstream is positive), not truth-minus-predicted."""
    results = [
        CaseResult(truth_step=3, predicted_step=3),
        CaseResult(truth_step=8, predicted_step=7),
        CaseResult(truth_step=2, predicted_step=2),
    ]

    report = score(results)

    assert report.exact_accuracy == pytest.approx(2 / 3)
    assert report.tolerance_accuracy[0] == pytest.approx(2 / 3)
    assert report.tolerance_accuracy[1] == pytest.approx(1.0)
    assert report.tolerance_accuracy[3] == pytest.approx(1.0)
    assert report.earliness_error == pytest.approx(-1 / 3, abs=1e-3)
    assert report.earliness_error == pytest.approx(-0.333, abs=1e-3)


def test_positive_earliness_means_blaming_a_downstream_symptom():
    """Sign-convention regression test, independent of the plan's own
    example: predicting consistently later than truth must yield a positive
    earliness_error, never negative. Flipping this sign is the single most
    damaging bug this harness could ship, since it is the headline number."""
    results = [CaseResult(truth_step=1, predicted_step=4), CaseResult(truth_step=2, predicted_step=6)]
    report = score(results)
    assert report.earliness_error > 0


def test_abstained_case_excluded_from_earliness_but_counted_in_abstention_rate():
    results = [
        CaseResult(truth_step=1, predicted_step=1),
        CaseResult(truth_step=5, predicted_step=None),
    ]
    report = score(results)
    assert report.n_abstained == 1
    assert report.abstention_rate == pytest.approx(0.5)
    assert report.earliness_error == pytest.approx(0.0)  # only the committed case counts
    assert report.accuracy_conditional_on_commit == pytest.approx(1.0)
    assert report.exact_accuracy == pytest.approx(0.5)  # abstained case counts as a miss overall


def test_span_and_class_accuracy_none_when_no_case_supplies_them():
    results = [CaseResult(truth_step=1, predicted_step=1)]
    report = score(results)
    assert report.span_accuracy is None
    assert report.class_accuracy is None
    assert report.brier_score is None
    assert report.ece is None


def test_class_accuracy_and_confusion_matrix_and_joint_accuracy():
    results = [
        CaseResult(truth_step=1, predicted_step=1, truth_class="tool_failure_unhandled",
                   predicted_class="tool_failure_unhandled"),
        CaseResult(truth_step=2, predicted_step=2, truth_class="retrieval_miss",
                   predicted_class="context_loss"),
    ]
    report = score(results)
    assert report.class_accuracy == pytest.approx(0.5)
    assert report.confusion_matrix[("tool_failure_unhandled", "tool_failure_unhandled")] == 1
    assert report.confusion_matrix[("retrieval_miss", "context_loss")] == 1
    assert report.joint_accuracy == pytest.approx(0.5)  # only case 1 matches on both step and class


def test_candidate_recall_at_k_excludes_cases_with_no_shortlist():
    results = [
        CaseResult(truth_step=5, predicted_step=5, candidate_steps=[5, 1, 2]),
        CaseResult(truth_step=9, predicted_step=1, candidate_steps=[1, 2, 3, 4, 9]),
        CaseResult(truth_step=0, predicted_step=0),  # no shortlist measured, excluded
    ]
    report = score(results)
    assert report.candidate_recall_at_k[1] == pytest.approx(0.5)  # only case 1 has truth at rank 1
    assert report.candidate_recall_at_k[3] == pytest.approx(0.5)  # case 2's truth (9) is rank 5, misses @3
    assert report.candidate_recall_at_k[5] == pytest.approx(1.0)  # both hit within top 5


def test_calibration_perfect_confidence_yields_zero_brier_and_ece():
    results = [
        CaseResult(truth_step=1, predicted_step=1, predicted_confidence=1.0),
        CaseResult(truth_step=2, predicted_step=99, predicted_confidence=0.0),
    ]
    report = score(results)
    assert report.brier_score == pytest.approx(0.0)
    assert report.ece == pytest.approx(0.0)
    assert len(report.reliability_points) == 2


def test_layer_ablation_delta_reports_negative_deltas_too():
    with_layer = score([CaseResult(truth_step=1, predicted_step=1)])
    without_layer = score([CaseResult(truth_step=1, predicted_step=1)])
    delta = layer_ablation_delta(with_layer, without_layer)
    assert delta == {"exact_accuracy_delta": 0.0, "joint_accuracy_delta": 0.0, "recall_at_5_delta": 0.0}


def test_layer_ablation_delta_is_directional_not_absolute():
    better = score([CaseResult(truth_step=1, predicted_step=1)])
    worse = score([CaseResult(truth_step=1, predicted_step=99)])
    delta = layer_ablation_delta(better, worse)
    assert delta["exact_accuracy_delta"] == pytest.approx(1.0)
    delta_reversed = layer_ablation_delta(worse, better)
    assert delta_reversed["exact_accuracy_delta"] == pytest.approx(-1.0)
