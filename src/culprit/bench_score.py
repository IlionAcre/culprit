"""Scores a benchmark run: one `CaseResult` per `BenchmarkCase` (see
`benchmarks/base.py`), pairing the pipeline's prediction against the fixed
ground truth, into a `BenchReport`.

**Sign convention, load-bearing.** `earliness_error` is the mean of
`predicted_step - truth_step` over committed (non-abstained) cases. Positive
means the system is on average blaming a symptom *downstream* of the true
cause; negative means it is blaming something *upstream* (or exactly on it,
at zero). That is precisely the failure mode culprit exists to eliminate, so
a flipped sign here would make the headline metric lie in the most damaging
possible direction. Verified against the hand-worked example from the plan:
predictions `[3, 7, 2]` against truth `[3, 8, 2]` gives
`((3-3) + (7-8) + (2-2)) / 3 == -0.333...` (`test_bench_score.py`).

**Every tolerance band is reported, never only the best one** (`@0`, `@1`,
`@3`): a harness that only surfaces `tolerance_accuracy[3]` because it looks
best is doing the exact thing Nubank's "9 Tips From the Trenches" warns
against - reporting the metric you like instead of the one that is true.

**`candidate_recall_at_k` is the most load-bearing field on `BenchReport`.**
End-to-end accuracy conflates two failure modes with unrelated fixes: L1/L2
narrowing never offered the true step as a candidate at all (a recall
problem, fixed by tuning the narrowing layers), versus L3 saw the right
shortlist and picked wrong anyway (a ranking/prompting problem, fixed in
adjudication). Only `candidate_steps` on `CaseResult` can tell them apart.
"""

import logging
import math
from dataclasses import dataclass, field

from culprit.logging_config import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)

_TOLERANCE_LEVELS = (0, 1, 3)
_RECALL_K_LEVELS = (1, 3, 5)
_CALIBRATION_BINS = 10


@dataclass
class CaseResult:
    """One scored case: a `BenchmarkCase`'s ground truth paired with whatever
    the pipeline actually produced for it. Every prediction field defaults to
    "no answer given" so a caller scoring only step-level accuracy (the
    `[3, 7, 2]` vs `[3, 8, 2]` example) never has to fill in class/confidence/
    candidate fields it has no opinion about."""

    truth_step: int
    predicted_step: int | None = None  # None means the run abstained
    truth_span_id: str | None = None
    predicted_span_id: str | None = None
    truth_class: str | None = None
    predicted_class: str | None = None
    predicted_confidence: float | None = None  # Adjudication.calibrated_confidence
    # The ranked step-index shortlist L1/L2 actually handed to L3, before
    # adjudication picked among them. Empty means "not measured for this
    # case", excluded from candidate_recall_at_k rather than counted as 0.
    candidate_steps: list[int] = field(default_factory=list)
    # Subset of candidate_steps whose source is not "filler"; tracked
    # separately so the evidenced (L1/L2-driven) recall series stays visible
    # after Task B adds filler padding to the shortlist.
    evidenced_candidate_steps: list[int] = field(default_factory=list)


@dataclass
class BenchReport:
    n_cases: int
    n_abstained: int
    abstention_rate: float
    exact_accuracy: float
    tolerance_accuracy: dict[int, float]  # {0: ..., 1: ..., 3: ...}
    accuracy_conditional_on_commit: float  # exact accuracy among non-abstained cases only
    span_accuracy: float | None  # None when no case supplied span ids
    class_accuracy: float | None  # None when no case supplied a truth class
    confusion_matrix: dict[tuple[str, str], int]  # (truth_class, predicted_class) -> count
    joint_accuracy: float  # tolerance@0 AND class match, the headline number
    earliness_error: float  # signed mean(predicted - truth) over committed cases; see module docstring
    brier_score: float | None  # None when no case supplied a confidence
    ece: float | None  # expected calibration error
    reliability_points: list[tuple[float, float, int]]  # (mean_confidence, accuracy, count) per bin
    candidate_recall_at_k: dict[int, float]  # {1: ..., 3: ..., 5: ...}; excludes cases with no candidates
    evidenced_candidate_recall_at_k: dict[int, float]  # same, but excluding filler-sourced candidates


def _safe_div(numerator: float, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def score(results: list[CaseResult]) -> BenchReport:
    """Per-item error isolation: a single malformed `CaseResult` (e.g. a
    `candidate_steps` list that is present but empty, or a missing
    confidence) never aborts the run, it is simply excluded from the metrics
    that need the field it lacks. There is nothing here to `raise` on, by
    construction - every field is optional and every metric degrades to
    `None`/an empty dict rather than crashing on a partial benchmark record."""
    n = len(results)
    if n == 0:
        return BenchReport(
            n_cases=0, n_abstained=0, abstention_rate=0.0, exact_accuracy=0.0,
            tolerance_accuracy={k: 0.0 for k in _TOLERANCE_LEVELS},
            accuracy_conditional_on_commit=0.0, span_accuracy=None, class_accuracy=None,
            confusion_matrix={}, joint_accuracy=0.0, earliness_error=0.0,
            brier_score=None, ece=None, reliability_points=[],
            candidate_recall_at_k={k: 0.0 for k in _RECALL_K_LEVELS},
            evidenced_candidate_recall_at_k={k: 0.0 for k in _RECALL_K_LEVELS},
        )

    committed = [r for r in results if r.predicted_step is not None]
    n_abstained = n - len(committed)

    exact_hits = sum(1 for r in committed if r.predicted_step == r.truth_step)
    tolerance_accuracy = {
        tol: _safe_div(sum(1 for r in committed if abs(r.predicted_step - r.truth_step) <= tol), n)
        for tol in _TOLERANCE_LEVELS
    }
    accuracy_conditional_on_commit = _safe_div(exact_hits, len(committed))

    span_pairs = [r for r in results if r.truth_span_id is not None]
    span_accuracy = (
        _safe_div(sum(1 for r in span_pairs if r.predicted_span_id == r.truth_span_id), len(span_pairs))
        if span_pairs else None
    )

    class_pairs = [r for r in results if r.truth_class is not None]
    class_accuracy = (
        _safe_div(sum(1 for r in class_pairs if r.predicted_class == r.truth_class), len(class_pairs))
        if class_pairs else None
    )
    confusion_matrix: dict[tuple[str, str], int] = {}
    for r in class_pairs:
        if r.predicted_class is not None:
            key = (r.truth_class, r.predicted_class)
            confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

    joint_hits = sum(
        1 for r in results
        if r.predicted_step is not None and r.truth_class is not None
        and abs(r.predicted_step - r.truth_step) <= 0 and r.predicted_class == r.truth_class
    )
    joint_accuracy = _safe_div(joint_hits, n)

    earliness_error = (
        sum(r.predicted_step - r.truth_step for r in committed) / len(committed) if committed else 0.0
    )

    brier_score, ece, reliability_points = _calibration(results)
    candidate_recall_at_k = _candidate_recall(results)
    evidenced_candidate_recall_at_k = _evidenced_candidate_recall(results)

    return BenchReport(
        n_cases=n, n_abstained=n_abstained, abstention_rate=_safe_div(n_abstained, n),
        exact_accuracy=_safe_div(exact_hits, n), tolerance_accuracy=tolerance_accuracy,
        accuracy_conditional_on_commit=accuracy_conditional_on_commit,
        span_accuracy=span_accuracy, class_accuracy=class_accuracy,
        confusion_matrix=confusion_matrix, joint_accuracy=joint_accuracy,
        earliness_error=earliness_error, brier_score=brier_score, ece=ece,
        reliability_points=reliability_points, candidate_recall_at_k=candidate_recall_at_k,
        evidenced_candidate_recall_at_k=evidenced_candidate_recall_at_k,
    )


def _calibration(results: list[CaseResult]) -> tuple[float | None, float | None, list[tuple[float, float, int]]]:
    """Brier score, expected calibration error, and per-bin reliability
    points, all against the same binary "was the committed prediction exactly
    right" outcome. `Diagnosis` carries both raw and `calibrated_confidence`
    precisely so this can be measured rather than assumed (CLAUDE.md: shipped
    as hand-set coefficients, documented as uncalibrated priors, not fitted
    values - this function is what would supply the fit)."""
    scored = [
        (r.predicted_confidence, 1.0 if r.predicted_step == r.truth_step else 0.0)
        for r in results
        if r.predicted_step is not None and r.predicted_confidence is not None
    ]
    if not scored:
        return None, None, []

    brier_score = sum((conf - outcome) ** 2 for conf, outcome in scored) / len(scored)

    bins: list[list[tuple[float, float]]] = [[] for _ in range(_CALIBRATION_BINS)]
    for conf, outcome in scored:
        idx = min(int(conf * _CALIBRATION_BINS), _CALIBRATION_BINS - 1)
        bins[idx].append((conf, outcome))

    ece = 0.0
    reliability_points: list[tuple[float, float, int]] = []
    for bucket in bins:
        if not bucket:
            continue
        mean_conf = sum(c for c, _ in bucket) / len(bucket)
        mean_acc = sum(o for _, o in bucket) / len(bucket)
        reliability_points.append((mean_conf, mean_acc, len(bucket)))
        ece += (len(bucket) / len(scored)) * abs(mean_conf - mean_acc)

    return brier_score, ece, reliability_points


def _candidate_recall(results: list[CaseResult]) -> dict[int, float]:
    """recall@k: fraction of cases where `truth_step` appears among the first
    k entries of `candidate_steps`, the ranked shortlist L1/L2 actually
    offered L3. Cases that never supplied a shortlist are excluded from the
    denominator rather than counted as misses, matching every other optional
    metric's degrade-gracefully rule."""
    measured = [r for r in results if r.candidate_steps]
    recall: dict[int, float] = {}
    for k in _RECALL_K_LEVELS:
        if not measured:
            recall[k] = 0.0
            continue
        hits = sum(1 for r in measured if r.truth_step in r.candidate_steps[:k])
        recall[k] = hits / len(measured)
    return recall


def _evidenced_candidate_recall(results: list[CaseResult]) -> dict[int, float]:
    """Same as `_candidate_recall`, but measured over
    `evidenced_candidate_steps` so filler padding added in Task B does not
    inflate the series that measures L1/L2 narrowing. Cases with no evidenced
    shortlist are excluded from the denominator."""
    measured = [r for r in results if r.evidenced_candidate_steps]
    recall: dict[int, float] = {}
    for k in _RECALL_K_LEVELS:
        if not measured:
            recall[k] = 0.0
            continue
        hits = sum(1 for r in measured if r.truth_step in r.evidenced_candidate_steps[:k])
        recall[k] = hits / len(measured)
    return recall


def layer_ablation_delta(with_layer: BenchReport, without_layer: BenchReport) -> dict[str, float]:
    """`with_layer.exact_accuracy - without_layer.exact_accuracy` and the same
    for `joint_accuracy` and `candidate_recall_at_k[5]`, the three numbers
    CLAUDE.md's layer-ablation study reports to prove L2 earns its complexity
    rather than asserting it. A negative delta is reported exactly like a
    positive one: this function does not know which layer is being tested,
    only computes the difference the caller asks it to."""
    return {
        "exact_accuracy_delta": with_layer.exact_accuracy - without_layer.exact_accuracy,
        "joint_accuracy_delta": with_layer.joint_accuracy - without_layer.joint_accuracy,
        "recall_at_5_delta": with_layer.candidate_recall_at_k.get(5, 0.0)
        - without_layer.candidate_recall_at_k.get(5, 0.0),
    }
