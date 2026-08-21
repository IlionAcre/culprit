"""Confidence calibration and trace-level abstention for L3 adjudication.

**Eleven coefficients (a0-a10), refit 2026-08-21 against the persisted rows
from the Phase 3 pre-refit benchmark pass.** The pre-refit population is larger
than the 2026-08-17 fit (3,021 rows, 410 positive, from all four benchmark
configurations after the shortlist was filled to five candidates) and includes
`is_filler` as an eleventh feature. Out-of-fold AUC is ~0.669 and Brier ~0.114
(5-fold CV averaged over 5 seeds). Full history, prior fits, and the
precision-at-threshold table are in CLAUDE.md's "L3 adjudication" section.

`select_diagnosis` implements the plan's three independent abstention
gates. Uber's "Project RADAR" is the citation CLAUDE.md records for this
posture: present ranked suspects, not a forced verdict.
"""

import math
from dataclasses import dataclass

from culprit.signals import Adjudication

# a0 intercept, a1 logit(model_conf), a2 prior, a3 agreement, a4 evidence
# density, a5 rank score, a6 step position, a7 step depth, a8 L1 signal
# count, a9 is_fallback, a10 is_filler. Refit 2026-08-21 against 3,021
# persisted adjudication rows (410 positive) from all four benchmark
# configurations after the shortlist was filled to five candidates; see
# CLAUDE.md's "L3 adjudication" section for the fit history, the
# precision-at-threshold table, and the honest caveats about sample size.
# `a2` (prior) and `a4` (evidence_density) came back negative here: once rank,
# position, depth, signal-count, and source flags are present, the raw prior
# and citation density do not carry independent positive signal in this
# population. `a10` (is_filler) is positive, which partly offsets the low
# rank_norm and zero-prior penalty of a filler rather than marking it down.
# Hand-set priors, for reference: a0=0.0, a1=1.0, a2=0.5, a3=0.3, a4=1.5,
# a5-a9=0.0. I7's 4-feature fit, superseded: a0=-2.6065, a1=-0.0285,
# a2=0.7678, a3=0.6492, a4=0.0053. 2026-08-17 richer-feature fit,
# superseded: a0=-4.3970, a1=-0.0482, a2=0.2273, a3=-0.0892, a4=0.0020,
# a5=0.1206, a6=2.1086, a7=1.1989, a8=0.2925, a9=-1.5935, a10=0.0.
DEFAULT_A0 = -5.3395
DEFAULT_A1 = 0.0077
DEFAULT_A2 = -1.4268
DEFAULT_A3 = -0.4531
DEFAULT_A4 = -0.4575
DEFAULT_A5 = 0.8888
DEFAULT_A6 = 1.3492
DEFAULT_A7 = 2.5913
DEFAULT_A8 = 0.7388
DEFAULT_A9 = -1.8539
DEFAULT_A10 = 0.7661

# Candidate.rank never exceeds candidates.py's own _DEFAULT_MAX_CANDIDATES,
# duplicated rather than imported (this module never imports outside
# signals.py, per the project's layering rule - only cli.py and the plugin
# registries import culprit.config).
_DEFAULT_MAX_CANDIDATES = 5

# 0.15: the lowest threshold, in the precision-at-threshold table computed
# against this same 447-row population (CLAUDE.md has the full table),
# where committed volume is trustworthy (n=122, well over the ~15 floor)
# and precision (23.8%, out-of-fold, averaged over 5 CV seeds) is
# meaningfully above the 10.3% base rate - roughly 2.3x, versus 1.4-1.8x at
# thresholds below it. Higher floors buy more precision (0.18: 41.7%) at
# the cost of most remaining coverage (13.4% vs 27.3%) and a smaller,
# more volatile committed set. Replaces the old 0.55 floor, which sat above
# this model's predecessor's ~0.29 ceiling and forced unconditional
# abstention; this model's own ceiling is far higher (~0.83-0.91 over the
# realistic input domain, CLAUDE.md), so 0.15 is comfortably reachable.
_DEFAULT_MIN_CONFIDENCE = 0.15
_DEFAULT_AMBIGUITY_MARGIN = 0.08

_EPS = 1e-6


def _logit(p: float) -> float:
    """Inverse sigmoid, clamped away from exactly 0.0/1.0 first: a model
    that reports confidence 1.0 or 0.0 (not rare - some models saturate)
    would otherwise send `math.log` to a domain error at the boundary."""
    clamped = min(max(p, _EPS), 1.0 - _EPS)
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(x: float) -> float:
    # Via tanh rather than 1/(1+exp(-x)) directly: well-defined at any input
    # magnitude with no need to reason about how large x can get from the
    # caller's coefficients, rather than assuming it never will.
    return 0.5 * (1.0 + math.tanh(x / 2.0))


def cite_check(cited_step_indices: list[int], visible_step_indices: frozenset[int]) -> float:
    """Fraction of `cited_step_indices` actually present in the context
    packet the model was shown (`ContextPacket.visible_step_indices`, see
    context_window.py). An empty citation list scores 0.0, not 1.0: this
    term exists to reward citing real, checkable evidence, and a rationale
    that cites nothing has none to check, so it earns no credit either."""
    if not cited_step_indices:
        return 0.0
    grounded = sum(1 for i in cited_step_indices if i in visible_step_indices)
    return grounded / len(cited_step_indices)


def agrees_with_l1(candidate_categories: list[str], chosen_failure_class: str) -> bool:
    """Whether L3's chosen `failure_class` matches any `Signal.category`
    already attached to this candidate by L1. `Signal.category` is a hint
    (signals.py), so agreement here is corroboration between two
    independent layers, not a guarantee either is correct."""
    return chosen_failure_class in candidate_categories


def step_position(step_index: int, step_count: int) -> float:
    """Candidate step's position within the trace's whole step sequence,
    normalized to [0, 1]: 0.0 at the first step, 1.0 at the last. A trace of
    0 or 1 steps has no meaningful position to normalize against, so this
    returns 0.0 rather than dividing by zero."""
    if step_count is None or step_count <= 1:
        return 0.0
    return step_index / (step_count - 1)


def depth_norm(depth: int, max_depth: int) -> float:
    """Step depth (nesting level in the span tree, see `schemas.Step`)
    normalized against the deepest step anywhere in this trace. A flat
    trace with no nesting (`max_depth == 0`) has nothing to normalize
    against, so this returns 0.0."""
    if not max_depth:
        return 0.0
    return depth / max_depth


def _rank_score(rank: int, max_candidates: int) -> float:
    """`Candidate.rank` (1 = top-ranked) transformed onto the same [0,1]
    scale as this module's other terms: 1.0 for the top-ranked candidate
    down to 0.0 for the lowest-ranked one out of `max_candidates`. Kept
    internal to `calibrate_confidence` since a raw rank integer is not
    itself a probability-like quantity, unlike `step_position`/`depth_norm`
    above, which the caller already normalizes before passing in."""
    if max_candidates <= 1:
        return 1.0
    return 1.0 - (rank - 1) / (max_candidates - 1)


def calibrate_confidence(
    model_confidence: float,
    prior: float,
    agreement: bool,
    evidence_density: float,
    rank: int,
    step_position: float,
    depth_norm: float,
    n_l1_signals: int,
    is_fallback: bool,
    is_filler: bool = False,
    *,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    a0: float = DEFAULT_A0,
    a1: float = DEFAULT_A1,
    a2: float = DEFAULT_A2,
    a3: float = DEFAULT_A3,
    a4: float = DEFAULT_A4,
    a5: float = DEFAULT_A5,
    a6: float = DEFAULT_A6,
    a7: float = DEFAULT_A7,
    a8: float = DEFAULT_A8,
    a9: float = DEFAULT_A9,
    a10: float = DEFAULT_A10,
) -> float:
    """`calibrated = sigmoid(a0 + a1*logit(model_conf) + a2*prior +
    a3*agreement + a4*evidence_density + a5*rank_score(rank) +
    a6*step_position + a7*depth_norm + a8*n_l1_signals +
    a9*is_fallback + a10*is_filler)`.

    `step_position` and `depth_norm` are pre-normalized floats (this
    module's own `step_position()`/`depth_norm()` functions above build
    them from a `Step`/`Trace`); `rank` is the raw 1-based
    `Candidate.rank` and is normalized internally by `_rank_score`, since
    unlike the other new terms it is not already a [0,1] quantity at the
    call site. `is_filler` joins the fit in Phase 3 so the refit can learn
    the value of evidence-free filler candidates rather than have the
    position terms absorb it. See module docstring for the fit this formula
    and its coefficients came from."""
    x = (
        a0
        + a1 * _logit(model_confidence)
        + a2 * prior
        + a3 * (1.0 if agreement else 0.0)
        + a4 * evidence_density
        + a5 * _rank_score(rank, max_candidates)
        + a6 * step_position
        + a7 * depth_norm
        + a8 * n_l1_signals
        + a9 * (1.0 if is_fallback else 0.0)
        + a10 * (1.0 if is_filler else 0.0)
    )
    return _sigmoid(x)


@dataclass
class DiagnosisVerdict:
    """The trace-level verdict this layer hands upward: which `Adjudication`
    (if any) won, and whether the abstention gates blocked committing to
    one. Not part of the frozen `Signal`/`Diagnosis` contract in signals.py
    - `pipeline.py` (Integration-owned) is what actually assembles a
    `Diagnosis` from this; this is only the decision this layer owns."""

    winner: Adjudication | None
    abstained: bool
    abstain_reason: str | None


def select_diagnosis(
    adjudications: list[Adjudication],
    *,
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ambiguity_margin: float = _DEFAULT_AMBIGUITY_MARGIN,
) -> DiagnosisVerdict:
    """Three independent abstention gates, in order, per CLAUDE.md's "L3
    adjudication" section:

    1. Model self-abstention. Only candidates the model itself called the
       root cause are eligible; `is_root_cause=False` is a correct, expected
       answer, not a failure to find one, so it never gets silently
       promoted into a runner-up verdict.
    2. Confidence floor: the top eligible candidate's calibrated confidence
       must clear `min_confidence`.
    3. Ambiguity margin: if the top two eligible candidates are within
       `ambiguity_margin` of each other *and* disagree on failure class,
       that is a coin flip dressed as a verdict, not a diagnosis.
    """
    eligible = [
        a for a in adjudications
        if a.error is None and not a.abstained and a.is_root_cause
    ]
    if not eligible:
        return DiagnosisVerdict(None, True, "no candidate self-reported as the root cause")

    eligible.sort(key=lambda a: (-a.calibrated_confidence, a.step_index))
    top = eligible[0]

    if top.calibrated_confidence < min_confidence:
        return DiagnosisVerdict(
            None, True,
            f"top calibrated confidence {top.calibrated_confidence:.2f} "
            f"below floor {min_confidence:.2f}",
        )

    if len(eligible) > 1:
        runner_up = eligible[1]
        margin = top.calibrated_confidence - runner_up.calibrated_confidence
        if margin <= ambiguity_margin and top.failure_class != runner_up.failure_class:
            return DiagnosisVerdict(
                None, True,
                f"ambiguous between step {top.step_index} ({top.failure_class}) and "
                f"step {runner_up.step_index} ({runner_up.failure_class})",
            )

    return DiagnosisVerdict(top, False, None)
