"""Confidence calibration and trace-level abstention for L3 adjudication.

**The five coefficients below (a0-a4) are HAND-SET PRIORS, not fitted
values.** No labeled ground truth exists yet to fit them against;
`bench_score.py` (WS-H) is what will eventually let them be refit by
logistic regression once labeled diagnoses exist. Calling this "calibrated"
today would be exactly the unearned claim this project's own decision log
warns against - it is a documented starting guess wired into the one place
that will make it measurable, not a claim of accuracy.

`cite_check` is the highest-value term in the formula (CLAUDE.md's "L3
adjudication" section): a rationale citing step 47 when only steps 12-16
were shown is direct, cheap evidence of confabulation, not a benign
paraphrase, and `a4` is deliberately the largest-magnitude coefficient so it
actually moves the number.

`select_diagnosis` implements the plan's three independent abstention
gates. Uber's "Project RADAR" is the citation CLAUDE.md records for this
posture: present ranked suspects, not a forced verdict.
"""

import math
from dataclasses import dataclass

from culprit.signals import Adjudication

# a0 intercept, a1 logit(model_conf), a2 prior, a3 agreement, a4 evidence
# density. See module docstring: hand-set, not fitted.
DEFAULT_A0 = 0.0
DEFAULT_A1 = 1.0
DEFAULT_A2 = 0.5
DEFAULT_A3 = 0.3
DEFAULT_A4 = 1.5

# Matches CulpritConfig.min_confidence / ambiguity_margin's hardcoded
# fallbacks (this module never imports config, per the project's layering
# rule - only cli.py and the plugin registries do).
_DEFAULT_MIN_CONFIDENCE = 0.55
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


def calibrate_confidence(
    model_confidence: float,
    prior: float,
    agreement: bool,
    evidence_density: float,
    *,
    a0: float = DEFAULT_A0,
    a1: float = DEFAULT_A1,
    a2: float = DEFAULT_A2,
    a3: float = DEFAULT_A3,
    a4: float = DEFAULT_A4,
) -> float:
    """`calibrated = sigmoid(a0 + a1*logit(model_conf) + a2*prior +
    a3*agreement + a4*evidence_density)`, the formula in CLAUDE.md's "L3
    adjudication" section. See module docstring for why the coefficients
    are priors, not fitted values."""
    x = (
        a0
        + a1 * _logit(model_confidence)
        + a2 * prior
        + a3 * (1.0 if agreement else 0.0)
        + a4 * evidence_density
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
