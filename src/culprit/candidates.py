"""L1+L2 candidate merge: turns whatever signals and divergences fired into
the bounded list of steps L3 actually adjudicates.

CLAUDE.md's "L3 adjudication" section is the spec this implements: union L1
and L2 by step index, `prior = max(l2_score, max_l1_severity) + 0.15` when
both sources fire (capped at 1.0), sort by prior descending with earlier
step winning ties, truncate to `max_candidates`. The co-location bonus is
the reason to run two independent layers instead of one - a step both a
deterministic detector and the contrastive aligner flagged independently is
stronger evidence than either alone.

Per-item error isolation (CLAUDE.md, "Per-item error isolation everywhere")
means a sentinel `Signal`/`DivergenceCandidate` with `error` set must never
enter this ranking: it carries no real severity or score, just a flag that
its own layer already failed on that item, and letting it count would let a
detector bug silently manufacture a fake root-cause candidate.
"""

from culprit.schemas import Outcome, Step, Trace
from culprit.signals import Candidate, DivergenceCandidate, Signal

# Matches CulpritConfig.max_candidates / min_reference_runs-style hardcoded
# fallbacks in culprit.toml (this module never imports config, per the
# project's layering rule - only cli.py and the plugin registries do).
_DEFAULT_MAX_CANDIDATES = 5
_DEFAULT_CO_LOCATION_BONUS = 0.15


def _step_span_lookup(steps: list[Step]) -> dict[int, str]:
    return {s.step_index: s.span_id for s in steps}


def _fallback_span_id(step_index: int, step_span: dict[int, str], signals: list[Signal], divergence: DivergenceCandidate | None) -> str:
    """`steps` is the authoritative source for span_id; a signal's own
    `span_id` (which can legitimately be `None`, see signals.py) or the
    divergence's span_id are only consulted if the step itself is somehow
    missing from `steps` - defensive, not expected on well-formed input."""
    if step_index in step_span:
        return step_span[step_index]
    for s in signals:
        if s.span_id:
            return s.span_id
    if divergence is not None:
        return divergence.span_id
    return ""


def merge_candidates(
    signals: list[Signal],
    divergences: list[DivergenceCandidate],
    trace: Trace,
    steps: list[Step],
    *,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    co_location_bonus: float = _DEFAULT_CO_LOCATION_BONUS,
) -> list[Candidate]:
    """Merge L1 `Signal`s and L2 `DivergenceCandidate`s into a ranked,
    truncated `Candidate` list for L3.

    Signals with `step_index == -1` (the sentinel value for "whole-trace or
    detector-failed", see signals.py) never anchor a candidate: there is no
    single step to attach them to, and including them would silently turn a
    detector crash into a phantom high-prior candidate at a fake index.

    Zero surviving candidates on a `FAILURE` trace still emits one synthetic
    `Candidate` at the terminal step with `source="fallback"` and `prior=0.0`
    - the plan's guarantee that L3 always examines something concrete rather
    than adjudicating nothing when the deterministic layers found no
    anomaly at all (the motivating case: a subtle failure no L1 detector's
    heuristics happened to catch and no L2 reference pool existed for).
    """
    step_span = _step_span_lookup(steps)

    clean_signals = [s for s in signals if s.error is None and s.step_index >= 0]
    clean_divergences = [d for d in divergences if d.error is None]

    signals_by_step: dict[int, list[Signal]] = {}
    for s in clean_signals:
        signals_by_step.setdefault(s.step_index, []).append(s)

    divergence_by_step: dict[int, DivergenceCandidate] = {}
    for d in clean_divergences:
        existing = divergence_by_step.get(d.step_index)
        if existing is None or d.divergence_score > existing.divergence_score:
            divergence_by_step[d.step_index] = d

    step_indices = set(signals_by_step) | set(divergence_by_step)

    built: list[Candidate] = []
    for step_index in step_indices:
        step_signals = signals_by_step.get(step_index, [])
        divergence = divergence_by_step.get(step_index)
        l1_severity = max((s.severity for s in step_signals), default=0.0)
        l2_score = divergence.divergence_score if divergence is not None else 0.0

        if step_signals and divergence is not None:
            source = "both"
            prior = min(1.0, max(l1_severity, l2_score) + co_location_bonus)
        elif step_signals:
            source, prior = "l1", l1_severity
        else:
            source, prior = "l2", l2_score

        built.append(Candidate(
            step_index=step_index,
            span_id=_fallback_span_id(step_index, step_span, step_signals, divergence),
            rank=0,
            prior=prior,
            source=source,
            signals=step_signals,
            divergence=divergence,
        ))

    # Earlier-step-wins on ties: the plan's earliness prior (also load-bearing
    # in L2's own D(i) scoring) applies here too - given two equally-ranked
    # candidates, the earlier one is more likely the cause, not the echo.
    built.sort(key=lambda c: (-c.prior, c.step_index))
    built = built[:max_candidates]
    for rank, candidate in enumerate(built, start=1):
        candidate.rank = rank

    if not built and trace is not None and trace.outcome == Outcome.FAILURE and steps:
        terminal = max(steps, key=lambda s: s.step_index)
        built = [Candidate(
            step_index=terminal.step_index,
            span_id=terminal.span_id,
            rank=1,
            prior=0.0,
            source="fallback",
            signals=[],
            divergence=None,
        )]

    return built
