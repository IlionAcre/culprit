"""L2 contrastive trajectory diff: aligns one failed trace's steps against a
pool of successful reference runs and scores each step by how much of the
achievable-success space it destroyed. See CLAUDE.md's "L2 contrastive"
section and the plan's "L2 contrastive algorithm" section for the full
design; this module is the orchestrator, `fingerprint.py`/`signature.py`/
`align.py`/`reference.py` hold the pieces.

**Contract fix, authorized after the WS-D handoff report.** The original
stub signature, `contrast(trace, steps, *, neighbor_fn, conn_fn)`, never
received spans, which was a defect in the frozen contract (the same
category as a missing parameter found elsewhere in `pipeline.diagnose`),
not a limitation to work around. It now takes `spans_by_id`, mirroring
`run_detectors(trace, steps, spans_by_id)` so the two analysis layers agree
on shape. `signature.sim`'s three previously-unreachable features (tool
argument keys/values, retrieved doc ids) are real now; see `signature.py`'s
module docstring for what changed there, including the small set of
payload-derived facts folded into the coarse alignment token itself.

**One gap remains, and is out of scope for this fix.** `D(i)`'s formula
includes `+0.15 * max_l1_severity_at(i)`, but `contrast()` still has no
`signals` parameter - L1 output still never reaches this layer. Left out of
the weighted sum rather than renormalized to compensate. Confirmed with the
coordinator: this is an intentional redundancy in the plan (WS-E's
candidate merge already applies the co-location bonus at the candidate
level), not a second instance of the same contract defect.

**Two correctness fixes discovered by actually running this against synth
data, not by re-reading the formula.**

1. Raw `A(i)` (residual alignability of the observed suffix `steps[i:]`)
   shrinks simply because a shorter suffix has fewer steps left to
   accumulate match score, entirely independent of any actual divergence.
   Comparing raw `A(i)` to raw `A(0)` in the point-of-no-return gate would
   make *every* trace's tail eligible, including a perfect match to a
   reference - failing the "identical sequences yield zero candidates"
   requirement outright. The gate instead compares the length-normalized
   rate `A(i) / (n - i)`, which stays flat across a well-matched trace and
   only drops where the trajectory actually leaves the achievable-success
   space.
2. The gate's baseline is the *best* rate observed anywhere in the trace,
   not `rate(0)` as "`A(i) < A(0)*0.85`" literally reads. A fault near the
   start (exactly where this layer's own synth tests, and plausibly many
   real failures, inject one) already drags `rate(0)` down with it, since
   the suffix starting at 0 includes the fault - comparing later positions
   against an already-contaminated baseline muted the exact drop the gate
   exists to catch.

Both remain after the contract fix; neither was weight-tuning, both were
measured bugs against the "identical sequences yield zero candidates" and
synth-injection requirements respectively.
"""

import logging
import math
from typing import Callable

from culprit.align import gotoh
from culprit.db import ConnFn
from culprit.embed import EmbedFn, embed_texts
from culprit.logging_config import LOGGER_NAME
from culprit.reference import build_profile
from culprit.schemas import Span, Step, Trace
from culprit.signals import ContrastResult, DivergenceCandidate, NeighborFn
from culprit.signature import signature_of
from culprit.signature import sim as fine_sim

logger = logging.getLogger(LOGGER_NAME)

MIN_REFERENCES = 3
MAX_REFERENCES = 25
GAP_OPEN = -1.0
GAP_EXTEND = -0.2
POINT_OF_NO_RETURN_RATIO = 0.85
PLATEAU_EPSILON = 0.05
EARLINESS_WEIGHT = 0.05
CLIFF_WEIGHT = 0.40
TRANSITION_WEIGHT = 0.20
PROFILE_WEIGHT = 0.20
TOP_K = 5
NO_COLUMN_SURPRISAL = 5.0  # ~-log(1/150): large but finite, for steps with no center column

# One element carries a step alongside its (possibly missing) span, so
# `align.gotoh` - fully generic over the aligned element type - can still
# call a single two-argument `sim_fn` while that function has both the
# structural fields (on Step) and the payload fields (on Span) it needs.
Elem = tuple[Step, "Span | None"]
ReferenceLoaderFn = Callable[[ConnFn, str], "tuple[list[Step], dict[str, Span]] | None"]


class PersistenceNotWiredError(Exception):
    """Raised by the default reference loader when `store_traces` (WS-B) is
    not importable, matching `jobs.py`'s `_seam` pattern: lazily import the
    real dependency at call time, fail with a clearly-named error naming
    the owner rather than a bare `ImportError`, and let a caller inject
    `reference_loader` explicitly until it lands (as every WS-D test does)."""


def _default_reference_loader(conn_fn: ConnFn, trace_id: str) -> "tuple[list[Step], dict[str, Span]] | None":
    try:
        from culprit.store_traces import read_trace
    except ImportError as e:
        raise PersistenceNotWiredError(
            "culprit.store_traces.read_trace is not available yet (WS-B). "
            "Inject reference_loader explicitly until then."
        ) from e
    _trace, spans, steps = read_trace(conn_fn, trace_id)
    return steps, {s.span_id: s for s in spans}


def _to_elems(steps: list[Step], spans_by_id: dict[str, Span]) -> list[Elem]:
    return [(s, spans_by_id.get(s.span_id)) for s in steps]


def _fine_sub(a: Elem, b: Elem) -> float:
    step_a, span_a = a
    step_b, span_b = b
    return 2.0 * fine_sim(step_a, step_b, span_a, span_b) - 1.0


def _norm(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def contrast(
    trace: Trace,
    steps: list[Step],
    spans_by_id: dict[str, Span],
    *,
    neighbor_fn: NeighborFn,
    conn_fn: ConnFn,
    embed_fn: EmbedFn = embed_texts,
    reference_loader: ReferenceLoaderFn = _default_reference_loader,
) -> ContrastResult:
    """Fingerprint implicitly via `embed_fn` (a full agent_key/task_key
    exact-match-then-kNN resolution lives inside the concrete `neighbor_fn`,
    `store_traces.nearest_successful`, owned by WS-B), resolve a reference
    pool, build the consensus profile and Markov model, align with Gotoh,
    and score each step's divergence via forward/reverse residual-
    alignability passes. Abstains with `abstain_reason="insufficient_references"`
    below 3 resolved references - a contrastive method with two references
    produces confident nonsense."""
    n = len(steps)
    if n == 0:
        return ContrastResult(candidates=[], reference_count=0, abstained=True, abstain_reason="empty_trace")

    query_embedding = embed_fn([trace.task_goal or ""])[0]
    candidate_ids = neighbor_fn(query_embedding, MAX_REFERENCES)

    pool: dict[str, tuple[list[Step], dict[str, Span]]] = {}
    for trace_id in candidate_ids:
        try:
            loaded = reference_loader(conn_fn, trace_id)
        except Exception:
            logger.warning("l2 reference load failed", extra={"event": "l2_reference_load_failed", "trace_id": trace_id})
            continue
        if loaded and loaded[0]:
            pool[trace_id] = loaded

    if len(pool) < MIN_REFERENCES:
        return ContrastResult(candidates=[], reference_count=len(pool), abstained=True, abstain_reason="insufficient_references")

    ref_ids = list(pool.keys())
    ref_steps_list = [pool[rid][0] for rid in ref_ids]
    ref_spans_list = [pool[rid][1] for rid in ref_ids]
    ref_sig_seqs = [
        [signature_of(s, spans.get(s.span_id)) for s in rs]
        for rs, spans in zip(ref_steps_list, ref_spans_list)
    ]

    profile = build_profile(ref_sig_seqs)
    center_steps = ref_steps_list[profile.center_index]
    center_spans = ref_spans_list[profile.center_index]

    elems = _to_elems(steps, spans_by_id)
    center_elems = _to_elems(center_steps, center_spans)
    center_alignment = gotoh(elems, center_elems, _fine_sub, GAP_OPEN, GAP_EXTEND)

    op_by_step: dict[int, str] = {}
    column_by_step: dict[int, int] = {}
    for op, a_idx, b_idx in center_alignment.ops:
        if a_idx is None:
            continue
        op_by_step[a_idx] = op
        if b_idx is not None:
            column_by_step[a_idx] = b_idx

    rev_elems = list(reversed(elems))
    per_ref_row_max: dict[str, list[float]] = {}
    for rid, rs, spans in zip(ref_ids, ref_steps_list, ref_spans_list):
        try:
            rev_ref_elems = list(reversed(_to_elems(rs, spans)))
            rev_alignment = gotoh(rev_elems, rev_ref_elems, _fine_sub, GAP_OPEN, GAP_EXTEND)
        except Exception:
            logger.warning("l2 reverse alignment failed", extra={"event": "l2_reverse_alignment_failed", "trace_id": rid})
            continue
        # matrix[i'][k'] = best score aligning steps' suffix of length i'
        # (= steps[n-i':]) against rs's suffix of length k'. Row max over
        # k' frees the reference-side start: best match against ANY
        # suffix of this reference, not one fixed by a prior alignment.
        per_ref_row_max[rid] = [max(row) for row in rev_alignment.matrix]

    reference_count = len(per_ref_row_max)
    if reference_count < MIN_REFERENCES:
        return ContrastResult(candidates=[], reference_count=reference_count, abstained=True, abstain_reason="insufficient_references")

    # Per-reference A_j(i), indexed by i (not i'): per_ref_A[rid][i] is the
    # best score aligning steps[i:] against any suffix of reference rid.
    per_ref_A = {rid: [row_max[n - i] for i in range(n + 1)] for rid, row_max in per_ref_row_max.items()}

    A = [0.0] * (n + 1)
    nearest_refs: list[list[str]] = [[] for _ in range(n + 1)]
    for i in range(n + 1):
        ranked = sorted(per_ref_A.items(), key=lambda kv: -kv[1][i])
        A[i] = ranked[0][1][i]
        nearest_refs[i] = [rid for rid, _ in ranked[:3]]

    rate = [A[i] / max(1, n - i) for i in range(n + 1)]
    cliffs = [max(0.0, A[i + 1] - A[i]) for i in range(n)]

    sigs = [signature_of(s, spans_by_id.get(s.span_id)) for s in steps]
    transition_surprisal = [profile.surprisal(sigs[i - 1] if i > 0 else None, sigs[i]) for i in range(n)]

    profile_surprisal: list[float] = []
    expected_by_step: dict[int, list[tuple[str, float]]] = {}
    for i in range(n):
        col = column_by_step.get(i)
        if col is None:
            profile_surprisal.append(NO_COLUMN_SURPRISAL)
            expected_by_step[i] = []
        else:
            profile_surprisal.append(-math.log(profile.column_prob(col, sigs[i])))
            expected_by_step[i] = profile.column_top(col, 3)

    norm_cliff, norm_transition, norm_profile = _norm(cliffs), _norm(transition_surprisal), _norm(profile_surprisal)
    D = [
        CLIFF_WEIGHT * norm_cliff[i]
        + TRANSITION_WEIGHT * norm_transition[i]
        + PROFILE_WEIGHT * norm_profile[i]
        - EARLINESS_WEIGHT * (i / n)
        for i in range(n)
    ]

    # Baseline is the best rate observed anywhere in the trace, not rate[0]
    # as the plan's "A(i) < A(0)*0.85" literally reads. A fault near the
    # very start (as most of this layer's own synth tests inject one, and
    # plausibly as many real failures do too) already drags rate[0] down
    # with it, since the suffix starting at 0 includes the fault; comparing
    # later positions against that already-contaminated baseline mutes the
    # exact drop the gate exists to catch. The best-observed rate is what a
    # clean stretch of *this* run actually achieved, wherever it occurs.
    baseline = max(rate[:n]) if n > 0 else 0.0
    if baseline > 0:
        threshold = baseline * POINT_OF_NO_RETURN_RATIO
        eligible = [i for i in range(n) if rate[i] < threshold]
    else:
        eligible = list(range(n))
    eligible.sort()

    groups: list[list[int]] = []
    for i in eligible:
        if groups and i == groups[-1][-1] + 1 and abs(D[i] - D[groups[-1][-1]]) <= PLATEAU_EPSILON:
            groups[-1].append(i)
        else:
            groups.append([i])
    collapsed = [(g[0], max(D[k] for k in g)) for g in groups]
    collapsed.sort(key=lambda t: (-t[1], t[0]))

    candidates = []
    for step_idx, score in collapsed[:TOP_K]:
        step = steps[step_idx]
        candidates.append(DivergenceCandidate(
            step_index=step_idx,
            span_id=step.span_id,
            divergence_score=score,
            cliff_delta=cliffs[step_idx],
            surprisal=transition_surprisal[step_idx],
            profile_surprisal=profile_surprisal[step_idx],
            reference_count=reference_count,
            observed_signature=sigs[step_idx],
            expected_signatures=expected_by_step.get(step_idx, []),
            nearest_reference_trace_ids=nearest_refs[step_idx],
            alignment_op=op_by_step.get(step_idx, "insertion"),
        ))

    return ContrastResult(candidates=candidates, reference_count=reference_count, abstained=False, abstain_reason=None)
