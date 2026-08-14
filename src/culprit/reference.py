"""The reference model: a consensus profile over the resolved pool of
successful runs (center-star progressive alignment, Laplace-smoothed
per-column signature distribution), a first-order signature Markov model
(add-0.5 smoothing) for position-independent local surprisal, and count
envelopes (median/MAD of step counts) so a later consumer can tell "three
calls to the same tool" apart from a genuine retry storm using the pool's
own typical range rather than a hardcoded threshold.

**Simplification, flagged rather than hidden.** True center-star MSA
propagates insertions found in *any* sequence back into new shared columns
across the whole alignment. This builds columns only from the center
reference's own positions: a reference's insertion relative to the center
(an extra step the center doesn't have) contributes to no column at all.
Every `P(sig | column)` this profile reports is still exactly correct for
what it claims - the fraction of the pool that did or didn't take the
center's action at that position - it just never grows a column for
action sequences the center itself never took. Revisit if profile columns
turn out to undercount common non-center detours.
"""

import math
import statistics
from dataclasses import dataclass, field

from culprit.align import gotoh

_EXACT = lambda a, b: 1.0 if a == b else -1.0  # noqa: E731


@dataclass
class ProfileColumn:
    counts: dict[str, int] = field(default_factory=dict)
    gap_count: int = 0

    @property
    def total(self) -> int:
        return sum(self.counts.values()) + self.gap_count


@dataclass
class Profile:
    columns: list[ProfileColumn]
    center_index: int
    vocab: set[str]
    prev_totals: dict[str | None, int] = field(default_factory=dict)
    transitions: dict[tuple[str | None, str], int] = field(default_factory=dict)
    step_count_median: float = 0.0
    step_count_mad: float = 0.0
    signature_count_median: dict[str, float] = field(default_factory=dict)
    signature_count_mad: dict[str, float] = field(default_factory=dict)

    def column_prob(self, column_idx: int, sig: str) -> float:
        """Laplace-smoothed P(sig | this profile column)."""
        v = len(self.vocab) or 1
        col = self.columns[column_idx]
        return (col.counts.get(sig, 0) + 0.5) / (col.total + 0.5 * v)

    def column_top(self, column_idx: int, k: int = 3) -> list[tuple[str, float]]:
        """Top-k `(signature, probability)` at this column - the string
        quoted verbatim to L3 and to the user ("82% called search_orders
        here"), so this must stay directly interpretable."""
        col = self.columns[column_idx]
        ranked = sorted(col.counts.items(), key=lambda kv: -kv[1])[:k]
        return [(sig, self.column_prob(column_idx, sig)) for sig, _ in ranked]

    def surprisal(self, prev_sig: str | None, sig: str) -> float:
        """nats, first-order Markov, add-0.5 smoothed. Complements
        `column_prob`: a step can be normal-in-general but wrong-here (low
        `surprisal`, low `column_prob`), or fine-here but never-seen-
        anywhere (high `surprisal`, high `column_prob`)."""
        v = len(self.vocab | {sig}) or 1
        denom = self.prev_totals.get(prev_sig, 0) + 0.5 * v
        count = self.transitions.get((prev_sig, sig), 0)
        return -math.log((count + 0.5) / denom)


def _pick_center(seqs: list[list[str]]) -> int:
    """The reference with the highest sum-of-pairs coarse-alignment score
    against every other reference becomes the center-star anchor."""
    if len(seqs) == 1:
        return 0
    scores = [0.0] * len(seqs)
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            s = gotoh(seqs[i], seqs[j], _EXACT).score
            scores[i] += s
            scores[j] += s
    return max(range(len(seqs)), key=lambda i: scores[i])


def _median_mad(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    return float(med), float(mad)


def build_profile(ref_signature_seqs: list[list[str]]) -> Profile:
    """One coarse-signature sequence per reference trace (already ordered
    by `step_index`). Callers (`contrast.py`) enforce the >=3-reference
    abstention floor before ever calling this."""
    if not ref_signature_seqs:
        raise ValueError("build_profile requires at least one reference sequence")

    vocab = {sig for seq in ref_signature_seqs for sig in seq}
    center_idx = _pick_center(ref_signature_seqs)
    center = ref_signature_seqs[center_idx]
    columns = [ProfileColumn() for _ in center]

    for r, seq in enumerate(ref_signature_seqs):
        if r == center_idx:
            for col, sig in zip(columns, center):
                col.counts[sig] = col.counts.get(sig, 0) + 1
            continue
        alignment = gotoh(center, seq, _EXACT)
        for op, a_idx, b_idx in alignment.ops:
            if a_idx is None:
                continue  # seq's insertion relative to center touches no column
            col = columns[a_idx]
            if b_idx is None:
                col.gap_count += 1
            else:
                sig = seq[b_idx]
                col.counts[sig] = col.counts.get(sig, 0) + 1

    transitions: dict[tuple[str | None, str], int] = {}
    prev_totals: dict[str | None, int] = {}
    for seq in ref_signature_seqs:
        prev: str | None = None
        for sig in seq:
            key = (prev, sig)
            transitions[key] = transitions.get(key, 0) + 1
            prev_totals[prev] = prev_totals.get(prev, 0) + 1
            prev = sig

    step_median, step_mad = _median_mad([len(seq) for seq in ref_signature_seqs])

    per_sig_counts: dict[str, list[int]] = {sig: [] for sig in vocab}
    for seq in ref_signature_seqs:
        seen: dict[str, int] = {}
        for sig in seq:
            seen[sig] = seen.get(sig, 0) + 1
        for sig in vocab:
            per_sig_counts[sig].append(seen.get(sig, 0))
    sig_median, sig_mad = {}, {}
    for sig, counts in per_sig_counts.items():
        med, mad = _median_mad(counts)
        sig_median[sig], sig_mad[sig] = med, mad

    return Profile(
        columns=columns,
        center_index=center_idx,
        vocab=vocab,
        prev_totals=prev_totals,
        transitions=transitions,
        step_count_median=step_median,
        step_count_mad=step_mad,
        signature_count_median=sig_median,
        signature_count_mad=sig_mad,
    )
