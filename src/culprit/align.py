"""Needleman-Wunsch global alignment with affine gap penalties (Gotoh 1982,
three DP matrices). Generic over element type: the caller supplies
`sim_fn(a_elem, b_elem) -> float` (a substitution score, positive for
similar, negative for dissimilar - `signature.py`'s `sim` scaled to
`2*sim-1` for Step-level alignment, or plain token equality for the coarse
profile alignment in `reference.py`).

Implemented as three plain nested-loop passes, not vectorized. A subtle
traceback bug is invisible in aggregate metrics and silently degrades
everything downstream (`CLAUDE.md`), so this stays readable; profiling
would need to demand the vectorized version before it's worth the loss of
clarity, and at the trace lengths this product deals with (tens to low
hundreds of steps) it never has.

Gotoh's three matrices, for sequences `a` (length n) and `b` (length m):
  M[i][j]  - best score aligning a[:i] to b[:j] ending in a substitution
             (a[i-1] paired with b[j-1])
  Ix[i][j] - best score ending with a[i-1] aligned to a gap in b
             ("insertion": a has a step b does not)
  Iy[i][j] - best score ending with b[j-1] aligned to a gap in a
             ("deletion": b has a step a does not)
A gap costs `gap_open` to start and `gap_extend` for each additional step,
so a five-step insertion burst costs `gap_open + 4*gap_extend` once, not
`5*gap_open` - the whole reason affine gaps were chosen over linear ones
(see CLAUDE.md's L2 section).
"""

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

NEG_INF = float("-inf")

# One op per consumed element (or pair), in a-then-b order, left to right:
#   ("match"/"mismatch", a_index, b_index)  - substitution, both consumed
#   ("insertion", a_index, None)            - a has an extra element
#   ("deletion", None, b_index)             - b has an extra element
AlignOp = tuple[str, int | None, int | None]


@dataclass
class Alignment:
    score: float
    ops: list[AlignOp]
    # matrix[i][j] = best score aligning a[:i] against b[:j] (max of the
    # three DP matrices at that cell). Exposed so a caller can run `gotoh`
    # a second time on reversed sequences and read off suffix-alignment
    # scores directly, rather than re-aligning per position - the "one
    # extra O(n*m) pass" residual-alignability needs (see `contrast.py`).
    matrix: list[list[float]]


def _best_of_three(m: float, ix: float, iy: float) -> tuple[float, str]:
    """Tie-break priority M > Ix > Iy: arbitrary but fixed, so a genuine
    tie (which none of this module's own tests hit) resolves the same way
    every run rather than depending on dict/float iteration order."""
    if m >= ix and m >= iy:
        return m, "M"
    if ix >= iy:
        return ix, "Ix"
    return iy, "Iy"


def gotoh(
    a: Sequence[T],
    b: Sequence[T],
    sim_fn: Callable[[T, T], float],
    gap_open: float = -1.0,
    gap_extend: float = -0.2,
) -> Alignment:
    n, m = len(a), len(b)

    M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Ix = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Iy = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    # Predecessor matrix label for each cell, used only during traceback.
    Mfrom: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    Ixfrom: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    Iyfrom: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    # Cached substitution scores, reused during traceback so the match vs
    # mismatch label doesn't recompute sim_fn.
    S: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            if i > 0 and j > 0:
                s = sim_fn(a[i - 1], b[j - 1])
                S[i][j] = s
                prev, frm = _best_of_three(M[i - 1][j - 1], Ix[i - 1][j - 1], Iy[i - 1][j - 1])
                M[i][j] = s + prev
                Mfrom[i][j] = frm
            if i > 0:
                open_score = M[i - 1][j] + gap_open
                ext_score = Ix[i - 1][j] + gap_extend
                if open_score >= ext_score:
                    Ix[i][j], Ixfrom[i][j] = open_score, "M"
                else:
                    Ix[i][j], Ixfrom[i][j] = ext_score, "Ix"
            if j > 0:
                open_score = M[i][j - 1] + gap_open
                ext_score = Iy[i][j - 1] + gap_extend
                if open_score >= ext_score:
                    Iy[i][j], Iyfrom[i][j] = open_score, "M"
                else:
                    Iy[i][j], Iyfrom[i][j] = ext_score, "Iy"

    score, end_mat = _best_of_three(M[n][m], Ix[n][m], Iy[n][m])

    ops: list[AlignOp] = []
    i, j, mat = n, m, end_mat
    while not (i == 0 and j == 0):
        if mat == "M":
            label = "match" if S[i][j] > 0 else "mismatch"
            ops.append((label, i - 1, j - 1))
            mat = Mfrom[i][j]
            i, j = i - 1, j - 1
        elif mat == "Ix":
            ops.append(("insertion", i - 1, None))
            mat = Ixfrom[i][j]
            i -= 1
        else:  # "Iy"
            ops.append(("deletion", None, j - 1))
            mat = Iyfrom[i][j]
            j -= 1
    ops.reverse()

    matrix = [[max(M[i][j], Ix[i][j], Iy[i][j]) for j in range(m + 1)] for i in range(n + 1)]
    return Alignment(score=score, ops=ops, matrix=matrix)
