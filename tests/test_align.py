"""Alignment correctness tests. The hand-computed 5x5 case is the
non-negotiable one (see CLAUDE.md's L2 section): a subtle traceback bug is
invisible in aggregate metrics, so this reproduces a fully worked-by-hand
Gotoh table, including every traceback step, rather than only checking the
final score.
"""

import pytest

from culprit.align import gotoh

_MATCH = lambda x, y: 1.0 if x == y else -1.0  # noqa: E731


def test_hand_computed_5x5_matches_worked_table_exactly():
    """a = [1,2,3,4,5], b = [1,3,4,5,6], match=+1, mismatch=-1, gap_open=-1.0,
    gap_extend=-0.2. Optimal alignment (hand-derived, full Gotoh table
    worked cell by cell): match(1,1), insert 2, match(3,3), match(4,4),
    match(5,5), delete 6. Score = 4 matches * (+1) + 2 gap opens * (-1.0)
    = 2.0 exactly, achieved via the Iy matrix at the final cell (b's
    trailing 6 opens a fresh deletion from M[5][4], not an extension of a
    prior gap - the two gaps in this example are each length 1, so this
    case alone does not exercise the extend cost, see the affine-gap test
    below for that).
    """
    a = [1, 2, 3, 4, 5]
    b = [1, 3, 4, 5, 6]

    alignment = gotoh(a, b, _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert alignment.score == 2.0
    assert alignment.ops == [
        ("match", 0, 0),
        ("insertion", 1, None),
        ("match", 2, 1),
        ("match", 3, 2),
        ("match", 4, 3),
        ("deletion", None, 4),
    ]


def test_affine_gap_extend_cheaper_than_repeated_open():
    """A 3-step insertion burst must cost `gap_open + 2*gap_extend`
    (-1.0 + 2*-0.2 = -1.4), not `3*gap_open` (-3.0). This is the entire
    reason affine gaps were chosen over linear ones: an inserted retry
    burst is structurally one event, and a linear-gap scheme would drown
    the real divergence elsewhere in the trace by charging full price per
    inserted step."""
    a = ["x", "y1", "y2", "y3", "z"]
    b = ["x", "z"]

    alignment = gotoh(a, b, _MATCH, gap_open=-1.0, gap_extend=-0.2)

    # match x (+1), insert y1/y2/y3 (-1.0 + 2*-0.2 = -1.4), match z (+1)
    assert alignment.score == pytest.approx(1.0 - 1.4 + 1.0)
    assert alignment.ops == [
        ("match", 0, 0),
        ("insertion", 1, None),
        ("insertion", 2, None),
        ("insertion", 3, None),
        ("match", 4, 1),
    ]


def test_identical_sequences_score_maximum_with_no_gaps():
    a = ["a", "b", "c", "d"]

    alignment = gotoh(a, list(a), _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert alignment.score == 4.0
    assert alignment.ops == [("match", i, i) for i in range(4)]


def test_score_is_symmetric_under_swapping_the_two_sequences():
    a = [1, 2, 3, 4, 5]
    b = [1, 3, 4, 5, 6]

    forward = gotoh(a, b, _MATCH, gap_open=-1.0, gap_extend=-0.2)
    backward = gotoh(b, a, _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert forward.score == backward.score


def test_empty_sequence_costs_exactly_one_gap_open_plus_extends():
    alignment = gotoh([], ["a", "b", "c"], _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert alignment.score == -1.0 + 2 * -0.2
    assert alignment.ops == [
        ("deletion", None, 0),
        ("deletion", None, 1),
        ("deletion", None, 2),
    ]


def test_both_empty_scores_zero_with_no_ops():
    alignment = gotoh([], [], _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert alignment.score == 0.0
    assert alignment.ops == []


def test_matrix_bottom_right_cell_equals_reported_score():
    a, b = [1, 2, 3], [1, 2, 4]

    alignment = gotoh(a, b, _MATCH, gap_open=-1.0, gap_extend=-0.2)

    assert alignment.matrix[len(a)][len(b)] == alignment.score
