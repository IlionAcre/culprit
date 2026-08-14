from culprit.reference import build_profile


def test_build_profile_rejects_empty_pool():
    import pytest

    with pytest.raises(ValueError):
        build_profile([])


def test_center_is_the_most_typical_sequence():
    """Two references are identical, one has an extra inserted step. The
    sum-of-pairs center pick should land on one of the two identical (more
    "typical") sequences, not the outlier."""
    typical = ["a", "b", "c", "d"]
    outlier = ["a", "b", "x", "c", "d"]

    profile = build_profile([typical, list(typical), outlier])

    assert profile.center_index in (0, 1)


def test_column_prob_reflects_majority_vote():
    """3 of 4 references call "search_orders" at position 2; the profile
    column there should assign it the highest probability, quoted
    verbatim to the user."""
    seqs = [
        ["a", "search_orders", "c"],
        ["a", "search_orders", "c"],
        ["a", "search_orders", "c"],
        ["a", "get_customer", "c"],
    ]

    profile = build_profile(seqs)

    top = profile.column_top(1, k=1)
    assert top[0][0] == "search_orders"
    assert top[0][1] > 0.5


def test_column_prob_is_laplace_smoothed_never_exactly_zero_or_one():
    seqs = [["a", "b"], ["a", "b"], ["a", "b"]]

    profile = build_profile(seqs)

    prob = profile.column_prob(0, "never_seen_anywhere")
    assert 0.0 < prob < 1.0


def test_surprisal_is_lower_for_common_transitions_than_rare_ones():
    seqs = [
        ["plan", "search", "answer"],
        ["plan", "search", "answer"],
        ["plan", "search", "answer"],
        ["plan", "weird_rare_step", "answer"],
    ]

    profile = build_profile(seqs)

    common = profile.surprisal("plan", "search")
    rare = profile.surprisal("plan", "weird_rare_step")
    never_seen = profile.surprisal("plan", "totally_unseen_token")

    assert common < rare < never_seen


def test_step_count_envelope_matches_hand_computed_median_and_mad():
    seqs = [["a"] * 4, ["a"] * 6, ["a"] * 6, ["a"] * 8]  # median 6, MAD = median(|2,0,0,2|) = 1

    profile = build_profile(seqs)

    assert profile.step_count_median == 6.0
    assert profile.step_count_mad == 1.0


def test_single_reference_pool_still_builds_a_profile():
    """Not a valid pool for contrast() (which enforces >=3), but
    build_profile itself must not special-case crash on a 1-sequence pool -
    the center is trivially that one sequence."""
    profile = build_profile([["a", "b", "c"]])

    assert profile.center_index == 0
    assert len(profile.columns) == 3
    assert profile.column_top(0, k=1)[0][0] == "a"
