"""Tests for `cluster.py`. The synthetic-groups test is the HDBSCAN
recovery proof from the definition of done: three well-separated groups
plus noise, recovered without a real embedding model or network access -
`embed_fn` here is a hand-crafted fake keyed on which group a card belongs
to, not fastembed.
"""

import numpy as np

from culprit.cluster import cluster_diagnoses
from culprit.synth_results import make_diagnosis, make_signal

# Three near-orthogonal unit directions in 3D, jittered per diagnosis and
# renormalized by cluster_embed.embed_cards - angular separation of ~90
# degrees guarantees the groups stay well apart after L2 normalization,
# regardless of the small per-item jitter.
_GROUP_DIRECTIONS = {
    "group_a": np.array([1.0, 0.0, 0.0]),
    "group_b": np.array([0.0, 1.0, 0.0]),
    "group_c": np.array([0.0, 0.0, 1.0]),
}
# The negative octant, far (~1.7-1.9 euclidean, near the max possible 2.0 on
# a unit sphere) from every positive-axis group direction above - verified
# empirically that this keeps both noise points as HDBSCAN noise (-1) rather
# than absorbed into whichever real cluster happens to be nearest, which a
# smaller separation (e.g. an off-axis corner like [1,1,0]) did not reliably
# do at this small a sample size.
_NOISE_BASE = np.array([-1.0, -1.0, -1.0])
_NOISE_JITTERS = {
    "noise_one": np.array([0.3, -0.2, 0.1]),
    "noise_two": np.array([-0.3, 0.25, -0.15]),
}


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i, text in enumerate(texts):
        base = None
        for tag, direction in _GROUP_DIRECTIONS.items():
            if f"detectors: {tag}" in text:
                jitter = np.array(
                    [0.01 * (i % 3), 0.01 * ((i + 1) % 3), 0.01 * ((i + 2) % 3)]
                )
                base = direction + jitter
                break
        if base is None:
            for tag, jitter in _NOISE_JITTERS.items():
                if f"detectors: {tag}" in text:
                    base = _NOISE_BASE + jitter
                    break
        assert base is not None, f"no group tag found in card: {text!r}"
        vectors.append(base.tolist())
    return vectors


def _make_group(tag: str, n: int) -> list:
    return [
        make_diagnosis(
            diagnosis_id=f"{tag}-{i}",
            signals=[make_signal(detector=tag)],
            rationale=f"synthetic case {i} for {tag}",
        )
        for i in range(n)
    ]


def test_hdbscan_recovers_three_well_separated_groups_plus_noise():
    diagnoses = (
        _make_group("group_a", 4)
        + _make_group("group_b", 4)
        + _make_group("group_c", 4)
        + [
            make_diagnosis(diagnosis_id="noise_one-0", signals=[make_signal(detector="noise_one")]),
            make_diagnosis(diagnosis_id="noise_two-0", signals=[make_signal(detector="noise_two")]),
        ]
    )

    assignment = cluster_diagnoses(diagnoses, embed_fn=_fake_embed_fn, min_cluster_size=3)

    assert set(assignment.keys()) == {d.diagnosis_id for d in diagnoses}

    labels_a = {assignment[f"group_a-{i}"] for i in range(4)}
    labels_b = {assignment[f"group_b-{i}"] for i in range(4)}
    labels_c = {assignment[f"group_c-{i}"] for i in range(4)}
    assert len(labels_a) == 1 and -1 not in labels_a
    assert len(labels_b) == 1 and -1 not in labels_b
    assert len(labels_c) == 1 and -1 not in labels_c
    assert labels_a != labels_b != labels_c
    assert labels_a.isdisjoint(labels_b)
    assert labels_a.isdisjoint(labels_c)
    assert labels_b.isdisjoint(labels_c)

    assert assignment["noise_one-0"] == -1
    assert assignment["noise_two-0"] == -1


def test_cluster_diagnoses_returns_empty_dict_for_empty_input():
    assert cluster_diagnoses([], embed_fn=_fake_embed_fn) == {}


def test_cluster_diagnoses_short_circuits_below_min_cluster_size():
    diagnoses = _make_group("group_a", 2)

    assignment = cluster_diagnoses(diagnoses, embed_fn=_fake_embed_fn, min_cluster_size=5)

    assert assignment == {"group_a-0": -1, "group_a-1": -1}


def test_one_unrenderable_diagnosis_does_not_break_the_rest_of_the_batch(monkeypatch):
    diagnoses = _make_group("group_a", 4)

    import culprit.cluster as cluster_module

    real_render_card = cluster_module.render_card
    poisoned_id = diagnoses[0].diagnosis_id

    def flaky_render_card(diagnosis):
        if diagnosis.diagnosis_id == poisoned_id:
            raise ValueError("boom")
        return real_render_card(diagnosis)

    monkeypatch.setattr(cluster_module, "render_card", flaky_render_card)

    assignment = cluster_diagnoses(diagnoses, embed_fn=_fake_embed_fn, min_cluster_size=3)

    assert assignment[poisoned_id] == -1
    assert set(assignment.keys()) == {d.diagnosis_id for d in diagnoses}


def test_cluster_diagnoses_embed_fn_is_called_once_per_batch_not_per_diagnosis():
    diagnoses = _make_group("group_a", 3) + _make_group("group_b", 3)
    call_count = 0

    def counting_embed_fn(texts):
        nonlocal call_count
        call_count += 1
        return _fake_embed_fn(texts)

    cluster_diagnoses(diagnoses, embed_fn=counting_embed_fn, min_cluster_size=3)

    assert call_count == 1
