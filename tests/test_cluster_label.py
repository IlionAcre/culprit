"""Tests for `cluster_label.py`. The two tests that matter most (CLAUDE.md's
L5 section): labeling makes exactly `len(clusters)` LLM calls, and a second
pass with unchanged state skips relabeling entirely. `litellm.completion` is
monkeypatched at the dotted attribute per this project's convention.
"""

import json

from culprit.cluster_label import (
    ClusterLabel,
    ClusterLabelState,
    label_clusters,
    needs_relabel,
)
from culprit.synth_results import make_diagnosis, make_signal

_NUMBER_WORDS = ["zero", "one", "two", "three", "four"]


def _verdict_response(label="a label", description="a description", suggested_fix="a fix"):
    class _Message:
        content = json.dumps({"label": label, "description": description, "suggested_fix": suggested_fix})

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


def _fenced_verdict_response(label="fenced label"):
    class _Message:
        content = "```json\n" + json.dumps(
            {"label": label, "description": "d", "suggested_fix": "f"}
        ) + "\n```"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


def _diagnoses_and_embeddings(n_clusters=3, per_cluster=3):
    diagnoses = []
    assignment = {}
    embeddings = {}
    for c in range(n_clusters):
        for i in range(per_cluster):
            diag_id = f"c{c}-{i}"
            diagnoses.append(
                make_diagnosis(
                    diagnosis_id=diag_id,
                    signals=[make_signal(detector=f"cluster_{_NUMBER_WORDS[c]}_detector")],
                )
            )
            assignment[diag_id] = c
            embeddings[diag_id] = [float(c), float(i) * 0.01, 0.0]
    # a noise diagnosis that must never be labeled
    noise_id = "noise-0"
    diagnoses.append(make_diagnosis(diagnosis_id=noise_id))
    assignment[noise_id] = -1
    embeddings[noise_id] = [99.0, 99.0, 99.0]
    return diagnoses, assignment, embeddings


def test_label_clusters_makes_exactly_one_llm_call_per_cluster(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=3, per_cluster=3)
    call_count = 0

    def fake_completion(model, messages):
        nonlocal call_count
        call_count += 1
        return _verdict_response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    results = label_clusters(diagnoses, assignment, embeddings, max_workers=1)

    assert call_count == 3  # exactly len(clusters), never per-diagnosis
    assert {r.cluster_label_id for r in results} == {0, 1, 2}
    assert all(r.error is None for r in results)


def test_label_clusters_never_labels_noise(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=1, per_cluster=3)
    monkeypatch.setattr("litellm.completion", lambda model, messages: _verdict_response())

    results = label_clusters(diagnoses, assignment, embeddings, max_workers=1)

    assert -1 not in {r.cluster_label_id for r in results}
    assert len(results) == 1  # the noise diagnosis never produces a ClusterLabel


def test_second_pass_with_unchanged_state_skips_relabeling(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=3, per_cluster=3)
    call_count = 0

    def fake_completion(model, messages):
        nonlocal call_count
        call_count += 1
        return _verdict_response(label=f"label-{call_count}")

    monkeypatch.setattr("litellm.completion", fake_completion)

    first_pass = label_clusters(diagnoses, assignment, embeddings, max_workers=1)
    assert call_count == 3

    previous = {
        r.cluster_label_id: ClusterLabelState(
            medoid_diagnosis_id=r.medoid_diagnosis_id,
            labeled_size=r.labeled_size,
            label=r.label,
            description=r.description,
            suggested_fix=r.suggested_fix,
        )
        for r in first_pass
    }

    second_pass = label_clusters(
        diagnoses, assignment, embeddings, max_workers=1, previous=previous
    )

    assert call_count == 3  # no new calls, every cluster was stable
    assert {r.label for r in second_pass} == {r.label for r in first_pass}


def test_second_pass_relabels_only_the_cluster_that_grew(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=2, per_cluster=3)
    call_count = 0

    def fake_completion(model, messages):
        nonlocal call_count
        call_count += 1
        return _verdict_response(label=f"label-{call_count}")

    monkeypatch.setattr("litellm.completion", fake_completion)

    first_pass = label_clusters(diagnoses, assignment, embeddings, max_workers=1)
    assert call_count == 2

    previous = {
        r.cluster_label_id: ClusterLabelState(
            medoid_diagnosis_id=r.medoid_diagnosis_id,
            labeled_size=r.labeled_size,
            label=r.label,
            description=r.description,
            suggested_fix=r.suggested_fix,
        )
        for r in first_pass
    }

    # cluster 0 grows from 3 to 5 diagnoses (>50% growth); cluster 1 is untouched
    grown_diagnoses = list(diagnoses)
    grown_assignment = dict(assignment)
    grown_embeddings = dict(embeddings)
    for i in range(3, 5):
        diag_id = f"c0-{i}"
        grown_diagnoses.append(make_diagnosis(diagnosis_id=diag_id))
        grown_assignment[diag_id] = 0
        grown_embeddings[diag_id] = [0.0, float(i) * 0.01, 0.0]

    label_clusters(
        grown_diagnoses, grown_assignment, grown_embeddings, max_workers=1, previous=previous
    )

    assert call_count == 3  # exactly one new call, for cluster 0 only


def test_a_bad_llm_response_produces_a_sentinel_error_without_costing_other_clusters(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=2, per_cluster=3)

    def fake_completion(model, messages):
        if "cluster_zero_detector" in messages[0]["content"]:
            raise RuntimeError("provider outage")
        return _verdict_response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    results = label_clusters(diagnoses, assignment, embeddings, max_workers=1)

    by_id = {r.cluster_label_id: r for r in results}
    assert by_id[0].error is not None
    assert by_id[1].error is None
    assert by_id[1].label == "a label"


def test_markdown_fenced_response_is_parsed(monkeypatch):
    diagnoses, assignment, embeddings = _diagnoses_and_embeddings(n_clusters=1, per_cluster=3)

    monkeypatch.setattr("litellm.completion", lambda model, messages: _fenced_verdict_response())

    results = label_clusters(diagnoses, assignment, embeddings, max_workers=1)

    assert results[0].label == "fenced label"
    assert results[0].error is None


def test_needs_relabel_true_when_no_previous_state():
    assert needs_relabel("m1", 5, None) is True


def test_needs_relabel_true_when_medoid_changed():
    previous = ClusterLabelState(
        medoid_diagnosis_id="m1", labeled_size=5, label="l", description="d", suggested_fix="f"
    )
    assert needs_relabel("m2", 5, previous) is True


def test_needs_relabel_true_when_size_grew_more_than_50_percent():
    previous = ClusterLabelState(
        medoid_diagnosis_id="m1", labeled_size=10, label="l", description="d", suggested_fix="f"
    )
    assert needs_relabel("m1", 16, previous) is True


def test_needs_relabel_false_when_stable():
    previous = ClusterLabelState(
        medoid_diagnosis_id="m1", labeled_size=10, label="l", description="d", suggested_fix="f"
    )
    assert needs_relabel("m1", 12, previous) is False
