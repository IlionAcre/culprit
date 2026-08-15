"""Tests for `store_cluster_labels.py` (INTEGRATION_ITEMS.md item 3: cluster
label persistence, and the closing half of item 1: proving `needs_relabel`'s
skip actually fires once identity survives a pass). Same two-tier structure
as the rest of this pair of modules: offline tests against a fake connection,
plus one stateful fake DB that chains `store_clusters.write_cluster_assignments`
and `label_and_persist_clusters` across two passes end to end.
"""

import contextlib
import json
import uuid

from culprit.cluster_label import ClusterLabelState
from culprit.store_cluster_labels import (
    label_and_persist_clusters,
    read_cluster_label_states,
    write_cluster_labels,
)
from culprit.store_clusters import write_cluster_assignments
from culprit.synth_results import make_diagnosis, make_signal

# --- offline: fake connection/cursor -----------------------------------


class _FakeCursor:
    def __init__(self, results: list[list] | None = None):
        self._results = list(results or [])
        self._current: list = []
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self._current = self._results.pop(0) if self._results else []
        return self

    def executemany(self, sql, seq):
        seq = list(seq)
        self.calls.append((sql, seq))
        return self

    def fetchall(self):
        return self._current

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, results: list[list] | None = None):
        self.cur = _FakeCursor(results)

    def cursor(self, row_factory=None):
        return self.cur

    def transaction(self):
        return contextlib.nullcontext()


def _conn_fn(conn):
    return lambda: conn


def _verdict_response(label="a label", description="a description", suggested_fix="a fix"):
    class _Message:
        content = json.dumps({"label": label, "description": description, "suggested_fix": suggested_fix})

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    return _Response()


# --- offline: read_cluster_label_states -------------------------------------


def test_read_cluster_label_states_returns_empty_for_empty_input():
    conn = _FakeConnection()
    assert read_cluster_label_states(_conn_fn(conn), {}) == {}


def test_read_cluster_label_states_maps_rows_back_to_this_pass_labels():
    cid = uuid.uuid4()
    row = {
        "cluster_id": cid,
        "label": "l",
        "description": "d",
        "suggested_fix": "f",
        "medoid_diagnosis_id": "m1",
        "labeled_size": 5,
    }
    conn = _FakeConnection(results=[[row]])

    states = read_cluster_label_states(_conn_fn(conn), {3: cid})

    assert states == {3: ClusterLabelState(medoid_diagnosis_id="m1", labeled_size=5, label="l", description="d", suggested_fix="f")}


# --- offline: write_cluster_labels -------------------------------------


def test_write_cluster_labels_skips_errored_and_reused_results():
    """A `ClusterLabel` with `error` set must never overwrite a prior good
    label, and a result `needs_relabel` would call "still stable" (reused
    verbatim from `previous`) must not bump `labeled_at` for nothing - see
    module docstring for why."""
    from culprit.cluster_label import ClusterLabel

    cid_ok, cid_err, cid_reused = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    cluster_ids = {0: cid_ok, 1: cid_err, 2: cid_reused}
    previous = {
        2: ClusterLabelState(medoid_diagnosis_id="m2", labeled_size=3, label="old", description="d", suggested_fix="f")
    }
    labels = [
        ClusterLabel(0, "new label", "new desc", "new fix", "m0", 3, error=None),
        ClusterLabel(1, "", "", "", "m1", 3, error="RuntimeError: boom"),
        # same medoid + size as `previous[2]` -> needs_relabel is False -> reused, must be skipped
        ClusterLabel(2, "old", "d", "f", "m2", 3, error=None),
    ]
    conn = _FakeConnection()

    write_cluster_labels(_conn_fn(conn), cluster_ids, labels, previous)

    (update_call,) = [c for c in conn.cur.calls if "UPDATE clusters" in c[0]]
    written_cluster_ids = {row[-1] for row in update_call[1]}
    assert written_cluster_ids == {cid_ok}


def test_write_cluster_labels_is_a_noop_when_nothing_needs_writing():
    conn = _FakeConnection()

    write_cluster_labels(_conn_fn(conn), {}, [])

    assert conn.cur.calls == []


# --- the proof: needs_relabel's skip fires across two real passes ----------


class _FakeClusterDB:
    """Stateful double for the two tables `store_clusters.py` and
    `store_cluster_labels.py` actually touch, shared across calls (unlike
    the canned-results fakes above) - the only way to prove this pair of
    modules closes the loop `needs_relabel` needs: that a cluster's identity
    *and* its labeled state both survive an independent second pass, even
    when HDBSCAN hands that pass a different integer label."""

    def __init__(self):
        self.diagnosis_cluster_id: dict[str, uuid.UUID | None] = {}
        self.clusters: dict[uuid.UUID, dict] = {}

    def conn_fn(self):
        return _StatefulConnection(self)


class _StatefulCursor:
    def __init__(self, db: _FakeClusterDB):
        self.db = db
        self._current: list = []

    def execute(self, sql, params=None):
        if "SELECT diagnosis_id, cluster_id FROM diagnoses" in sql:
            (diagnosis_ids,) = params
            self._current = [
                (did, self.db.diagnosis_cluster_id[did])
                for did in diagnosis_ids
                if self.db.diagnosis_cluster_id.get(did) is not None
            ]
        elif "SELECT cluster_id, label, description, suggested_fix" in sql:
            (cluster_ids,) = params
            self._current = [
                {"cluster_id": cid, **self.db.clusters[cid]}
                for cid in cluster_ids
                if self.db.clusters.get(cid, {}).get("label") is not None
            ]
        else:
            raise AssertionError(f"unexpected SQL in fake cursor: {sql!r}")
        return self

    def executemany(self, sql, seq):
        seq = list(seq)
        if "INSERT INTO clusters" in sql:
            for cluster_id, run_id, size in seq:
                row = self.db.clusters.setdefault(
                    cluster_id,
                    {"label": None, "description": None, "suggested_fix": None, "medoid_diagnosis_id": None, "labeled_size": None},
                )
                row.update(run_id=run_id, size=size)
        elif "UPDATE diagnoses SET cluster_id" in sql:
            for cluster_id, diagnosis_id in seq:
                self.db.diagnosis_cluster_id[diagnosis_id] = cluster_id
        elif "UPDATE clusters" in sql and "SET label" in sql:
            for label, description, suggested_fix, medoid_diagnosis_id, labeled_size, cluster_id in seq:
                self.db.clusters[cluster_id].update(
                    label=label, description=description, suggested_fix=suggested_fix,
                    medoid_diagnosis_id=medoid_diagnosis_id, labeled_size=labeled_size,
                )
        else:
            raise AssertionError(f"unexpected SQL in fake cursor: {sql!r}")
        return self

    def fetchall(self):
        return self._current

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _StatefulConnection:
    def __init__(self, db: _FakeClusterDB):
        self.db = db

    def cursor(self, row_factory=None):
        return _StatefulCursor(self.db)

    def transaction(self):
        return contextlib.nullcontext()


def test_needs_relabel_skip_fires_on_a_second_pass_after_real_persistence(monkeypatch):
    """End to end: cluster five diagnoses under label 0, persist identity and
    a label (one LLM call). Recluster the *same* diagnoses under label 7
    (HDBSCAN reshuffled, nothing about the cluster's content changed) and
    label again - identity resolves back to the same cluster_id
    (store_clusters, item 1) and that cluster_id's medoid/size are unchanged,
    so `needs_relabel` returns False and this second pass makes zero LLM
    calls. This is the test the integration item explicitly asked for."""
    db = _FakeClusterDB()
    diagnoses = [
        make_diagnosis(diagnosis_id=f"d{i}", signals=[make_signal(detector="group_a")])
        for i in range(5)
    ]
    embed_fn = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]  # identical -> stable medoid

    call_count = 0

    def fake_completion(model, messages):
        nonlocal call_count
        call_count += 1
        return _verdict_response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    assignment_pass_one = {d.diagnosis_id: 0 for d in diagnoses}
    cluster_ids_one = write_cluster_assignments(db.conn_fn, assignment_pass_one)
    labels_one = label_and_persist_clusters(
        db.conn_fn, diagnoses, assignment_pass_one, cluster_ids_one, embed_fn=embed_fn, max_workers=1
    )
    assert call_count == 1
    assert labels_one[0].error is None

    assignment_pass_two = {d.diagnosis_id: 7 for d in diagnoses}  # different HDBSCAN label
    cluster_ids_two = write_cluster_assignments(db.conn_fn, assignment_pass_two)
    labels_two = label_and_persist_clusters(
        db.conn_fn, diagnoses, assignment_pass_two, cluster_ids_two, embed_fn=embed_fn, max_workers=1
    )

    assert cluster_ids_two[7] == cluster_ids_one[0]  # item 1: same stable identity
    assert call_count == 1  # item 3 + item 1 together: the skip actually fired
    assert labels_two[0].label == labels_one[0].label  # reused label content, not re-derived


def test_needs_relabel_skip_does_not_fire_when_the_cluster_actually_grew(monkeypatch):
    """The negative control for the test above: growing a cluster past the
    50% threshold must still trigger a real second LLM call, proving the
    skip is conditional and not just permanently short-circuited."""
    db = _FakeClusterDB()
    diagnoses = [
        make_diagnosis(diagnosis_id=f"d{i}", signals=[make_signal(detector="group_a")])
        for i in range(5)
    ]
    embed_fn = lambda texts: [[1.0, 0.0, 0.0] for _ in texts]

    call_count = 0

    def fake_completion(model, messages):
        nonlocal call_count
        call_count += 1
        return _verdict_response()

    monkeypatch.setattr("litellm.completion", fake_completion)

    assignment_one = {d.diagnosis_id: 0 for d in diagnoses}
    cluster_ids_one = write_cluster_assignments(db.conn_fn, assignment_one)
    label_and_persist_clusters(
        db.conn_fn, diagnoses, assignment_one, cluster_ids_one, embed_fn=embed_fn, max_workers=1
    )
    assert call_count == 1

    grown = diagnoses + [
        make_diagnosis(diagnosis_id=f"d{i}", signals=[make_signal(detector="group_a")])
        for i in range(5, 9)  # 5 -> 9 members, > 50% growth
    ]
    assignment_two = {d.diagnosis_id: 7 for d in grown}
    cluster_ids_two = write_cluster_assignments(db.conn_fn, assignment_two)
    label_and_persist_clusters(
        db.conn_fn, grown, assignment_two, cluster_ids_two, embed_fn=embed_fn, max_workers=1
    )

    assert call_count == 2  # growth past the threshold forces a real re-label
