"""L5 clustering entrypoint: groups already-persisted diagnoses by embedding
their canonical diagnosis card (`cluster_embed.render_card`) and running
HDBSCAN over the embeddings.

`sklearn.cluster.HDBSCAN` over L2-normalized embeddings, `metric="euclidean"`
- monotonically equivalent to cosine once every vector has unit length (see
`cluster_embed.l2_normalize`'s docstring), which sidesteps sklearn's HDBSCAN
not accepting `metric="cosine"` directly.

*Rejected: KMeans* - requires k up front and forces every diagnosis into a
cluster, with no way to say "this is genuinely novel."
*Rejected: agglomerative + a distance threshold* - the threshold is as
arbitrary as k, and still has no noise concept. HDBSCAN's `-1` noise label is
kept as a valid, meaningful value rather than special-cased away: a novel
failure mode surfacing as unclustered is useful information, not a defect.
*Rejected UMAP-then-HDBSCAN*, the standard large-scale recipe - adds
`umap-learn` plus `numba` for gains that only matter well past this
project's scale; documented as the first knob to turn if the diagnosis
corpus grows into the tens of thousands.
"""

import numpy as np
from sklearn.cluster import HDBSCAN

from culprit.cluster_embed import embed_cards, render_card
from culprit.embed import EmbedFn
from culprit.signals import Diagnosis

# Matches config.py's CulpritConfig.min_cluster_size fallback, so the
# hardcoded default here and the documented culprit.toml default agree.
_DEFAULT_MIN_CLUSTER_SIZE = 5


def _partition_renderable(
    diagnoses: list[Diagnosis],
) -> tuple[list[Diagnosis], dict[str, int]]:
    """Per-item error isolation (CLAUDE.md): a diagnosis whose card cannot
    be rendered (e.g. a malformed signal) must not take down clustering for
    every other diagnosis in the batch, and must not silently vanish from
    the result either. It is excluded from the embedding/HDBSCAN batch and
    assigned straight to noise (`-1`), the same value HDBSCAN itself uses
    for "does not belong to any cluster" - a card that cannot even be built
    certainly cannot be said to belong to a cluster."""
    renderable: list[Diagnosis] = []
    sentinel_assignment: dict[str, int] = {}
    for diagnosis in diagnoses:
        try:
            render_card(diagnosis)
        except Exception:  # noqa: BLE001 - deliberately broad, see docstring
            sentinel_assignment[diagnosis.diagnosis_id] = -1
            continue
        renderable.append(diagnosis)
    return renderable, sentinel_assignment


def cluster_diagnoses(
    diagnoses: list[Diagnosis],
    *,
    embed_fn: EmbedFn,
    min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
) -> dict[str, int]:
    """Render each diagnosis's canonical card, embed and L2-normalize the
    cards, and run `sklearn.cluster.HDBSCAN(metric="euclidean")` over them.

    Owned by WS-F. Returns `diagnosis_id -> cluster_id`; HDBSCAN's `-1`
    noise label is a valid value, not an error, a novel failure mode
    surfacing as unclustered is useful information.

    Fewer diagnoses than `min_cluster_size` cannot form any real cluster by
    HDBSCAN's own definition, so that case is short-circuited to "everyone
    is noise" rather than handed to HDBSCAN, which would do the same thing
    less legibly (and, for very small inputs, can raise on degenerate
    distance matrices).
    """
    if not diagnoses:
        return {}

    renderable, assignment = _partition_renderable(diagnoses)
    if not renderable:
        return assignment

    if len(renderable) < min_cluster_size:
        assignment.update({d.diagnosis_id: -1 for d in renderable})
        return assignment

    embeddings = np.array(embed_cards(renderable, embed_fn=embed_fn))
    # copy=True pinned explicitly: sklearn warns that the default flips from
    # False to True in 1.10, this avoids both the warning now and a silent
    # behavior change on that future upgrade.
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean", copy=True
    ).fit_predict(embeddings)

    assignment.update(
        {d.diagnosis_id: int(label) for d, label in zip(renderable, labels)}
    )
    return assignment
