"""L5 clustering entrypoint: groups already-persisted diagnoses by embedding
their canonical diagnosis card and running HDBSCAN over the embeddings.

Stubbed here at its real signature during Phase 0 so downstream workstreams
can be written and tested against it before WS-F exists, including against
`culprit.synth_results.make_diagnosis` directly, with no dependency on WS-C
or WS-D. WS-F fills the body and additionally owns `cluster_embed.py` and
`cluster_label.py`.
"""

from culprit.embed import EmbedFn
from culprit.signals import Diagnosis


def cluster_diagnoses(
    diagnoses: list[Diagnosis], *, embed_fn: EmbedFn
) -> dict[str, int]:
    """Render each diagnosis's canonical card (`cluster_embed.render_card`),
    embed and L2-normalize the cards, and run
    `sklearn.cluster.HDBSCAN(metric="euclidean")` over them.

    Owned by WS-F. Returns `diagnosis_id -> cluster_id`; HDBSCAN's `-1`
    noise label is a valid value, not an error, a novel failure mode
    surfacing as unclustered is useful information.
    """
    raise NotImplementedError("owned by WS-F")
