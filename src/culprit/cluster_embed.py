"""Canonical diagnosis card rendering and embedding for L5 clustering.

`render_card` distills a `Diagnosis` down to the fields that actually
describe *the failure pattern* - failure class, the step signature involved,
expected versus observed, which detectors fired, and a short slice of the
rationale - deliberately excluding `trace_id`, `diagnosis_id`, `span_id`,
and any raw evidence text. Those excluded fields are exactly where unique
identifiers live, and identifiers are the load-bearing problem this module
exists to solve: without masking, two diagnoses of the *same underlying bug*
differ only by an order id, a UUID, or a timestamp, HDBSCAN sees them as
maximally dissimilar, and a batch of 40 duplicate failures comes back as 40
singleton clusters instead of one. `_mask_ids` is the second, belt-and-braces
layer of that same defense: even the fields kept (rationale text especially)
can carry embedded ids, so every card is regex-masked before it is embedded.
A dedicated test asserts two diagnoses differing only by an order id render
byte-identical cards - that test is what proves the masking actually works,
not just that the regexes compile.

Card construction is pure string work with no I/O, so it is cheap to call
twice (once to validate a diagnosis renders cleanly, once inside a batch) -
see `cluster.py`'s per-diagnosis error isolation, which relies on that.
"""

import re

from culprit.embed import EmbedFn
from culprit.signals import Diagnosis

# Order matters: UUIDs and dates are masked first with their own tags so the
# generic digit-bearing-token pass below does not need to special-case them.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_DATE_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)?\b"
)
# Any token containing at least one digit - covers order ids ("ORD-48213"),
# span/step ids, plain numbers, and mixed alnum identifiers in one pass.
# Deliberately broad: "numbers" is explicitly in scope per CLAUDE.md's L5
# section, not just structured ids.
_ID_RE = re.compile(r"\b\w*\d\w*\b")


def _mask_ids(text: str) -> str:
    text = _UUID_RE.sub("<UUID>", text)
    text = _DATE_RE.sub("<DATE>", text)
    text = _ID_RE.sub("<ID>", text)
    return text


def _step_signature(diagnosis: Diagnosis) -> str:
    """The observed signature at the root-cause step if L2 divergence data is
    present there, else the first divergence, else empty - a diagnosis with
    no `divergences` (e.g. L2 abstained, see `degraded_layers`) still renders
    a valid, if sparser, card rather than raising."""
    for d in diagnosis.divergences:
        if d.step_index == diagnosis.root_cause_step_index:
            return d.observed_signature
    if diagnosis.divergences:
        return diagnosis.divergences[0].observed_signature
    return ""


def _expected_signature(diagnosis: Diagnosis) -> str:
    for d in diagnosis.divergences:
        if d.step_index == diagnosis.root_cause_step_index and d.expected_signatures:
            return d.expected_signatures[0][0]
    if diagnosis.divergences and diagnosis.divergences[0].expected_signatures:
        return diagnosis.divergences[0].expected_signatures[0][0]
    return ""


def _detector_names(diagnosis: Diagnosis) -> str:
    """Sorted and deduped so two diagnoses whose L1 detectors fired in a
    different order, or fired the same detector twice, still render the same
    card - cluster structure should reflect which detectors fired, not how
    many times or in what order."""
    names = sorted({s.detector for s in diagnosis.signals if s.error is None})
    return ", ".join(names)


def render_card(diagnosis: Diagnosis) -> str:
    """Build the canonical, id-masked text card that stands in for a
    `Diagnosis` in embedding space. Field order is fixed so the same logical
    diagnosis always renders the same bytes."""
    raw = "\n".join(
        [
            f"failure_class: {diagnosis.failure_class or 'unknown'}",
            f"observed: {_step_signature(diagnosis)}",
            f"expected: {_expected_signature(diagnosis)}",
            f"detectors: {_detector_names(diagnosis)}",
            f"rationale: {diagnosis.rationale[:300]}",
        ]
    )
    return _mask_ids(raw)


def l2_normalize(vector: list[float]) -> list[float]:
    """`sklearn.cluster.HDBSCAN` does not accept `metric="cosine"` directly
    (see `cluster.py`'s docstring); normalizing every embedding to unit
    length first makes euclidean distance monotonically equivalent to cosine
    distance, so HDBSCAN over normalized vectors behaves like clustering on
    cosine similarity without needing a custom metric."""
    norm = sum(x * x for x in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [x / norm for x in vector]


def embed_cards(diagnoses: list[Diagnosis], *, embed_fn: EmbedFn) -> list[list[float]]:
    """Render every diagnosis's card and embed the whole batch in one
    `embed_fn` call (fastembed batches internally), then L2-normalize each
    resulting vector. Callers that need per-diagnosis error isolation should
    filter diagnoses with `render_card` first (see `cluster.py`) - this
    function assumes every diagnosis passed in already renders cleanly."""
    cards = [render_card(d) for d in diagnoses]
    vectors = embed_fn(cards)
    return [l2_normalize(v) for v in vectors]
