"""Module-level dict registry mapping vocabulary name -> `VocabModule`
instance. The extension point the whole `vocab/` subpackage exists for:
adding a new vocabulary means adding one module under `vocab/` and one line
here, nothing else in the codebase changes. Sole owner per CLAUDE.md's "one
owner per plugin registry" rule - only WS-A (this workstream) edits this
file.
"""

from culprit.vocab import openinference, otel_genai, raw_upload
from culprit.vocab.base import VocabModule

VOCABULARIES: dict[str, VocabModule] = {
    otel_genai.VOCAB.name: otel_genai.VOCAB,
    openinference.VOCAB.name: openinference.VOCAB,
    raw_upload.VOCAB.name: raw_upload.VOCAB,
}


def detect(raw: dict) -> str:
    """The vocabulary name whose module claims `raw`, or "unknown" if none
    does. `normalize.py` treats "unknown" as the signal to set
    `normalize_error` while still retaining every raw attribute (CLAUDE.md:
    raw attributes are never discarded, even for an unrecognized span)."""
    for name, module in VOCABULARIES.items():
        if module.matches(raw):
            return name
    return "unknown"
