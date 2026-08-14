"""Task fingerprinting: `agent_key` groups traces run by the same agent
config (framework + actor set + tool universe), `task_key` groups traces
attempting the same underlying task after normalizing away the specific
values that make two occurrences of the same task look different ("Refund
order 88213" and "Refund order 90114" must hash identically).

Both keys feed reference-pool resolution (`store_traces.nearest_successful`,
owned by WS-B): exact `(agent_key, task_key)` match first, falling back to
`agent_key` plus embedding kNN. WS-D never performs that lookup itself, only
computes the two keys the lookup is keyed on.

`agent_key` needs the actor set and tool universe, neither of which lives on
`Trace` (see `schemas.py`: `Trace` intentionally carries no agent/tool list).
The interface note in the plan writes it as `agent_key(trace) -> str`, but
that is unsatisfiable from `Trace` alone; `steps` is where actor names and
tool names actually live, entirely recoverable from `Step.actor` and
`Step.signature` (parsed the same way `signature.py` reads them) without
needing span payloads. Taking `steps` as a second argument is the one place
this module's real signature diverges from the plan's shorthand.
"""

import hashlib
import re

from culprit.schemas import Step, Trace

_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INT_RE = re.compile(r"\b\d+\b")
_WS_RE = re.compile(r"\s+")


def _tool_name(signature: str) -> str | None:
    """Pulls the tool/action name out of a coarse signature token, using
    the same `kind:actor:...` shape `signature.py` and `linearize.py` both
    read. Returns None for tokens that name no concrete tool (`agent:...`,
    `llm:{actor}:plan`, `llm:{actor}:answer`, `retr:...`)."""
    parts = signature.split(":")
    if parts[0] == "llm" and len(parts) >= 4 and parts[2] == "call":
        return parts[3]
    if parts[0] == "tool" and len(parts) >= 3:
        return parts[2]
    return None


def agent_key(trace: Trace, steps: list[Step]) -> str:
    """`sha1(framework | sorted(actors) | sorted(tool_universe))`. Two
    traces from the same agent deployment hash identically regardless of
    which task they were running, which is exactly the grouping the
    reference-pool lookup needs before it narrows further by task."""
    actors = sorted({s.actor for s in steps})
    tools = sorted({t for s in steps if (t := _tool_name(s.signature)) is not None})
    raw = "|".join([trace.framework or "", ",".join(actors), ",".join(tools)])
    return hashlib.sha1(raw.encode()).hexdigest()


def task_key(goal: str | None) -> str:
    """`sha1(normalize(goal))`. Normalization lowercases, replaces UUIDs,
    ISO dates, emails, and URLs with typed placeholders (in that order,
    since a UUID or date substring would otherwise get eaten by the bare
    integer pass first), collapses remaining integers, and squashes
    whitespace, so two instances of the same task template with different
    concrete values collide on the same key."""
    text = (goal or "").lower()
    text = _URL_RE.sub("<url>", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = _UUID_RE.sub("<uuid>", text)
    text = _DATE_RE.sub("<date>", text)
    text = _INT_RE.sub("<int>", text)
    text = _WS_RE.sub(" ", text).strip()
    return hashlib.sha1(text.encode()).hexdigest()
