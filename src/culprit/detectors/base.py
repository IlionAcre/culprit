"""Shared detector infrastructure: the `DetectorContext` every detector reads
from, the `Detector` protocol every detector implements, and small text
helpers used across the family modules.

`DetectorContext` is built exactly once per `run_detectors()` call (see
`run_detectors.py`) so detectors never each recompute the same signature
run-length-encoding, argument/result hashes, and token series.
CLAUDE.md's "L1 detectors" section is why that matters: 5ms vs 200ms per
trace.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Protocol

from culprit.schemas import (
    AgentPayload,
    LlmPayload,
    RetrievalPayload,
    Span,
    SpanKind,
    Step,
    ToolPayload,
    Trace,
)
from culprit.signals import Signal

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str | None) -> frozenset[str]:
    """Lowercased alphanumeric word tokens, the alphabet every token-overlap
    heuristic in `retrieval.py` compares over."""
    if not text:
        return frozenset()
    return frozenset(_WORD_RE.findall(text.lower()))


def truncate(text: str, limit: int = 400) -> str:
    """Head-and-tail truncation, per `Evidence.excerpt`'s contract: the end
    of a tool result or message is very often where the interesting content
    lives, so a plain head-truncate would hide it."""
    if len(text) <= limit:
        return text
    head = limit // 2 - 2
    tail = limit - head - 5
    return f"{text[:head]} ... {text[-tail:]}"


def _hash(value: object) -> str:
    """Stable content hash for loop detection (`loops.py`): args/results need
    a cheap equality-comparable digest, not a security property."""
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _step_text(span: Span) -> str:
    """The text a substring-provenance or token-overlap check should search:
    everything a human could plausibly have derived this step's content
    from. Feeds `DetectorContext.step_text`, which `provenance.py`'s
    substring search reads directly."""
    p = span.payload
    if isinstance(p, ToolPayload):
        return " ".join([p.result_text, json.dumps(p.arguments, default=str)])
    if isinstance(p, LlmPayload):
        return " ".join(m.content or "" for m in p.request_messages + p.response_messages)
    if isinstance(p, RetrievalPayload):
        return " ".join([p.query] + [d.content_preview for d in p.documents])
    if isinstance(p, AgentPayload):
        return " ".join([p.input_text, p.output_text])
    return ""


@dataclass(frozen=True)
class DetectorContext:
    """Everything every detector reads, computed once. Every field
    exists because at least one detector family needs it; see the
    per-detector docstrings in the family modules for which."""

    trace: Trace
    steps: list[Step]
    spans_by_id: dict[str, Span]
    args_hash_by_step: dict[int, str | None]
    result_hash_by_step: dict[int, str | None]
    signature_runs: list[tuple[str, int, int]]  # (signature, start_index, length)
    token_series: dict[int, tuple[int | None, int | None, int | None]]  # prompt/completion/total
    tool_universe: frozenset[str]
    goal_terms: frozenset[str]
    step_text: list[str]  # step_text[i] is step i's own searchable text
    llm_steps_by_actor: dict[str, list[int]]

    def had_provenance_before(self, token: str, step_index: int) -> bool:
        """Whether `token` appears verbatim (case-insensitive) in the task
        goal or in any step strictly before `step_index`. Used by
        `provenance.py`'s `parameter_drift`, the one detector that needs a
        substring search rather than a token-set overlap."""
        needle = token.lower()
        if needle in (self.trace.task_goal or "").lower():
            return True
        return any(needle in self.step_text[j].lower() for j in range(step_index))


class Detector(Protocol):
    def __call__(self, ctx: DetectorContext) -> list[Signal]: ...


def build_context(trace: Trace, steps: list[Step], spans_by_id: dict[str, Span]) -> DetectorContext:
    args_hash: dict[int, str | None] = {}
    result_hash: dict[int, str | None] = {}
    token_series: dict[int, tuple[int | None, int | None, int | None]] = {}
    tool_names: set[str] = set()
    step_text: list[str] = []
    llm_steps_by_actor: dict[str, list[int]] = {}

    for step in steps:
        span = spans_by_id.get(step.span_id)
        step_text.append(_step_text(span) if span is not None else "")
        payload = span.payload if span is not None else None

        if step.kind == SpanKind.TOOL and isinstance(payload, ToolPayload):
            args_hash[step.step_index] = _hash(payload.arguments)
            result_hash[step.step_index] = _hash(payload.result_text)
            tool_names.add(payload.tool_name)
        else:
            args_hash[step.step_index] = None
            result_hash[step.step_index] = None

        if step.kind == SpanKind.LLM and isinstance(payload, LlmPayload):
            token_series[step.step_index] = (payload.prompt_tokens, payload.completion_tokens, payload.total_tokens)
            llm_steps_by_actor.setdefault(step.actor, []).append(step.step_index)

    signature_runs: list[tuple[str, int, int]] = []
    for step in steps:
        if signature_runs and signature_runs[-1][0] == step.signature:
            sig, start, length = signature_runs[-1]
            signature_runs[-1] = (sig, start, length + 1)
        else:
            signature_runs.append((step.signature, step.step_index, 1))

    return DetectorContext(
        trace=trace,
        steps=steps,
        spans_by_id=spans_by_id,
        args_hash_by_step=args_hash,
        result_hash_by_step=result_hash,
        signature_runs=signature_runs,
        token_series=token_series,
        tool_universe=frozenset(tool_names),
        goal_terms=tokens(trace.task_goal),
        step_text=step_text,
        llm_steps_by_actor=llm_steps_by_actor,
    )
