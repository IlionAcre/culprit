"""instruction_noncompliance: detects when an LLM step's response omits a
literal token the current-turn instruction explicitly required.

TRAIL's formatting and instruction-violation annotations (42% of the
benchmark) are the motivating case: the plan-generation prompt tells the
model to end with `<end_plan>`, and the generated plan frequently omits it.
The check extracts required tokens from the instruction, then tests for
their presence in the response, which keeps L1 deterministic and cheap.
Reach is bounded by shape, not by tag name: `_INSTRUCTION_RE` captures
angle-bracket tags (`<end_plan>`), bracketed markers (`[[DONE]]`),
hash-delimited markers (`###END###`), fenced blocks (```` ```json ````), and
quoted literals (`"FINISHED"`). `_HTML_NOISE` drops tags that are markup
rather than constraints.
"""

import re

from culprit.detectors.base import DetectorContext
from culprit.schemas import LlmPayload, SpanKind
from culprit.signals import Evidence, Signal
from culprit.taxonomy import FailureClass

# Instruction phrases that introduce a required literal. The catalogue uses
# these exact phrases; "write the ... tag" and "append the ... tag" are
# included because TRAIL's actual prompts use them (e.g. "write the
# '\\n<end_plan>' tag and stop there"), which is semantically an "end with"
# constraint.
_INSTRUCTION_RE = re.compile(
    r"(?:end with|must contain|respond with|use the format|wrap in|write the|append the)\s+"
    r"(?:"
    r"['\"`]?\\?n?\s*(<[^>\s]+[^>]*>)|"
    r"['\"`]?\s*(\[\[[^\]\s]+\]\])|"
    r"['\"`]?\s*(###[^#\s]+###)|"
    r"['\"`]?\s*(```[a-zA-Z0-9_-]*)|"
    r"['\"`]([A-Za-z0-9_-]{3,})['\"`]"
    r")",
    re.IGNORECASE,
)

# HTML-ish noise that sometimes appears in prompts but is not a meaningful
# required-output token.
_HTML_NOISE = {
    "<i>", "<b>", "<p>", "<br>", "<ul>", "<li>", "<ol>", "<a>",
    "<div>", "<span>", "<strong>", "<em>", "<pre>", "<code>",
}

_SEVERITY = 0.65


def _current_instruction_text(payload: LlmPayload) -> str:
    """Current-turn instruction text: the system prompt plus the first user
    message. Later user messages in TRAIL traces are conversation history
    (facts list, prior plan, tool observations), not fresh instructions, so
    ignoring them avoids re-firing on stale constraints every turn."""
    system_text = ""
    first_user_text = ""
    system_msgs = [m for m in payload.request_messages if m.role == "system"]
    user_msgs = [m for m in payload.request_messages if m.role == "user"]
    if system_msgs:
        system_text = system_msgs[0].content or ""
    if user_msgs:
        first_user_text = user_msgs[0].content or ""
    return f"{system_text}\n{first_user_text}"


def instruction_noncompliance(ctx: DetectorContext) -> list[Signal]:
    """Fire when a required output literal is missing from an LLM response.

    Each LLM step is judged against its own current-turn instruction, so a
    trace that repeats a constraint and violates it repeatedly earns one
    signal per violating step. Volume is bounded naturally: only steps whose
    instruction names a literal can fire at all, and `merge_candidates`
    truncates the shortlist to `max_candidates`.
    """
    signals: list[Signal] = []

    for step in ctx.steps:
        if step.kind != SpanKind.LLM:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, LlmPayload):
            continue

        payload = span.payload
        instruction_text = _current_instruction_text(payload)
        required: list[str] = []
        for match in _INSTRUCTION_RE.finditer(instruction_text):
            literal = next((g for g in match.groups() if g is not None), None)
            if not literal or literal.lower() in _HTML_NOISE:
                continue
            if literal not in required:
                required.append(literal)
        if not required:
            continue

        response_lower = " ".join(m.content or "" for m in payload.response_messages).lower()
        missing = [lit for lit in required if lit.lower() not in response_lower]
        if not missing:
            continue

        signals.append(Signal(
            detector="instruction_noncompliance",
            step_index=step.step_index,
            span_id=step.span_id,
            severity=_SEVERITY,
            category=FailureClass.CONSTRAINT_VIOLATION.value,
            message=f"Required literal(s) missing from response: {', '.join(missing)}",
            evidence=[Evidence(
                span_id=step.span_id,
                step_index=step.step_index,
                field="payload.request_messages",
                excerpt=instruction_text[:200],
            )],
        ))

    return signals
