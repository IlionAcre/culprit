"""instruction_noncompliance: detects when an LLM step's response omits a
literal token the current-turn instruction explicitly required.

TRAIL's formatting and instruction-violation annotations (42% of the
benchmark) are the motivating case: the plan-generation prompt tells the
model to end with `<end_plan>`, and the generated plan frequently omits it.
The check is deliberately literal: extract required tokens from the
instruction, then test for their presence in the response. This keeps L1
deterministic and cheap.
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
    r"['\"`]?\\?n?\s*"
    r"(<[^>\s]+[^>]*>)",
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


def _literal_category(literal: str) -> str | None:
    """Bucket a required literal into 'plan', 'code', or None. Only these two
    delimiter categories appear in TRAIL with enough volume and precision to
    move the evidenced rate; other literal shapes are left for future work."""
    lower = literal.lower()
    if "plan" in lower:
        return "plan"
    if "code" in lower:
        return "code"
    return None


def instruction_noncompliance(ctx: DetectorContext) -> list[Signal]:
    """Fire when a required output literal is missing from an LLM response.

    Per trace, only the first plan step and the first code step are flagged.
    TRAIL replans and multi-step code blocks repeat the same constraint, and
    firing on every repetition floods the shortlist without adding precision.
    """
    signals: list[Signal] = []
    fired_categories: set[str] = set()

    for step in ctx.steps:
        if step.kind != SpanKind.LLM:
            continue
        span = ctx.spans_by_id.get(step.span_id)
        if span is None or not isinstance(span.payload, LlmPayload):
            continue

        payload = span.payload
        instruction_text = _current_instruction_text(payload)
        required_by_category: dict[str, list[str]] = {}
        for match in _INSTRUCTION_RE.finditer(instruction_text):
            literal = match.group(1)
            if literal.lower() in _HTML_NOISE:
                continue
            category = _literal_category(literal)
            if category is None:
                continue
            required_by_category.setdefault(category, []).append(literal)

        if not required_by_category:
            continue

        # Only fire once per category per trace to control volume.
        categories_to_fire = [
            cat for cat in required_by_category if cat not in fired_categories
        ]
        if not categories_to_fire:
            continue

        response_text = " ".join(m.content or "" for m in payload.response_messages)
        response_lower = response_text.lower()
        missing: list[str] = []
        for category in categories_to_fire:
            category_literals = required_by_category[category]
            category_missing = [
                lit for lit in category_literals if lit.lower() not in response_lower
            ]
            if category_missing:
                missing.append(category_missing[0])
                fired_categories.add(category)

        if missing:
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
