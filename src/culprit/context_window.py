"""Bounded per-candidate context packet for L3 adjudication: task goal, a
one-line-per-step spine, a zoom window around the candidate step, and
terminal-outcome evidence, hard-capped at `token_budget` tokens.

**Known contract gap, worked around here rather than by editing a frozen
file** (see the workstream report): `adjudicate.py`'s signature -
`adjudicate(candidates, trace, steps, *, call_fn, model)` - is already
committed to by the frozen `pipeline.py` and by `tests/test_synth_results.py`,
neither editable here. It passes `Trace` and `list[Step]` only, never
`Span`/`spans_by_id`, and `Step.summary` (schemas.py) is a short templated
sentence ("search_orders returns ok"), not raw payload text - so the plan's
"zoom window: full payloads, truncated head-and-tail" cannot be built from
raw span content at all. What fills the gap instead: `Candidate.signals`
already carries each L1 detector's own `Evidence.excerpt` (real, up to
400-char, head-and-tail-truncated payload snippets), and `Candidate.divergence`
carries L2's `observed_signature`/`expected_signatures` - but both are
attached only to the candidate's own step, not its i-2..i+2 neighbors, which
still only get `Step.summary`. Recommend a Foundation amendment adding
`spans_by_id` to `adjudicate()`, mirroring the one already applied to
`contrast()` after WS-D's handoff report, if full-payload neighbor evidence
is wanted.
"""

from dataclasses import dataclass

from culprit.schemas import Step, Trace
from culprit.signals import Candidate

_DEFAULT_TOKEN_BUDGET = 12_000
_DEFAULT_ZOOM_RADIUS = 2
_DEFAULT_GOAL_CHAR_LIMIT = 1500
# Chars-per-token approximation. No tokenizer dependency is available here
# (not in the frozen pyproject.toml dependency set, and adding one is out of
# scope for this workstream) - this heuristic is what lets the packet
# self-enforce a token budget without one. Documented as an estimate, not a
# measured count, same posture as confidence.py's calibration coefficients.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class ContextPacket:
    task_goal: str
    spine: str
    zoom: str
    terminal: str
    token_estimate: int
    # Step indices actually present in the rendered packet (spine, possibly
    # trimmed, plus the zoom window plus the terminal step). Feeds
    # confidence.py's cite_check: a rationale citing a step outside this set
    # cited something the model was never shown, i.e. confabulated it.
    visible_step_indices: frozenset[int]

    def render(self) -> str:
        return (
            f"TASK GOAL:\n{self.task_goal}\n\n"
            f"TRACE SPINE (one line per step, for global orientation):\n{self.spine}\n\n"
            f"ZOOM WINDOW (steps around the candidate under judgment, with any "
            f"detector evidence attached):\n{self.zoom}\n\n"
            f"TERMINAL STEP / TRACE OUTCOME:\n{self.terminal}\n"
        )


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN if text else 0


def _truncate(text: str, limit: int) -> str:
    """Head-and-tail truncation: the end of a tool result or message is very
    often where the interesting content lives (Evidence.excerpt's own
    contract in signals.py), so a plain head-cut would hide it. Below
    `limit=20` there is no room for the " ... " scaffold, so this falls back
    to a plain head-cut - only reachable via the last-resort budget
    safety net below, never via the normal per-section limits."""
    if len(text) <= limit:
        return text
    if limit < 20:
        return text[:limit]
    head = limit // 2 - 2
    tail = limit - head - 5
    return f"{text[:head]} ... {text[-tail:]}"


def _spine_line(step: Step) -> str:
    return f"{step.step_index:>4}  {step.signature}  {_truncate(step.summary, 80)}"


def _render_spine(steps: list[Step], budget_tokens: int) -> tuple[str, frozenset[int]]:
    """Budget enforcement drops the middle of the spine first, per the plan:
    the zoom window is where a specific candidate's actual evidence lives,
    so it is protected until the spine has nothing left to give. Keeps as
    many head and tail lines as the remaining budget allows and collapses
    the middle into a single marker line naming how many steps were
    dropped, so the model still knows the run had that many more steps even
    though it cannot see them."""
    ordered = sorted(steps, key=lambda s: s.step_index)
    lines = [_spine_line(s) for s in ordered]
    indices = [s.step_index for s in ordered]
    full = "\n".join(lines)
    if not lines or _estimate_tokens(full) <= budget_tokens:
        return full, frozenset(indices)

    half = (budget_tokens * _CHARS_PER_TOKEN) // 2
    head_n, head_chars = 0, 0
    while head_n < len(lines) and head_chars + len(lines[head_n]) + 1 <= half:
        head_chars += len(lines[head_n]) + 1
        head_n += 1
    tail_n, tail_chars = 0, 0
    while tail_n < len(lines) - head_n and tail_chars + len(lines[-(tail_n + 1)]) + 1 <= half:
        tail_chars += len(lines[-(tail_n + 1)]) + 1
        tail_n += 1

    head_lines, tail_lines = lines[:head_n], (lines[len(lines) - tail_n:] if tail_n else [])
    visible = frozenset(indices[:head_n] + (indices[len(indices) - tail_n:] if tail_n else []))
    omitted = len(lines) - head_n - tail_n
    if omitted <= 0:
        return "\n".join(head_lines + tail_lines), visible
    marker = f"...  [{omitted} step(s) omitted to stay within the context budget]  ..."
    return "\n".join(head_lines + [marker] + tail_lines), visible


def _zoom_block(step: Step, candidate: Candidate) -> str:
    lines = [
        f"--- step {step.step_index} ({step.kind.value}, actor={step.actor}, "
        f"{step.duration_ms:.0f}ms) ---",
        f"signature: {step.signature}",
        f"summary: {_truncate(step.summary, 400)}",
    ]
    if step.step_index == candidate.step_index:
        lines.append(f"flagged by: {candidate.source} (prior={candidate.prior:.2f})")
        for sig in candidate.signals:
            lines.append(
                f"[L1 signal] {sig.detector} (severity {sig.severity:.2f}, "
                f"category {sig.category}): {sig.message}"
            )
            for ev in sig.evidence:
                lines.append(f"  evidence @ {ev.field}: {_truncate(ev.excerpt, 400)}")
        if candidate.divergence is not None:
            d = candidate.divergence
            expected = ", ".join(f"{sig} ({p:.0%})" for sig, p in d.expected_signatures)
            lines.append(
                f"[L2 divergence] score={d.divergence_score:.2f} cliff={d.cliff_delta:.2f} "
                f"observed={d.observed_signature} expected=[{expected}]"
            )
    return "\n".join(lines)


def _render_zoom(candidate: Candidate, steps: list[Step], radius: int) -> tuple[str, frozenset[int]]:
    lo, hi = candidate.step_index - radius, candidate.step_index + radius
    in_window = sorted((s for s in steps if lo <= s.step_index <= hi), key=lambda s: s.step_index)
    if not in_window:
        return "(no steps in zoom window)", frozenset()
    text = "\n\n".join(_zoom_block(s, candidate) for s in in_window)
    return text, frozenset(s.step_index for s in in_window)


def _render_terminal(trace: Trace | None, steps: list[Step]) -> tuple[str, frozenset[int]]:
    outcome = trace.outcome.value if trace is not None else "unknown"
    if not steps:
        return f"trace outcome: {outcome} (no steps recorded)", frozenset()
    terminal = max(steps, key=lambda s: s.step_index)
    text = (
        f"trace outcome: {outcome}\n"
        f"terminal step {terminal.step_index} ({terminal.kind.value}, "
        f"actor={terminal.actor}): {terminal.signature} - "
        f"{_truncate(terminal.summary, 400)}"
    )
    return text, frozenset({terminal.step_index})


def _total_tokens(task_goal: str, zoom: str, terminal: str, spine: str) -> int:
    return sum(_estimate_tokens(t) for t in (task_goal, zoom, terminal, spine))


def build_context_packet(
    candidate: Candidate,
    trace: Trace | None,
    steps: list[Step],
    *,
    token_budget: int = _DEFAULT_TOKEN_BUDGET,
    zoom_radius: int = _DEFAULT_ZOOM_RADIUS,
    goal_char_limit: int = _DEFAULT_GOAL_CHAR_LIMIT,
) -> ContextPacket:
    """Assemble the four-section packet for one candidate. Terminal and zoom
    are built first and treated as fixed cost; the spine gets whatever
    budget remains (see `_render_spine`). The two `if total > token_budget`
    blocks below are a last-resort safety net for the pathological case
    where zoom/terminal alone already exceed the budget (e.g. many L1
    detectors firing at the same step, each with several evidence
    excerpts) - the cap is a real invariant every candidate's independent
    adjudication call relies on, not just a typical-case target."""
    task_goal = _truncate(
        (trace.task_goal if trace is not None else None) or "(no task goal recorded)",
        goal_char_limit,
    )
    zoom_text, zoom_visible = _render_zoom(candidate, steps, zoom_radius)
    terminal_text, terminal_visible = _render_terminal(trace, steps)

    fixed_tokens = _total_tokens(task_goal, zoom_text, terminal_text, "")
    spine_text, spine_visible = _render_spine(steps, max(0, token_budget - fixed_tokens))

    total = _total_tokens(task_goal, zoom_text, terminal_text, spine_text)
    if total > token_budget:
        overflow_chars = (total - token_budget) * _CHARS_PER_TOKEN
        zoom_text = _truncate(zoom_text, max(200, len(zoom_text) - overflow_chars))
        total = _total_tokens(task_goal, zoom_text, terminal_text, spine_text)
    if total > token_budget:
        overflow_chars = (total - token_budget) * _CHARS_PER_TOKEN
        terminal_text = _truncate(terminal_text, max(100, len(terminal_text) - overflow_chars))
        total = _total_tokens(task_goal, zoom_text, terminal_text, spine_text)

    return ContextPacket(
        task_goal=task_goal,
        spine=spine_text,
        zoom=zoom_text,
        terminal=terminal_text,
        token_estimate=total,
        visible_step_indices=zoom_visible | terminal_visible | spine_visible,
    )
