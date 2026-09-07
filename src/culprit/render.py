"""Rich rendering for the CLI's read surfaces.

Its own module rather than more of `cli.py`, which CLAUDE.md already records
as the one module allowed past the ~200 line ceiling and which should not
grow further. Every function here takes the plain dicts `views.py` produces,
never a `Diagnosis`/`Adjudication` model, so the "one reviewed place decides
what a surface exposes" rule that `views.py`'s docstring states still holds
with a renderer in front of it.

Colour carries meaning and nothing else: the winning candidate is the only
highlighted row, evidence-free filler candidates are dimmed to show what the
shortlist padded with, and an abstention is yellow because it is a
non-answer rather than a failure. Rich drops styling automatically when
stdout is not a terminal, so piping this stays readable.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Diagnoses are read output, so they go to stdout; `console.print` is used
# for everything here rather than typer.echo because the tables and panels
# are the point.
console = Console()

_SOURCE_STYLES = {
    "l1": "cyan",
    "l2": "magenta",
    "both": "bold cyan",
    "filler": "dim",
    "fallback": "dim yellow",
}


def _confidence_style(confidence: float) -> str:
    """Confidence is calibrated against a measured precision-at-threshold
    table (see CLAUDE.md's L3 section), so the bands here are about how much
    weight a reader should put on the number, not decoration."""
    if confidence >= 0.60:
        return "green"
    if confidence >= 0.15:
        return "yellow"
    return "red"


def _step_line(step: dict, *, is_root: bool) -> Text:
    marker = "->" if is_root else "  "
    actor = step.get("actor") or "unknown"
    summary = step.get("summary") or "(no summary recorded)"
    duration = step.get("duration_ms")
    tail = f"  {duration:.0f}ms" if duration is not None else ""
    line = Text(f"{marker} {step['step_index']:>3}  {step['kind']:<10} {actor:<18} {summary}{tail}")
    if is_root:
        line.stylize("bold red")
    else:
        line.stylize("dim")
    return line


def verdict_panel(view: dict, context: list[dict] | None = None) -> Panel:
    """The headline: what broke, where, and how sure culprit is.

    `context` is `store_traces_query.step_context`'s output, or None when the
    trace's steps are unavailable. An abstention renders as its own shape
    rather than as `abstained=True` tacked onto a verdict, because an
    abstention is the pipeline declining to answer and reads as a different
    kind of result.
    """
    body = Text()

    if view["abstained"]:
        body.append("ABSTAINED", style="bold yellow")
        body.append("  culprit declined to name a step.\n")
        reason = view.get("abstain_reason") or "no reason recorded"
        body.append(f"reason: {reason}\n", style="yellow")
    else:
        body.append("step ", style="dim")
        body.append(f"{view['root_cause_step_index']}", style="bold red")
        body.append("   ")
        body.append(f"{view['failure_class']}\n", style="bold")

    calibrated = view["calibrated_confidence"]
    raw = view.get("confidence")
    body.append("confidence ", style="dim")
    body.append(f"{calibrated:.2f}", style=_confidence_style(calibrated))
    if raw is not None and abs(raw - calibrated) >= 0.005:
        # Both numbers are carried so a reader can see how far calibration
        # moved the model's own number, which is the point of measuring it.
        body.append(f"  (raw {raw:.2f})", style="dim")
    body.append("\n")

    if context:
        body.append("\n")
        root = view["root_cause_step_index"]
        for step in context:
            body.append(_step_line(step, is_root=step["step_index"] == root))
            body.append("\n")

    if view.get("rationale"):
        body.append("\n")
        body.append(view["rationale"].strip() + "\n")

    if view.get("counterfactual"):
        body.append("\ninstead: ", style="dim")
        body.append(view["counterfactual"].strip() + "\n", style="italic")

    degraded = view.get("degraded_layers") or []
    if degraded:
        body.append("\ndegraded layers: ", style="dim")
        body.append(", ".join(degraded) + "\n", style="yellow")
        body.append(
            "this diagnosis ran without them, so it is thinner than a full one\n",
            style="dim yellow",
        )

    return Panel(
        body,
        title=f"culprit  {view['trace_id']}",
        title_align="left",
        # A trace can carry several diagnoses (diagnoses.trace_id is
        # deliberately not unique), so the id stays visible to tell them
        # apart, just not at the top where the verdict belongs.
        subtitle=Text(view["diagnosis_id"], style="dim"),
        subtitle_align="right",
        expand=False,
    )


def candidates_table(view: dict) -> Table | None:
    """The shortlist L3 actually judged, in rank order.

    This is the architectural claim made visible: thousands of steps narrowed
    to a handful, with each candidate's source shown so a reader can see how
    many were real evidence and how many were evidence-free padding. Returns
    None when nothing was adjudicated, so the caller prints nothing rather
    than an empty table.
    """
    adjudications = view.get("adjudications") or []
    if not adjudications:
        return None

    table = Table(title="candidates adjudicated", title_justify="left", expand=False)
    table.add_column("step", justify="right")
    table.add_column("source")
    table.add_column("class")
    table.add_column("conf", justify="right")
    table.add_column("", justify="left")

    root = view.get("root_cause_step_index")
    for adjudication in adjudications:
        source = adjudication.get("source") or "?"
        is_winner = (
            not view["abstained"]
            and root is not None
            and adjudication.get("step_index") == root
            and adjudication.get("is_root_cause")
        )
        confidence = adjudication.get("confidence")
        row_style = "bold" if is_winner else _SOURCE_STYLES.get(source, "")
        table.add_row(
            str(adjudication.get("step_index")),
            Text(source, style=_SOURCE_STYLES.get(source, "")),
            adjudication.get("failure_class") or "-",
            f"{confidence:.2f}" if confidence is not None else "-",
            "<- named" if is_winner else "",
            style=row_style,
        )
    return table


def signals_table(view: dict, *, limit: int | None = None) -> Table | None:
    """The deterministic L1 signals behind the shortlist.

    Capped by `limit` in the default view because a real trace can fire
    dozens; `culprit show -v` passes None to see all of them.
    """
    signals = view.get("signals") or []
    if not signals:
        return None

    shown = signals if limit is None else signals[:limit]
    table = Table(title="signals fired", title_justify="left", expand=False)
    table.add_column("step", justify="right")
    table.add_column("detector")
    table.add_column("severity", justify="right")
    table.add_column("message")

    for signal in shown:
        severity = signal.get("severity")
        table.add_row(
            str(signal.get("step_index")),
            signal.get("detector") or "?",
            f"{severity:.2f}" if severity is not None else "-",
            (signal.get("message") or "").strip(),
        )
    if limit is not None and len(signals) > limit:
        table.caption = f"{len(signals) - limit} more, see culprit show -v"
        table.caption_justify = "left"
    return table


def render_diagnosis(
    view: dict, *, context: list[dict] | None = None, verbose: bool = False
) -> None:
    """Print one diagnosis: verdict, then the shortlist, then the evidence."""
    console.print(verdict_panel(view, context))

    candidates = candidates_table(view)
    if candidates is not None:
        console.print(candidates)

    signals = signals_table(view, limit=None if verbose else 5)
    if signals is not None:
        console.print(signals)


def render_no_diagnoses(trace_id: str) -> None:
    """The empty case, which is a routine state rather than an error: a
    trace is ingested before it is diagnosed, so this names the command that
    fills the gap instead of only reporting the absence."""
    console.print(
        Panel(
            Text.from_markup(
                f"No diagnoses yet for trace [bold]{trace_id}[/bold].\n\n"
                f"Run one with:  [bold]culprit run --trace-id {trace_id}[/bold]",
            ),
            expand=False,
        )
    )
