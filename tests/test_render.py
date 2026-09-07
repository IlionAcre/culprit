"""Rendering tests assert on content, never on layout.

Rich decides borders, padding, and wrapping from the console width, so a
test that pins those is testing the terminal rather than the renderer.
Everything here goes through `_text()`, which renders to a fixed-width
console and flattens the result, and then checks that the facts a reader
needs are present.
"""

from rich.console import Console

from culprit.render import (
    candidates_table,
    render_diagnosis,
    signals_table,
    verdict_panel,
)

_BOX_CHARS = "─│┌┐└┘├┤┬┴┼━┃╭╮╰╯"


def _text(renderable) -> str:
    console = Console(width=200, no_color=True, force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    stripped = "".join(" " if ch in _BOX_CHARS else ch for ch in capture.get())
    return " ".join(stripped.split())


def _view(**overrides) -> dict:
    base = {
        "diagnosis_id": "diag-1",
        "trace_id": "trace-1",
        "root_cause_step_index": 3,
        "root_cause_span_id": "span-3",
        "failure_class": "silent_empty_result_misread",
        "confidence": 0.90,
        "calibrated_confidence": 0.72,
        "abstained": False,
        "abstain_reason": None,
        "rationale": "The tool returned an empty result and the agent read it as success.",
        "counterfactual": "Retry the lookup or surface the empty result as an error.",
        "candidates_considered": 5,
        "degraded_layers": [],
        "signals": [],
        "divergences": [],
        "adjudications": [],
    }
    base.update(overrides)
    return base


def test_verdict_names_the_step_and_the_failure_class():
    out = _text(verdict_panel(_view()))

    assert "step 3" in out
    assert "silent_empty_result_misread" in out


def test_verdict_shows_the_raw_confidence_beside_the_calibrated_one():
    """Both numbers are carried through the schema so a reader can see how
    far calibration moved the model's own number; the renderer must not
    quietly drop one of them."""
    out = _text(verdict_panel(_view(confidence=0.90, calibrated_confidence=0.72)))

    assert "0.72" in out
    assert "0.90" in out


def test_verdict_omits_the_raw_confidence_when_calibration_barely_moved_it():
    out = _text(verdict_panel(_view(confidence=0.72, calibrated_confidence=0.72)))

    assert "raw" not in out


def test_verdict_renders_an_abstention_as_a_non_answer_with_its_reason():
    out = _text(
        verdict_panel(
            _view(
                abstained=True,
                abstain_reason="below_confidence_floor",
                root_cause_step_index=None,
                failure_class=None,
            )
        )
    )

    assert "ABSTAINED" in out
    assert "below_confidence_floor" in out


def test_verdict_shows_the_counterfactual():
    """`counterfactual` is what should have happened instead, and the old
    one-line output never printed it at all."""
    out = _text(verdict_panel(_view()))

    assert "Retry the lookup" in out


def test_verdict_names_degraded_layers_rather_than_presenting_a_thin_answer_as_full():
    out = _text(verdict_panel(_view(degraded_layers=["l2"])))

    assert "degraded" in out.lower()
    assert "l2" in out


def test_verdict_marks_the_root_cause_step_within_its_neighbours():
    context = [
        {"step_index": 2, "kind": "llm", "actor": "planner", "summary": "pick a tool", "duration_ms": 12.0},
        {"step_index": 3, "kind": "tool", "actor": "search", "summary": "returned nothing", "duration_ms": 9.0},
        {"step_index": 4, "kind": "llm", "actor": "planner", "summary": "wrote the answer", "duration_ms": 30.0},
    ]
    out = _text(verdict_panel(_view(), context))

    assert "returned nothing" in out
    assert "pick a tool" in out
    assert "->" in out


def test_verdict_renders_without_context_when_the_steps_cannot_be_read():
    out = _text(verdict_panel(_view(), None))

    assert "step 3" in out


def test_candidates_table_marks_the_winner_and_names_each_source():
    adjudications = [
        {"step_index": 3, "source": "l1", "failure_class": "silent_empty_result_misread",
         "confidence": 0.90, "is_root_cause": True},
        {"step_index": 7, "source": "filler", "failure_class": None,
         "confidence": 0.02, "is_root_cause": False},
    ]
    out = _text(candidates_table(_view(adjudications=adjudications)))

    assert "named" in out
    assert "l1" in out
    # Filler candidates are evidence-free padding; showing them is how the
    # shortlist's real evidence density stays visible.
    assert "filler" in out


def test_candidates_table_is_absent_when_nothing_was_adjudicated():
    assert candidates_table(_view(adjudications=[])) is None


def test_signals_table_caps_the_default_view_and_says_how_many_it_hid():
    signals = [
        {"step_index": i, "detector": f"det_{i}", "severity": 0.5, "message": f"m{i}"}
        for i in range(9)
    ]
    out = _text(signals_table(_view(signals=signals), limit=5))

    assert "det_0" in out
    assert "4 more" in out


def test_signals_table_shows_everything_when_uncapped():
    signals = [
        {"step_index": i, "detector": f"det_{i}", "severity": 0.5, "message": f"m{i}"}
        for i in range(9)
    ]
    out = _text(signals_table(_view(signals=signals), limit=None))

    assert "det_8" in out
    assert "more" not in out


def test_signals_table_is_absent_when_no_detector_fired():
    assert signals_table(_view(signals=[])) is None


def test_render_diagnosis_prints_verdict_candidates_and_signals_together(capsys):
    view = _view(
        adjudications=[
            {"step_index": 3, "source": "l1", "failure_class": "x",
             "confidence": 0.9, "is_root_cause": True}
        ],
        signals=[{"step_index": 3, "detector": "empty_tool_result", "severity": 0.6, "message": "empty"}],
    )
    render_diagnosis(view)
    out = capsys.readouterr().out

    assert "step 3" in out
    assert "empty_tool_result" in out
