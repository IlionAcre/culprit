import json

from culprit.adjudicate import adjudicate
from culprit.schemas import Outcome, SpanKind, Step, Trace
from culprit.signals import Candidate


def _trace(outcome: Outcome = Outcome.FAILURE) -> Trace:
    return Trace(
        trace_id="t1", source="test", outcome=outcome, span_count=0, step_count=0,
        task_goal="refund order 88213",
    )


def _step(index: int) -> Step:
    return Step(
        trace_id="t1", step_index=index, span_id=f"span-{index}", kind=SpanKind.TOOL,
        actor="agent", depth=0, tree_path="0", signature=f"tool:agent:tool{index}:ok",
        summary="does something", start_ns=0, end_ns=1_000_000, duration_ms=1.0,
    )


def _candidate(step_index: int, *, source: str = "l1") -> Candidate:
    return Candidate(
        step_index=step_index, span_id=f"span-{step_index}", rank=1, prior=0.6,
        source=source, signals=[], divergence=None,
    )


def _verdict_json(step_index: int, **overrides) -> str:
    body = {
        "is_root_cause": True,
        "failure_class": "tool_failure_unhandled",
        "confidence": 0.8,
        "rationale": f"step {step_index} looks like the cause",
        "counterfactual": "the agent should have retried",
        "cited_step_indices": [step_index],
    }
    body.update(overrides)
    return json.dumps(body)


def test_adjudicate_returns_an_empty_list_for_no_candidates():
    assert adjudicate([], _trace(), [], call_fn=lambda m, p: ("{}", 0.0, 0.0, 0, 0), model="m") == []


def _judged_step(prompt: str) -> int:
    """The prompt names its own candidate step via a distinctive phrase
    ("whether step N is where"); a plain "step N" substring search would
    also match the zoom window's neighboring step headers ("--- step N
    ...") and misidentify which candidate is actually being judged."""
    marker = "whether step "
    start = prompt.index(marker) + len(marker)
    end = prompt.index(" is where", start)
    return int(prompt[start:end])


def test_adjudicate_calls_call_fn_once_per_candidate_and_preserves_input_order():
    steps = [_step(i) for i in range(5)]
    candidates = [_candidate(1), _candidate(3)]
    calls = []

    def call_fn(model, prompt):
        step_in_prompt = _judged_step(prompt)
        calls.append(step_in_prompt)
        return _verdict_json(step_in_prompt), 5.0, 0.001, 20, 8

    result = adjudicate(candidates, _trace(), steps, call_fn=call_fn, model="m")

    assert sorted(calls) == [1, 3]
    assert [a.step_index for a in result] == [1, 3]


def test_adjudicate_produces_a_valid_adjudication_from_a_clean_response():
    steps = [_step(i) for i in range(3)]
    candidates = [_candidate(1)]

    def call_fn(model, prompt):
        return _verdict_json(1), 5.0, 0.002, 12, 4

    result = adjudicate(candidates, _trace(), steps, call_fn=call_fn, model="gpt-test")

    a = result[0]
    assert a.is_root_cause is True
    assert a.failure_class == "tool_failure_unhandled"
    assert a.confidence == 0.8
    assert 0.0 <= a.calibrated_confidence <= 1.0
    assert a.abstained is False
    assert a.error is None
    assert a.model == "gpt-test"
    assert a.cost_usd == 0.002
    assert a.prompt_tokens == 12
    assert a.completion_tokens == 4


# --- de-fencing: the four cases proven against real model output in
# Litmus's test_scoring_llm_judge.py, reused verbatim here since adjudicate.py
# copies the same regex deliberately (see its module docstring).

def test_adjudicate_strips_markdown_code_fence_with_language_tag():
    steps = [_step(i) for i in range(3)]
    raw = f"```json\n{_verdict_json(1)}\n```"

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=lambda m, p: (raw, 1.0, 0.0, 10, 5), model="m")

    assert result[0].error is None
    assert result[0].is_root_cause is True


def test_adjudicate_strips_bare_code_fence_without_language_tag():
    steps = [_step(i) for i in range(3)]
    raw = f"```\n{_verdict_json(1)}\n```"

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=lambda m, p: (raw, 1.0, 0.0, 10, 5), model="m")

    assert result[0].error is None


def test_adjudicate_strips_single_line_code_fence_with_no_internal_newline():
    steps = [_step(i) for i in range(3)]
    raw = f"```{_verdict_json(1)}```"

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=lambda m, p: (raw, 1.0, 0.0, 10, 5), model="m")

    assert result[0].error is None


def test_adjudicate_strips_code_fence_with_prose_before_it():
    steps = [_step(i) for i in range(3)]
    raw = f"Here is my answer:\n```json\n{_verdict_json(1)}\n```"

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=lambda m, p: (raw, 1.0, 0.0, 10, 5), model="m")

    assert result[0].error is None


# --- per-item error isolation ---------------------------------------------

def test_adjudicate_produces_a_sentinel_abstained_adjudication_on_an_unparseable_response():
    steps = [_step(i) for i in range(3)]
    raw = "I think this passes, looks good to me!"

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=lambda m, p: (raw, 1.0, 0.0, 10, 5), model="m")

    a = result[0]
    assert a.abstained is True
    assert a.error is not None
    assert a.is_root_cause is False
    assert a.calibrated_confidence == 0.0


def test_adjudicate_produces_a_sentinel_abstained_adjudication_when_call_fn_raises():
    steps = [_step(i) for i in range(3)]

    def call_fn(model, prompt):
        raise RuntimeError("rate limited")

    result = adjudicate([_candidate(1)], _trace(), steps, call_fn=call_fn, model="m")

    a = result[0]
    assert a.abstained is True
    assert "rate limited" in a.error


def test_adjudicate_one_failing_candidate_does_not_cost_the_others():
    """Per-item error isolation: a candidate's own call_fn failure must not
    prevent the other four from being judged."""
    steps = [_step(i) for i in range(5)]
    candidates = [_candidate(1), _candidate(2), _candidate(3)]

    def call_fn(model, prompt):
        step_in_prompt = _judged_step(prompt)
        if step_in_prompt == 2:
            raise RuntimeError("boom")
        return _verdict_json(step_in_prompt), 1.0, 0.0, 10, 5

    result = adjudicate(candidates, _trace(), steps, call_fn=call_fn, model="m")

    by_step = {a.step_index: a for a in result}
    assert len(result) == 3
    assert by_step[2].abstained is True
    assert by_step[2].error is not None
    assert by_step[1].abstained is False
    assert by_step[3].abstained is False


# --- citation grounding must measurably move calibrated confidence --------

def test_out_of_window_citation_measurably_lowers_calibrated_confidence():
    """A rationale citing a step far outside anything shown is direct
    evidence of confabulation and must reduce calibrated_confidence relative
    to an identical response that cites a step actually in the packet."""
    steps = [_step(i) for i in range(5)]
    candidate = _candidate(2)

    grounded = adjudicate(
        [candidate], _trace(), steps,
        call_fn=lambda m, p: (_verdict_json(2, cited_step_indices=[2]), 1.0, 0.0, 10, 5),
        model="m",
    )[0]
    confabulated = adjudicate(
        [candidate], _trace(), steps,
        call_fn=lambda m, p: (_verdict_json(2, cited_step_indices=[9999]), 1.0, 0.0, 10, 5),
        model="m",
    )[0]

    assert confabulated.calibrated_confidence < grounded.calibrated_confidence


def test_adjudicate_max_workers_one_runs_sequentially_with_the_same_result():
    steps = [_step(i) for i in range(3)]
    candidates = [_candidate(1)]

    result = adjudicate(
        candidates, _trace(), steps,
        call_fn=lambda m, p: (_verdict_json(1), 1.0, 0.0, 10, 5), model="m", max_workers=1,
    )

    assert result[0].is_root_cause is True


def test_adjudicate_preserves_candidate_source_on_the_adjudication():
    """`source` is a stored fact (Task G), not reconstructed; it must be
    carried unchanged from Candidate to Adjudication for every source kind."""
    steps = [_step(i) for i in range(3)]

    for source in ("l1", "l2", "both", "fallback", "filler"):
        result = adjudicate(
            [_candidate(1, source=source)], _trace(), steps,
            call_fn=lambda m, p: (_verdict_json(1), 1.0, 0.0, 10, 5), model="m",
        )
        assert result[0].source == source


def test_adjudicate_does_not_change_calibrated_confidence_for_filler_when_a10_is_zero():
    """`is_filler` defaults to coefficient 0.0, so a filler candidate with
    otherwise identical features must calibrate to the same value as a real
    candidate."""
    steps = [_step(i) for i in range(3)]

    real = adjudicate(
        [_candidate(1, source="l1")], _trace(), steps,
        call_fn=lambda m, p: (_verdict_json(1), 1.0, 0.0, 10, 5), model="m",
    )[0]
    filler = adjudicate(
        [_candidate(1, source="filler")], _trace(), steps,
        call_fn=lambda m, p: (_verdict_json(1), 1.0, 0.0, 10, 5), model="m",
    )[0]

    assert filler.calibrated_confidence == real.calibrated_confidence


def test_adjudicate_preserves_source_on_a_sentinel_abstained_adjudication():
    """Per-item error isolation must not drop the source fact when a candidate
    call fails."""
    steps = [_step(i) for i in range(3)]

    def call_fn(model, prompt):
        raise RuntimeError("boom")

    result = adjudicate(
        [_candidate(1, source="filler")], _trace(), steps, call_fn=call_fn, model="m",
    )

    assert result[0].abstained is True
    assert result[0].source == "filler"
