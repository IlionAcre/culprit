"""DetectorContext / build_context sanity: the shared state every detector
reads, built once per run_detectors() call."""

from culprit.detectors.base import build_context, tokens, truncate
from culprit.schemas import SpanKind
from culprit.synth import successful_run


def _spans_by_id(run):
    return {s.span_id: s for s in run.spans}


def test_build_context_tool_universe_contains_every_executed_tool():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    executed = {
        ctx.spans_by_id[s.span_id].payload.tool_name
        for s in run.steps
        if s.kind == SpanKind.TOOL
    }
    assert executed <= ctx.tool_universe
    assert executed  # sanity: this run actually has tool steps


def test_build_context_goal_terms_are_lowercased_tokens_of_the_task_goal():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    assert "refund" in ctx.goal_terms
    assert all(t == t.lower() for t in ctx.goal_terms)


def test_build_context_signature_runs_reconstruct_the_full_step_sequence():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    total = sum(length for _sig, _start, length in ctx.signature_runs)
    assert total == len(run.steps)


def test_build_context_args_and_result_hash_are_none_for_non_tool_steps():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    for step in run.steps:
        if step.kind != SpanKind.TOOL:
            assert ctx.args_hash_by_step[step.step_index] is None
            assert ctx.result_hash_by_step[step.step_index] is None
        else:
            assert ctx.args_hash_by_step[step.step_index] is not None
            assert ctx.result_hash_by_step[step.step_index] is not None


def test_build_context_llm_steps_by_actor_only_contains_llm_steps():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    for actor, indices in ctx.llm_steps_by_actor.items():
        for i in indices:
            assert run.steps[i].kind == SpanKind.LLM
            assert run.steps[i].actor == actor
        assert indices == sorted(indices)


def test_had_provenance_before_finds_a_goal_token_but_not_a_fabricated_one():
    run = successful_run(seed=1)
    ctx = build_context(run.trace, run.steps, _spans_by_id(run))
    order_token = next(iter(ctx.goal_terms & {t for t in ctx.goal_terms if t.startswith("ord")}), None)
    assert order_token is not None
    assert ctx.had_provenance_before(order_token, step_index=len(run.steps))
    assert not ctx.had_provenance_before("totally-fabricated-token-xyz", step_index=len(run.steps))


def test_tokens_helper_lowercases_and_splits_on_non_alnum():
    assert tokens("Refund ORDER-42133!") == {"refund", "order", "42133"}
    assert tokens(None) == frozenset()


def test_truncate_keeps_head_and_tail_under_the_limit():
    text = "x" * 1000
    out = truncate(text, limit=100)
    assert len(out) <= 110  # small allowance for the " ... " separator
    assert out.startswith("x")
    assert out.endswith("x")


def test_truncate_leaves_short_text_untouched():
    assert truncate("short") == "short"
