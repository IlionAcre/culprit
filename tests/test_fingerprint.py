from culprit.fingerprint import agent_key, task_key
from culprit.synth import successful_run


def test_task_key_collapses_different_order_ids_to_the_same_key():
    """CLAUDE.md's canonical example: "Refund order 88213" and "Refund
    order 90114" must hash identically."""
    assert task_key("Refund order 88213: customer says it never arrived") == task_key(
        "Refund order 90114: customer says it never arrived"
    )


def test_task_key_differs_for_genuinely_different_goals():
    assert task_key("Refund order 88213") != task_key("Cancel subscription 88213")


def test_task_key_is_stable_for_none_and_empty_goal():
    assert task_key(None) == task_key("")


def test_task_key_normalizes_uuids_dates_emails_and_urls():
    a = "Contact a@example.com about order abcdef12-3456-7890-abcd-ef1234567890 on 2026-08-11, see https://x.co/a"
    b = "Contact b@other.com about order 11111111-2222-3333-4444-555555555555 on 2026-01-01, see https://y.co/b"
    assert task_key(a) == task_key(b)


def test_agent_key_is_invariant_to_step_order_and_argument_values():
    """agent_key depends only on the (framework, actor set, tool universe)
    triple, never on execution order or argument values - shuffling a run's
    steps must not change it."""
    run = successful_run(seed=1)
    shuffled_steps = list(reversed(run.steps))

    assert agent_key(run.trace, run.steps) == agent_key(run.trace, shuffled_steps)


def test_agent_key_varies_when_synth_seeds_pick_different_optional_tools():
    """Real finding, not a bug in this module: `agent_key` is computed from
    the tools *observed* in one trace's own steps (the only signal `Step`
    carries - see this module's docstring), and `synth.py`'s seeds
    independently include or omit `check_duplicate_refund` and
    `verify_eligibility`. Two runs of the identical underlying agent can
    therefore land on different observed tool universes and hash to
    different agent_keys purely because of which optional path a given run
    happened to take. Flagged in the WS-D handoff report: a production
    `agent_key` would ideally be computed from the agent's *registered*
    tool set, not its per-run observed calls, but nothing in the frozen
    `Trace`/`Step` schema carries a registered tool list to use instead."""
    seeds_with_both_optional_blocks = []
    for seed in range(1, 30):
        run = successful_run(seed)
        sigs = {s.signature for s in run.steps}
        has_dup = any("check_duplicate_refund" in s for s in sigs)
        has_verify = any("verify_eligibility" in s for s in sigs)
        seeds_with_both_optional_blocks.append((seed, has_dup, has_verify))

    combos = {(has_dup, has_verify) for _, has_dup, has_verify in seeds_with_both_optional_blocks}
    assert len(combos) > 1, "expected synth seeds to exercise more than one optional-block combination"

    keys_by_combo = {}
    for seed, has_dup, has_verify in seeds_with_both_optional_blocks:
        run = successful_run(seed)
        keys_by_combo.setdefault((has_dup, has_verify), agent_key(run.trace, run.steps))

    assert len(set(keys_by_combo.values())) > 1


def test_agent_key_differs_when_the_tool_universe_differs():
    run = successful_run(seed=1)
    key_full = agent_key(run.trace, run.steps)

    trimmed_steps = [s for s in run.steps if "process_refund" not in s.signature]
    key_trimmed = agent_key(run.trace, trimmed_steps)

    assert key_full != key_trimmed
