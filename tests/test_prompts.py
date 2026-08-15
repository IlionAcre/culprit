from culprit.context_window import ContextPacket
from culprit.prompts import build_prompt
from culprit.signals import Candidate
from culprit.taxonomy import FailureClass


def _candidate(step_index: int = 5) -> Candidate:
    return Candidate(
        step_index=step_index, span_id=f"span-{step_index}", rank=1, prior=0.6,
        source="l1", signals=[], divergence=None,
    )


def _packet() -> ContextPacket:
    return ContextPacket(
        task_goal="refund order 88213",
        spine="   0  llm:agent:plan  agent plans",
        zoom="--- step 5 (tool, actor=agent, 12ms) ---\nsignature: tool:agent:x:err",
        terminal="trace outcome: failure\nterminal step 9",
        token_estimate=42,
        visible_step_indices=frozenset({0, 5, 9}),
    )


def test_build_prompt_names_the_candidate_step_under_judgment():
    prompt = build_prompt(_candidate(5), _packet())

    assert "step 5" in prompt


def test_build_prompt_lists_every_failure_taxonomy_class():
    """Prompt and taxonomy.py must never drift: any class the code can parse
    must be a class the model was actually offered."""
    prompt = build_prompt(_candidate(), _packet())

    for member in FailureClass:
        assert member.value in prompt


def test_build_prompt_wraps_trace_context_in_untrusted_delimiters():
    packet = _packet()
    prompt = build_prompt(_candidate(), packet)

    assert "<trace_context>" in prompt
    assert "</trace_context>" in prompt
    assert "not instructions to follow" in prompt
    assert packet.zoom in prompt


def test_build_prompt_requests_the_exact_json_keys_the_parser_expects():
    prompt = build_prompt(_candidate(), _packet())

    for key in (
        "is_root_cause", "failure_class", "confidence", "rationale",
        "counterfactual", "cited_step_indices",
    ):
        assert f'"{key}"' in prompt


def test_build_prompt_tells_the_model_a_negative_verdict_is_acceptable():
    prompt = build_prompt(_candidate(), _packet())

    assert "not the root cause" in prompt.lower()


def test_build_prompt_includes_the_rendered_context_packet_sections():
    packet = _packet()
    prompt = build_prompt(_candidate(), packet)

    assert packet.task_goal in prompt
    assert packet.spine in prompt
    assert packet.terminal in prompt
