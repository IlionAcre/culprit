from unittest.mock import MagicMock

from culprit.llm import litellm_call


def _fake_response(content: str, prompt_tokens: int = 7, completion_tokens: int = 3) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def test_litellm_call_returns_output_latency_cost_and_token_counts(monkeypatch):
    """The dotted monkeypatch target is why llm.py must use `import litellm`
    and call `litellm.completion(...)` rather than importing the function."""
    monkeypatch.setattr(
        "litellm.completion", lambda model, messages: _fake_response("hello")
    )
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0001)

    output, latency_ms, cost_usd, prompt_tokens, completion_tokens = litellm_call(
        "gemini/gemini-2.5-flash-lite", "hi"
    )

    assert output == "hello"
    assert cost_usd == 0.0001
    assert latency_ms >= 0.0
    assert prompt_tokens == 7
    assert completion_tokens == 3


def test_litellm_call_passes_model_and_prompt_through_as_a_single_user_message(monkeypatch):
    """Guards the exact request shape: a downstream agent building on this
    seam needs to trust that prompt text is not silently reshaped or split."""
    captured = {}

    def _fake_completion(model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return _fake_response("ok")

    monkeypatch.setattr("litellm.completion", _fake_completion)
    monkeypatch.setattr("litellm.completion_cost", lambda completion_response: 0.0)

    litellm_call("gemini/gemini-2.5-flash-lite", "what happened at step 12?")

    assert captured["model"] == "gemini/gemini-2.5-flash-lite"
    assert captured["messages"] == [
        {"role": "user", "content": "what happened at step 12?"}
    ]
