# llm.py - deliberately thin. No retries, no logging: all log calls sit at the
# orchestration layer where context is already assembled, and per-item error
# isolation happens one layer up (a raised call becomes a sentinel result
# there, not here). Mirrors Litmus/src/litmus/llm.py.
#
# `import litellm` then `litellm.completion(...)` is mandatory, not a style
# choice: every test in this project mocks via
# `monkeypatch.setattr("litellm.completion", ...)`, which only patches the
# dotted attribute on the `litellm` module. `from litellm import completion`
# would bind a local name before the patch exists and silently break every
# downstream agent's mocking strategy.
import time
from typing import Callable

import litellm

CallFn = Callable[[str, str], tuple[str, float, float, int | None, int | None]]
# (model, prompt) -> (raw_output, latency_ms, cost_usd, prompt_tokens, completion_tokens)


def _int_or_none(value) -> int | None:
    """Usage objects may omit token fields or be mocked; coerce only real ints."""
    return value if isinstance(value, int) else None


def litellm_call(model: str, prompt: str) -> tuple[str, float, float, int | None, int | None]:
    start = time.perf_counter()
    response = litellm.completion(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    latency_ms = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content
    cost_usd = litellm.completion_cost(completion_response=response)
    usage = getattr(response, "usage", None)
    prompt_tokens = _int_or_none(getattr(usage, "prompt_tokens", None)) if usage else None
    completion_tokens = _int_or_none(getattr(usage, "completion_tokens", None)) if usage else None
    return output, latency_ms, cost_usd, prompt_tokens, completion_tokens
