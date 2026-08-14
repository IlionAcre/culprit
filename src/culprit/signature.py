"""The alignment alphabet (coarse signature token) and the fine-grained
similarity function `align.py` uses for substitution scoring.

**Load-bearing property, inherited from `linearize.py` and restated here
because getting it wrong collapses the whole layer**: the coarse token for
an LLM step encodes the decision the step made (`llm:{actor}:call:{tool}` /
`:plan` / `:answer`), never the model name. Signed by model name instead,
every LLM step becomes an identical token and alignment degenerates to
"how many steps are LLM calls," which carries no positional information at
all.
"""

from culprit.schemas import Step

# Feature order and weights from the plan's L2 spec, `sim(a, b)` in [0, 1]:
# [kind_match, actor_match, tool_match, outcome_match, arg_key_jaccard,
#  arg_value_jaccard, 1-|depth_delta|/4, retrieval_docid_overlap]
_WEIGHTS = (0.25, 0.15, 0.25, 0.15, 0.08, 0.05, 0.04, 0.03)

# `Step` carries no payload (arguments, retrieved doc ids, response text) -
# only the structural/coarse fields `linearize.py` computed. The three
# payload-dependent features (arg_key_jaccard, arg_value_jaccard,
# retrieval_docid_overlap) are therefore unreachable from what `contrast()`
# is actually given (`steps: list[Step]`, never `spans: list[Span]`). Rather
# than silently scoring them 0 (which would penalize every pair identically
# and quietly bias substitution scores low) or crashing, they degrade to a
# fixed neutral 0.5: a constant offset shared by every comparison, so it
# cannot invert a ranking between two candidate alignments, only compress
# the score range this layer never claimed to need for its interpretable
# columns anyway. Flagged in the WS-D handoff report as a real gap between
# the spec's fine feature vector and what the frozen `contrast()` signature
# actually provides.
_NEUTRAL = 0.5


def signature_of(step: Step) -> str:
    """The coarse alignment token for one step. `Step.signature` already
    carries this exact value - `linearize.py` computes it at ingestion time
    using the identical `kind:actor:...` convention this module documents,
    and `Step` has no payload to recompute it from even if that seemed
    worth doing. This exists as a named function, rather than every caller
    reading `step.signature` directly, so the alignment alphabet has one
    obvious call site to point at."""
    return step.signature


def _tool_token(signature: str) -> str:
    parts = signature.split(":")
    if parts[0] == "llm" and len(parts) >= 4 and parts[2] == "call":
        return parts[3]
    if parts[0] == "tool" and len(parts) >= 3:
        return parts[2]
    return ""


def _outcome_token(signature: str) -> str:
    """The trailing ok/err/empty/hit/miss suffix, "" for tokens that carry
    none (`agent:...`, `llm:{actor}:plan`, `llm:{actor}:answer`)."""
    parts = signature.split(":")
    return parts[-1] if len(parts) >= 3 else ""


def sim(a: Step, b: Step) -> float:
    """Weighted 8-feature similarity in [0, 1]. `align.py`'s substitution
    score is `2 * sim(a, b) - 1`, so identical steps score close to +1 and
    thoroughly unrelated steps score close to -1."""
    sig_a, sig_b = a.signature, b.signature
    tool_a, tool_b = _tool_token(sig_a), _tool_token(sig_b)
    out_a, out_b = _outcome_token(sig_a), _outcome_token(sig_b)

    features = (
        1.0 if a.kind == b.kind else 0.0,
        1.0 if a.actor == b.actor else 0.0,
        1.0 if tool_a == tool_b else 0.0,
        1.0 if out_a and out_a == out_b else 0.0,
        _NEUTRAL,  # arg_key_jaccard - unreachable, see module docstring
        _NEUTRAL,  # arg_value_jaccard - unreachable
        1.0 - min(abs(a.depth - b.depth), 4) / 4,
        _NEUTRAL,  # retrieval_docid_overlap - unreachable
    )
    return sum(w * f for w, f in zip(_WEIGHTS, features))
