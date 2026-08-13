"""fastembed passthrough for BAAI/bge-small-en-v1.5 (384-dim, ONNX, CPU-only,
no torch).

One lazily-loaded `TextEmbedding` instance serves both the L2 trace
similarity lookup and the L5 diagnosis clustering, because the model is
roughly 130MB resident and a second instance would double that for no
benefit. Loading happens on first call, not at import time, so
`import culprit.embed` does no network or disk work: the offline test suite
monkeypatches `TextEmbedding` before `embed_texts` ever touches the real
model, and `uv run pytest` never downloads it.
"""

from typing import Callable

from fastembed import TextEmbedding

EmbedFn = Callable[[list[str]], list[list[float]]]

_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return [vector.tolist() for vector in model.embed(texts)]
