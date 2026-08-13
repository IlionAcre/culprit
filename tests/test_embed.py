import numpy as np

import culprit.embed as embed_module
from culprit.embed import embed_texts


class _FakeModel:
    """Stands in for fastembed.TextEmbedding: a deterministic vector shaped by
    input length, with a call counter to prove the model is loaded once.
    Yields numpy arrays, matching fastembed's real Iterable[NumpyArray]
    contract, since embed_texts relies on `.tolist()` being available."""

    instances_created = 0

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        _FakeModel.instances_created += 1

    def embed(self, texts):
        for text in texts:
            yield np.array([float(len(text))] * 4)


def _reset_model_cache(monkeypatch):
    monkeypatch.setattr(embed_module, "_model", None)
    _FakeModel.instances_created = 0


def test_embed_texts_returns_one_vector_per_input_without_touching_the_real_model(monkeypatch):
    """fastembed's real BAAI/bge-small-en-v1.5 model is ~130MB and must never
    be downloaded during a test run; monkeypatching TextEmbedding keeps this
    fully offline."""
    _reset_model_cache(monkeypatch)
    monkeypatch.setattr(embed_module, "TextEmbedding", _FakeModel)

    result = embed_texts(["hi", "hello!"])

    assert result == [[2.0, 2.0, 2.0, 2.0], [6.0, 6.0, 6.0, 6.0]]
    assert all(isinstance(value, float) for row in result for value in row)


def test_embed_texts_loads_the_model_lazily_and_reuses_one_instance(monkeypatch):
    """One model instance must serve both the L2 similarity lookup and L5
    clustering call sites; loading it twice would double the ~130MB resident
    cost for no benefit."""
    _reset_model_cache(monkeypatch)
    monkeypatch.setattr(embed_module, "TextEmbedding", _FakeModel)

    assert _FakeModel.instances_created == 0

    embed_texts(["first call"])
    embed_texts(["second call"])

    assert _FakeModel.instances_created == 1
