import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from lineup.backends.base import Message
from lineup.backends.transformers_backend import TransformersModel

MODEL = "sshleifer/tiny-gpt2"


@pytest.mark.slow
def test_generate_is_greedy_deterministic():
    model = TransformersModel(MODEL, device="cpu", dtype="float32", max_new_tokens=8)
    messages = [Message("user", "Hello")]
    first = model.generate(messages)
    second = model.generate(messages)
    assert first.token_ids == second.token_ids


@pytest.mark.slow
def test_score_is_deterministic_and_finite():
    model = TransformersModel(MODEL, device="cpu", dtype="float32")
    messages = [Message("user", "The capital of France is")]
    first = model.score(messages, " Paris.")
    second = model.score(messages, " Paris.")
    assert first.logprobs == second.logprobs
    assert len(first.logprobs) == len(first.token_ids)
    assert all(math.isfinite(value) for value in first.logprobs)
