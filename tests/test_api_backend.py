from types import SimpleNamespace

import pytest

from lineup.backends.api_backend import APIModel
from lineup.backends.base import Message


class _FakeCompletions:
    def __init__(self, text):
        self._text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))])


class _FakeClient:
    def __init__(self, text):
        self.chat = SimpleNamespace(completions=_FakeCompletions(text))


def test_generate_parses_and_strips_chat_response():
    model = APIModel("x", client=_FakeClient("  Paris  "))
    generation = model.generate([Message("user", "capital of France?")], max_new_tokens=8)
    assert generation.text == "Paris"
    assert generation.token_logprobs == []                      # no logprobs from a chat API
    assert model.client.chat.completions.calls[0]["max_tokens"] == 8


def test_score_raises_so_logprob_methods_cannot_run_silently():
    model = APIModel("x", client=_FakeClient("y"))
    with pytest.raises(NotImplementedError):
        model.score([Message("user", "q")], "answer")
