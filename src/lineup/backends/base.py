from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: list[int]
    token_logprobs: list[float]   # logprob of each generated token under greedy decoding

    @property
    def total_logprob(self) -> float:
        return float(sum(self.token_logprobs))


@dataclass(frozen=True)
class Scoring:
    tokens: list[str]
    token_ids: list[int]
    logprobs: list[float]         # teacher-forced logprob of each response token

    @property
    def total_logprob(self) -> float:
        return float(sum(self.logprobs))

    @property
    def mean_logprob(self) -> float:
        return float(sum(self.logprobs) / len(self.logprobs)) if self.logprobs else 0.0


class LanguageModel(ABC):
    """A backend that can generate a response and score a fixed response token by token.

    The scoring path is what the attribution methods and the counterfactual oracle
    depend on, so every backend must expose teacher-forced token log-probabilities,
    not just generated text.
    """

    @abstractmethod
    def generate(self, messages: Sequence[Message], max_new_tokens: int | None = None) -> Generation:
        ...

    @abstractmethod
    def score(self, messages: Sequence[Message], response: str) -> Scoring:
        ...
