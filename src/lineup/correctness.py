from __future__ import annotations

import re

from .backends.base import Message
from .textnorm import contains_phrase, normalize

_POSITIVE = {"yes", "correct", "true"}
_WORD = re.compile(r"[a-z]+")


def normalized_exact_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def matches_intended_wrong(prediction: str, intended_wrong: str) -> bool:
    """Did the model echo the value the misleading chunk was built to induce? A diagnostic
    signal that the misleading chunk likely drove the error, not a correctness label."""
    return contains_phrase(prediction, intended_wrong)


def judge_correct(question: str, gold: str, prediction: str, llm_judge=None):
    """Decide correctness: an empty answer is wrong; otherwise normalized exact match
    first, then defer the phrasing variants it misses to the LLM judge when one is
    supplied. Returns (is_correct, tier)."""
    if not prediction.strip():
        return False, "exact"
    if normalized_exact_match(prediction, gold):
        return True, "exact"
    if llm_judge is not None:
        return llm_judge.is_correct(question, gold, prediction), "judge"
    return False, "exact"


class LLMJudge:
    """Decide correctness for phrasing variants that exact match misses (e.g. the model
    answers "President Roosevelt" where the reference is "Roosevelt")."""

    def __init__(self, model):
        self.model = model

    def is_correct(self, question: str, gold: str, prediction: str) -> bool:
        messages = [
            Message(
                "system",
                "You are grading a question-answering system. Compare the model answer to the "
                "reference answer and reply with a single word: yes if the model answer is "
                "correct, no otherwise.",
            ),
            Message(
                "user",
                f"Question: {question}\nReference answer: {gold}\nModel answer: {prediction}\n"
                "Is the model answer correct?",
            ),
        ]
        reply = self.model.generate(messages, max_new_tokens=4).text
        words = _WORD.findall(reply.lower())
        return bool(words) and words[0] in _POSITIVE
