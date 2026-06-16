from __future__ import annotations

import re
import string

from .backends.base import Message

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles, fold spaces."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def normalized_exact_match(prediction: str, gold: str) -> bool:
    return normalize(prediction) == normalize(gold)


def matches_intended_wrong(prediction: str, intended_wrong: str) -> bool:
    """Did the model echo the value the misleading chunk was built to induce? A diagnostic
    signal that the misleading chunk likely drove the error, not a correctness label."""
    wrong = normalize(intended_wrong)
    return bool(wrong) and wrong in normalize(prediction)


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
        return reply.strip().lower().startswith("y")
