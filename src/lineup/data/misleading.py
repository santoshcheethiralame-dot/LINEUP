from __future__ import annotations

import re
from abc import ABC, abstractmethod
from random import Random

from .schema import Chunk, QAExample
from .substitution import classify_answer, perturb_value

_BOOLEAN = {"yes", "no"}


def _answer_pattern(answer: str) -> re.Pattern:
    # Anchor on a word boundary only where the answer edge is alphanumeric, so that
    # answers containing punctuation ("U.S.") still match, while a short answer is not
    # matched inside a longer word ("Mann" within "Manning").
    left = r"\b" if answer[:1].isalnum() else ""
    right = r"\b" if answer[-1:].isalnum() else ""
    return re.compile(left + re.escape(answer) + right, re.IGNORECASE)


def find_answer_location(example: QAExample):
    """Locate the gold sentence that states the answer, preferring annotated
    supporting sentences. Returns (chunk, sentence_id) or None."""
    pattern = _answer_pattern(example.answer.strip())
    for chunk in example.gold_chunks:
        for sentence_id in chunk.supporting_sentence_ids:
            if 0 <= sentence_id < len(chunk.sentences) and pattern.search(chunk.sentences[sentence_id]):
                return chunk, sentence_id
    for chunk in example.gold_chunks:
        for sentence_id, sentence in enumerate(chunk.sentences):
            if pattern.search(sentence):
                return chunk, sentence_id
    return None


def substitution_check(example: QAExample) -> str | None:
    """Return a reason the example cannot carry a constructed near-miss, or None."""
    answer = example.answer.strip()
    if not answer:
        return "empty answer"
    if answer.lower() in _BOOLEAN:
        return "boolean answer"
    if find_answer_location(example) is None:
        return "answer not present in gold sentences"
    return None


def _substitute(answer: str, replacement: str, sentences: list[str]) -> list[str]:
    pattern = _answer_pattern(answer)
    return [pattern.sub(replacement, sentence) for sentence in sentences]


class MisleadingChunkBuilder(ABC):
    @abstractmethod
    def build(self, example: QAExample, pool: dict, rng: Random, *, context: str | None = None):
        ...


class ValueSubstitutionBuilder(MisleadingChunkBuilder):
    """Construct a near-miss by cloning the gold paragraph that states the answer and
    swapping the answer for a plausible, same-type wrong value. The result is
    topically identical to the gold evidence and differs only in the value it asserts.
    """

    def build(self, example: QAExample, pool: dict, rng: Random, *, context: str | None = None):
        answer = example.answer.strip()
        if not answer:
            return None
        location = find_answer_location(example)
        if location is None:
            return None
        source_chunk, sentence_id = location

        answer_type = classify_answer(answer)
        # The caller may supply the full assembled context so the planted value avoids every
        # passage in the scenario; otherwise fall back to the question and source paragraph.
        if context is None:
            context = example.question + " " + " ".join(source_chunk.sentences)
        replacement = perturb_value(answer, answer_type, pool, rng, context=context)
        if not replacement or replacement.lower() == answer.lower():
            return None

        rewritten = _substitute(answer, replacement, source_chunk.sentences)
        if rewritten == source_chunk.sentences:
            return None

        chunk = Chunk(
            chunk_id=f"{example.qid}::misleading",
            title=source_chunk.title,
            text=" ".join(rewritten).strip(),
            sentences=rewritten,
            provenance="misleading",
        )
        recipe_bits = {
            "original_value": answer,
            "intended_wrong_answer": replacement,
            "substitution_type": answer_type,
            "source_gold_chunk_id": source_chunk.chunk_id,
            "source_sentence_id": sentence_id,
        }
        return chunk, recipe_bits
