from __future__ import annotations

import re
from random import Random
from typing import Iterable

from ..textnorm import contains_phrase, normalize
from .schema import QAExample

_YEAR = re.compile(r"^(1\d{3}|20\d{2})$")
_NUMBER = re.compile(r"^-?\d[\d,]*(\.\d+)?$")
_YEAR_OFFSETS = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]
_NUMBER_FACTORS = [0.5, 0.75, 1.25, 1.5, 2.0]


def classify_answer(answer: str) -> str:
    value = answer.strip()
    if _YEAR.match(value):
        return "year"
    if _NUMBER.match(value):
        return "number"
    return "entity"


def build_answer_pool(examples: Iterable[QAExample]) -> dict:
    """Collect the distinct gold answers seen across the corpus, grouped by type, so
    that a wrong value can be drawn from real same-type answers rather than invented."""
    pool: dict = {"year": [], "number": [], "entity": []}
    seen = set()
    for example in examples:
        value = example.answer.strip()
        key = value.lower()
        if not value or key in seen:
            continue
        seen.add(key)
        pool[classify_answer(value)].append(value)
    return pool


def _shape(value: str) -> tuple:
    # Per-token capitalization pattern: keeps a proper name from being swapped for a
    # same-length lowercase phrase, and vice versa.
    return tuple(token[:1].isupper() for token in value.split())


def _present(candidate: str, context: str) -> bool:
    # Whole-token containment, matching how the oracle and correctness checks test for a
    # value, so the builder's "already present" notion agrees with the evaluation's.
    return bool(normalize(candidate)) and contains_phrase(context, candidate)


def _sample(answer: str, pool: dict, key: str, rng: Random, context: str) -> str | None:
    candidates = [
        c for c in pool.get(key, [])
        if c.lower() != answer.lower()
        and not contains_phrase(c, answer) and not contains_phrase(answer, c)
        and normalize(c) and not _present(c, context)
    ]
    if not candidates:
        return None
    same_shape = [c for c in candidates if _shape(c) == _shape(answer)]
    return rng.choice(sorted(same_shape or candidates))


def perturb_value(
    answer: str, answer_type: str, pool: dict, rng: Random, *, context: str = ""
) -> str | None:
    """Produce a believable wrong value of the same type, never one that already appears
    in `context` (the question and source paragraph)."""
    value = answer.strip()

    if answer_type == "year":
        base = int(value)
        offsets = list(_YEAR_OFFSETS)
        rng.shuffle(offsets)
        for offset in offsets:
            candidate = str(base + offset)
            if candidate != value and not _present(candidate, context):
                return candidate
        return None

    if answer_type == "number":
        try:
            base = float(value.replace(",", ""))
        except ValueError:
            base = None
        if base not in (None, 0.0):
            factors = list(_NUMBER_FACTORS)
            rng.shuffle(factors)
            for factor in factors:
                scaled = base * factor
                candidate = str(int(round(scaled))) if "." not in value else f"{scaled:.2f}"
                if candidate.lower() != value.lower() and not _present(candidate, context):
                    return candidate
            return None
        return _sample(value, pool, "number", rng, context)

    return _sample(value, pool, "entity", rng, context)
