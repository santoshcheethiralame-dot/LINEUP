from __future__ import annotations

import re
from random import Random
from typing import Iterable

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
    tokens = value.split()
    return (len(tokens), value[:1].isupper())


def _sample(answer: str, pool: dict, key: str, rng: Random) -> str | None:
    lowered = answer.lower()
    candidates = [
        c for c in pool.get(key, [])
        if c.lower() != lowered and c.lower() not in lowered and lowered not in c.lower()
    ]
    if not candidates:
        return None
    same_shape = [c for c in candidates if _shape(c) == _shape(answer)]
    return rng.choice(sorted(same_shape or candidates))


def perturb_value(answer: str, answer_type: str, pool: dict, rng: Random) -> str | None:
    value = answer.strip()
    if answer_type == "year":
        return str(int(value) + rng.choice(_YEAR_OFFSETS))
    if answer_type == "number":
        try:
            base = float(value.replace(",", ""))
        except ValueError:
            base = None
        if base not in (None, 0.0):
            scaled = base * rng.choice(_NUMBER_FACTORS)
            return str(int(round(scaled))) if "." not in value else f"{scaled:.2f}"
        return _sample(value, pool, "number", rng)
    return _sample(value, pool, "entity", rng)
