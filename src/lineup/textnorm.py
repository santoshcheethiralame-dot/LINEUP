from __future__ import annotations

import re
import string

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop punctuation and articles, fold spaces."""
    text = text.lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def contains_phrase(text: str, phrase: str) -> bool:
    """True if the normalized `phrase` appears as a contiguous run of whole tokens in
    `text`. Whole-token matching keeps a short value (e.g. "5") from being found inside
    a longer one ("1885")."""
    needle = normalize(phrase).split()
    haystack = normalize(text).split()
    if not needle:
        return False
    return any(
        haystack[i : i + len(needle)] == needle
        for i in range(len(haystack) - len(needle) + 1)
    )
