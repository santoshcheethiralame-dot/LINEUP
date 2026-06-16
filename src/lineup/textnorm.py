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
