from __future__ import annotations

from typing import Iterator

from .schema import QAExample

DATASETS = ("hotpotqa", "2wiki")


def load_examples(dataset: str, split: str = "validation", *, limit: int | None = None) -> Iterator[QAExample]:
    """Load multi-hop QA examples from a named source: 'hotpotqa' or '2wiki'. Both share the
    same schema and feed the rest of the pipeline identically, so a cross-dataset run is just
    a change of name."""
    if dataset == "hotpotqa":
        from .hotpotqa import load_examples as _load
    elif dataset == "2wiki":
        from .wiki2 import load_examples as _load
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    return _load(split, limit=limit)
