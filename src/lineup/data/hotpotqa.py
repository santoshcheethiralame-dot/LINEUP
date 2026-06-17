from __future__ import annotations

from typing import Iterator

from .multihop import parse_example
from .schema import QAExample

__all__ = ["parse_example", "load_examples"]


def load_examples(split: str = "validation", *, limit: int | None = None) -> Iterator[QAExample]:
    from datasets import load_dataset

    # HotpotQA now lives at the namespaced, parquet-native repo: the bare "hotpot_qa" id is
    # rejected by current huggingface_hub, and parquet needs no loading script.
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    for index, raw in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield parse_example(raw, source="hotpotqa")
