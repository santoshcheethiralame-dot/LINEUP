from __future__ import annotations

from typing import Iterator

from .multihop import parse_example
from .schema import QAExample

DEFAULT_REPO = "framolfese/2WikiMultihopQA"


def load_examples(
    split: str = "validation", *, limit: int | None = None, repo: str = DEFAULT_REPO
) -> Iterator[QAExample]:
    from datasets import load_dataset

    # 2WikiMultiHopQA shares HotpotQA's schema; this parquet-native mirror needs no loading
    # script. Pass a different `repo` if the mirror moves.
    dataset = load_dataset(repo, split=split)
    for index, raw in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield parse_example(raw, source="2wiki")
