from __future__ import annotations

import re
from collections import OrderedDict
from typing import Iterable, Sequence

from rank_bm25 import BM25Okapi

from .schema import Chunk, QAExample

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def build_corpus(examples: Iterable[QAExample]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for example in examples:
        chunks.extend(example.gold_chunks)
        chunks.extend(example.distractor_pool)
    return chunks


class BM25DistractorRetriever:
    """Sparse retriever over the paragraph corpus, used to pull realistic distractors."""

    def __init__(self, corpus: Sequence[Chunk]):
        unique = OrderedDict()
        for chunk in corpus:
            unique.setdefault(chunk.title, chunk)       # one entry per article title
        self.chunks = list(unique.values())
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in self.chunks])

    def retrieve(
        self, question: str, *, k: int = 10, exclude_titles: Iterable[str] = ()
    ) -> list[Chunk]:
        excluded = set(exclude_titles)
        scores = self._bm25.get_scores(tokenize(question))
        order = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)

        results: list[Chunk] = []
        for i in order:
            source = self.chunks[i]
            if source.title in excluded:
                continue
            results.append(
                Chunk(
                    chunk_id=source.chunk_id,
                    title=source.title,
                    text=source.text,
                    sentences=list(source.sentences),
                    provenance="distractor",
                    retrieval_score=float(scores[i]),
                )
            )
            if len(results) >= k:
                break
        return results
