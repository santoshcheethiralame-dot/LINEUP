from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    provenance: str = "distractor"             # "gold" or "distractor": where it came from
    supporting_sentence_ids: tuple[int, ...] = ()
    retrieval_score: float | None = None


@dataclass
class QAExample:
    qid: str
    question: str
    answer: str
    gold_chunks: list[Chunk]
    distractor_pool: list[Chunk]
    answer_aliases: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
