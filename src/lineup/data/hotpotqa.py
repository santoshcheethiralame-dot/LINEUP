from __future__ import annotations

from typing import Iterator

from .schema import Chunk, QAExample


def _supporting_map(supporting_facts: dict) -> dict:
    by_title: dict = {}
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        by_title.setdefault(title, set()).add(int(sent_id))
    return by_title


def parse_example(raw: dict) -> QAExample:
    titles = raw["context"]["title"]
    sentences = raw["context"]["sentences"]
    supporting = _supporting_map(raw["supporting_facts"])

    gold: list[Chunk] = []
    distractors: list[Chunk] = []
    for index, (title, sents) in enumerate(zip(titles, sentences)):
        supporting_ids = tuple(sorted(supporting.get(title, ())))
        chunk = Chunk(
            chunk_id=f"{raw['id']}::{index}",
            title=title,
            text=" ".join(sents).strip(),
            sentences=list(sents),
            provenance="gold" if supporting_ids else "distractor",
            supporting_sentence_ids=supporting_ids,
        )
        (gold if supporting_ids else distractors).append(chunk)

    return QAExample(
        qid=raw["id"],
        question=raw["question"].strip(),
        answer=raw["answer"].strip(),
        gold_chunks=gold,
        distractor_pool=distractors,
        meta={
            "type": raw.get("type"),
            "level": raw.get("level"),
            "source": "hotpotqa",
        },
    )


def load_examples(split: str = "validation", *, limit: int | None = None) -> Iterator[QAExample]:
    from datasets import load_dataset

    dataset = load_dataset("hotpot_qa", "distractor", split=split, trust_remote_code=True)
    for index, raw in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield parse_example(raw)
