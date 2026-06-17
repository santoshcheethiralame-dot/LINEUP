from __future__ import annotations

from typing import Iterator

from .schema import Chunk, QAExample


def _columns(value, *keys):
    """Read either a dict of parallel lists or a list of dicts as parallel lists, so the
    parser works whether a dataset row uses the columnar or the record layout."""
    if isinstance(value, dict):
        return tuple(value[key] for key in keys)
    return tuple([row[key] for row in value] for key in keys)


def _supporting_map(supporting_facts) -> dict:
    titles, sent_ids = _columns(supporting_facts, "title", "sent_id")
    by_title: dict = {}
    for title, sent_id in zip(titles, sent_ids):
        by_title.setdefault(title, set()).add(int(sent_id))
    return by_title


def parse_example(raw: dict) -> QAExample:
    titles, sentences = _columns(raw["context"], "title", "sentences")
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
        meta={"type": raw.get("type"), "level": raw.get("level"), "source": "hotpotqa"},
    )


def load_examples(split: str = "validation", *, limit: int | None = None) -> Iterator[QAExample]:
    from datasets import load_dataset

    # HotpotQA now lives at the namespaced, parquet-native repo: the bare "hotpot_qa" id is
    # rejected by current huggingface_hub, and parquet needs no loading script.
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    for index, raw in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        yield parse_example(raw)
