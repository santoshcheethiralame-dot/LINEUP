from __future__ import annotations

import re
from typing import Iterator

from .schema import Chunk, QAExample

# MuSiQue is multi-hop QA with gold supporting facts, like HotpotQA/2Wiki, but its rows mark support
# per paragraph (is_supporting) rather than via a separate supporting_facts list, and paragraphs are
# not pre-split into sentences. Hence its own parser. Set `repo` if the HF mirror moves.
DEFAULT_REPO = "dgslibisey/MuSiQue"
_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def parse_musique(raw: dict, source: str = "musique") -> QAExample:
    """Turn one MuSiQue row into a QAExample. Supporting paragraphs become gold chunks, the rest
    distractors; the answer's hop count is recorded as the question 'type' for the structure analysis."""
    qid = str(raw.get("id") or raw.get("_id") or "")
    gold: list[Chunk] = []
    distractors: list[Chunk] = []
    for index, para in enumerate(raw["paragraphs"]):
        text = (para.get("paragraph_text") or para.get("text") or "").strip()
        is_sup = bool(para.get("is_supporting", para.get("is_support", False)))
        sentences = _sentences(text)
        chunk = Chunk(
            chunk_id=f"{qid}::{index}",
            title=para.get("title", "") or "",
            text=text,
            sentences=sentences,
            provenance="gold" if is_sup else "distractor",
            supporting_sentence_ids=tuple(range(len(sentences))) if is_sup else (),
        )
        (gold if is_sup else distractors).append(chunk)

    hops = raw.get("question_decomposition") or []
    return QAExample(
        qid=qid,
        question=str(raw["question"]).strip(),
        answer=str(raw["answer"]).strip(),
        gold_chunks=gold,
        distractor_pool=distractors,
        meta={"type": f"{len(hops)}hop" if hops else "musique", "level": None, "source": source},
    )


def load_examples(split: str = "validation", *, limit: int | None = None, repo: str = DEFAULT_REPO) -> Iterator[QAExample]:
    from datasets import load_dataset

    dataset = load_dataset(repo, split=split)
    for index, raw in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        if not str(raw.get("answer", "")).strip():   # skip unanswerable items (MuSiQue-Full)
            continue
        yield parse_musique(raw, source="musique")
