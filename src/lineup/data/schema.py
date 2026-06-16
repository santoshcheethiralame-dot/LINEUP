from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    sentences: list[str] = field(default_factory=list)
    provenance: str = "distractor"             # "gold", "distractor", or "misleading"
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


@dataclass
class Recipe:
    """Everything needed to regenerate a scenario, and to interpret it later."""

    seed: int
    k: int
    original_value: str               # the true answer that was perturbed
    intended_wrong_answer: str        # the value the misleading chunk asserts instead
    substitution_type: str            # "year", "number", or "entity"
    source_gold_chunk_id: str         # gold chunk the misleading chunk was cloned from
    source_sentence_id: int           # sentence within it that carried the answer
    gold_chunk_ids: list[str]
    misleading_chunk_id: str
    distractor_chunk_ids: list[str]
    order: list[str]                  # chunk ids in the order presented to the model


@dataclass
class Scenario:
    qid: str
    question: str
    gold_answer: str
    chunks: list[Chunk]               # ordered as presented to the model
    recipe: Recipe
    meta: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """The model's answer to one scenario and how it was judged."""

    qid: str
    question: str
    gold_answer: str
    model_answer: str
    is_correct: bool
    judged_by: str                    # "exact" or "judge"
    matched_intended_wrong: bool      # did the answer echo the planted misleading value?
    answer_logprob: float             # mean per-token logprob of the answer (a confidence proxy)
