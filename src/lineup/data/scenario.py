from __future__ import annotations

import hashlib
import random

from .misleading import MisleadingChunkBuilder, ValueSubstitutionBuilder, redundant_decoys, substitution_check
from .schema import QAExample, Recipe, Scenario


class ScenarioBuilder:
    """Assemble a single test case: gold chunks + one constructed near-miss + distractors,
    in randomized order, with a recipe that makes the whole thing reproducible."""

    def __init__(
        self,
        *,
        answer_pool: dict,
        k: int = 10,
        seed: int = 0,
        misleading_builder: MisleadingChunkBuilder | None = None,
        hard_traps: bool = False,
        n_decoys: int | None = None,
    ):
        self.answer_pool = answer_pool
        self.k = k
        self.seed = seed
        self.misleading_builder = misleading_builder or ValueSubstitutionBuilder()
        # n_decoys is the coalition-size knob; hard_traps is the boolean shorthand for n_decoys=1.
        self.n_decoys = n_decoys if n_decoys is not None else (1 if hard_traps else 0)

    def _rng(self, qid: str, salt: str) -> random.Random:
        # A stable per-question seed: the built-in hash() is salted per process and would
        # not reproduce across runs, so derive it from a fixed digest. Substitution and
        # ordering draw from independent streams, so changing one does not perturb the other.
        digest = hashlib.sha256(f"{self.seed}:{salt}:{qid}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def build(self, example: QAExample) -> Scenario | None:
        if substitution_check(example) is not None:
            return None

        order_rng = self._rng(example.qid, "order")
        gold = list(example.gold_chunks)
        distractor_pool = list(example.distractor_pool)
        order_rng.shuffle(distractor_pool)
        # k is a floor on context size: the gold chunks and the near-miss are always kept,
        # so the count is max(k, len(gold) + 1).
        n_distractors = max(0, self.k - len(gold) - 1 - self.n_decoys)
        distractors = distractor_pool[:n_distractors]

        # Build the near-miss against the full assembled context, so the planted wrong value
        # is unique to the misleading chunk and never coincides with a gold or distractor.
        avoid_context = " ".join(
            [example.question]
            + [sentence for chunk in gold for sentence in chunk.sentences]
            + [sentence for chunk in distractors for sentence in chunk.sentences]
        )
        built = self.misleading_builder.build(
            example, self.answer_pool, self._rng(example.qid, "substitution"), context=avoid_context
        )
        if built is None:
            return None
        misleading_chunk, recipe_bits = built

        extra = redundant_decoys(
            example.qid, misleading_chunk, recipe_bits["intended_wrong_answer"], self.n_decoys
        ) if self.n_decoys else []
        chunks = gold + [misleading_chunk] + extra + distractors
        order_rng.shuffle(chunks)

        recipe = Recipe(
            seed=self.seed,
            k=len(chunks),
            original_value=recipe_bits["original_value"],
            intended_wrong_answer=recipe_bits["intended_wrong_answer"],
            substitution_type=recipe_bits["substitution_type"],
            source_gold_chunk_id=recipe_bits["source_gold_chunk_id"],
            source_sentence_id=recipe_bits["source_sentence_id"],
            gold_chunk_ids=[chunk.chunk_id for chunk in gold],
            misleading_chunk_id=misleading_chunk.chunk_id,
            distractor_chunk_ids=[chunk.chunk_id for chunk in distractors],
            decoy_chunk_id=extra[0].chunk_id if extra else "",
            order=[chunk.chunk_id for chunk in chunks],
        )
        return Scenario(
            qid=example.qid,
            question=example.question,
            gold_answer=example.answer,
            chunks=chunks,
            recipe=recipe,
            meta=dict(example.meta),
        )
