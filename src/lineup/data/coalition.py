"""Synthetic coalitions with designed ground truth.

The scenario builder already plants redundant support for the wrong value — the near-miss plus
n decoys — an OR-cover: any one member alone sustains the wrong answer. This module makes that
designed structure explicit and machine-readable (``from_recipe`` retrofits every constructed
scenario), and adds its dual: a chain-split AND-pair, where one passage asserts a fabricated
bridge entity and another gives that entity the wrong value, so only the two together connect
the question to the wrong answer. Designed truth is what set-valued attribution is validated
against; whether the model actually behaves as designed is measured downstream, never assumed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import random
from itertools import combinations

from .misleading import _answer_pattern, _substitute, find_answer_location, substitution_check
from .schema import Chunk, QAExample, Recipe, Scenario
from .serialization import _read_jsonl, _write_jsonl
from .substitution import classify_answer, perturb_value


@dataclasses.dataclass
class DesignedCoalition:
    """The support structure a scenario was built to carry.

    ``threshold`` is how many cover members jointly suffice: 1 for an OR-cover (each member
    alone asserts the wrong value), len(cover) for an AND-pair (only together do they).
    """

    qid: str
    structure: str                 # "or" or "and"
    cover_chunk_ids: list[str]
    threshold: int
    intended_wrong_answer: str

    def designed_sufficient_sets(self) -> list[frozenset[str]]:
        """The minimal sufficient sets the construction intends."""
        return [frozenset(combo) for combo in combinations(self.cover_chunk_ids, self.threshold)]


def from_recipe(scenario: Scenario) -> DesignedCoalition | None:
    """The designed OR-cover of an already-built scenario: the near-miss and every decoy each
    assert the planted wrong value independently. None when nothing was planted (natural and
    retrieval-only scenarios). Retrofits designed truth onto every existing constructed run."""
    wrong = scenario.recipe.intended_wrong_answer
    cover = [chunk.chunk_id for chunk in scenario.chunks if chunk.provenance in ("misleading", "decoy")]
    if not wrong or not cover:
        return None
    return DesignedCoalition(
        qid=scenario.qid,
        structure="or",
        cover_chunk_ids=cover,
        threshold=1,
        intended_wrong_answer=wrong,
    )


def write_designed(path, items) -> None:
    _write_jsonl(path, items, dataclasses.asdict)


def read_designed(path) -> list[DesignedCoalition]:
    return _read_jsonl(path, lambda data: DesignedCoalition(**data))


class ChainSplitScenarioBuilder:
    """Plant the AND dual of the redundant cover: a fabricated two-hop chain split across two
    passages. The link passage rewrites the gold paragraph that bridges to the answer entity so
    it points at a fabricated entity instead; the value passage rewrites the answer paragraph to
    be about that entity and to carry the wrong value. Neither passage alone connects the
    question to the wrong value, so the designed minimal sufficient set is the pair — and the
    link passage is a designed silent chunk: causal by construction, never salient. Gold stays
    in context; the trap is available, not forced, so errors remain organic."""

    def __init__(self, *, answer_pool: dict, k: int = 10, seed: int = 0):
        self.answer_pool = answer_pool
        self.k = k
        self.seed = seed

    def _rng(self, qid: str, salt: str) -> random.Random:
        # Same stable per-question seeding as ScenarioBuilder, so substitution and ordering
        # draw from independent streams.
        digest = hashlib.sha256(f"{self.seed}:{salt}:{qid}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def build(self, example: QAExample) -> tuple[Scenario, DesignedCoalition] | None:
        if substitution_check(example) is not None:
            return None
        location = find_answer_location(example)
        if location is None:
            return None
        value_source, sentence_id = location

        # The bridge entity is the answer paragraph's title. If the question already names it,
        # the chain cannot be redirected — the model needs no link passage to reach the value.
        bridge = value_source.title.strip()
        if not bridge or _answer_pattern(bridge).search(example.question):
            return None
        answer = example.answer.strip()
        link_source = next(
            (
                chunk for chunk in example.gold_chunks
                if chunk.chunk_id != value_source.chunk_id
                and _answer_pattern(bridge).search(chunk.text)
                and not _answer_pattern(answer).search(chunk.text)
            ),
            None,
        )
        if link_source is None:
            return None

        order_rng = self._rng(example.qid, "order")
        distractor_pool = list(example.distractor_pool)
        order_rng.shuffle(distractor_pool)
        gold = list(example.gold_chunks)
        n_distractors = max(0, self.k - len(gold) - 2)
        distractors = distractor_pool[:n_distractors]

        avoid_context = " ".join(
            [example.question]
            + [sentence for chunk in gold for sentence in chunk.sentences]
            + [sentence for chunk in distractors for sentence in chunk.sentences]
        )

        # The fabricated entity is a real title from the unused distractor pool — plausible,
        # retrieval-adjacent, and absent from everything the model will actually see.
        fake = next(
            (
                chunk.title.strip() for chunk in distractor_pool[n_distractors:]
                if chunk.title.strip()
                and chunk.title.strip().lower() != bridge.lower()
                and not _answer_pattern(chunk.title.strip()).search(avoid_context)
            ),
            None,
        )
        if fake is None:
            return None

        replacement = perturb_value(
            answer,
            classify_answer(answer),
            self.answer_pool,
            self._rng(example.qid, "substitution"),
            context=avoid_context + " " + fake,
        )
        if not replacement or replacement.lower() == answer.lower():
            return None

        link_sentences = _substitute(bridge, fake, link_source.sentences)
        if link_sentences == link_source.sentences:
            return None
        value_sentences = _substitute(answer, replacement, value_source.sentences)
        if value_sentences == value_source.sentences:
            return None
        value_sentences = _substitute(bridge, fake, value_sentences)

        link_chunk = Chunk(
            chunk_id=f"{example.qid}::link",
            title=link_source.title,
            text=" ".join(link_sentences).strip(),
            sentences=link_sentences,
            provenance="link",
        )
        value_chunk = Chunk(
            chunk_id=f"{example.qid}::value",
            title=fake,
            text=" ".join(value_sentences).strip(),
            sentences=value_sentences,
            provenance="misleading",
        )

        chunks = gold + [link_chunk, value_chunk] + distractors
        order_rng.shuffle(chunks)
        recipe = Recipe(
            seed=self.seed,
            k=len(chunks),
            original_value=answer,
            intended_wrong_answer=replacement,
            substitution_type=classify_answer(answer),
            source_gold_chunk_id=value_source.chunk_id,
            source_sentence_id=sentence_id,
            gold_chunk_ids=[chunk.chunk_id for chunk in gold],
            misleading_chunk_id=value_chunk.chunk_id,
            distractor_chunk_ids=[chunk.chunk_id for chunk in distractors],
            decoy_chunk_id="",
            order=[chunk.chunk_id for chunk in chunks],
        )
        scenario = Scenario(
            qid=example.qid,
            question=example.question,
            gold_answer=example.answer,
            chunks=chunks,
            recipe=recipe,
            meta={**example.meta, "coalition_structure": "and"},
        )
        designed = DesignedCoalition(
            qid=example.qid,
            structure="and",
            cover_chunk_ids=[link_chunk.chunk_id, value_chunk.chunk_id],
            threshold=2,
            intended_wrong_answer=replacement,
        )
        return scenario, designed
