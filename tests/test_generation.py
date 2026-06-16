import os
import tempfile

from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.correctness import LLMJudge
from lineup.data.schema import Chunk, Recipe, Scenario
from lineup.data.serialization import read_generations, write_generations
from lineup.generation import generate_and_judge


class _Model(LanguageModel):
    """Returns a fixed answer; lets us test the Stage 3 flow without a real model."""

    def __init__(self, answer: str):
        self._answer = answer

    def generate(self, messages, max_new_tokens=None):
        return Generation(text=self._answer, token_ids=[1, 2], token_logprobs=[-0.1, -0.3])

    def score(self, messages, response):
        return Scoring(tokens=[], token_ids=[], logprobs=[])


def _scenario(gold: str = "Gustave Eiffel", wrong: str = "Alexandre Bartholdi") -> Scenario:
    chunks = [
        Chunk("a", "Eiffel Tower", "Designed by someone.", ["Designed by someone."], provenance="gold"),
        Chunk("q::misleading", "Eiffel Tower", "Designed by someone else.", ["Designed by someone else."], provenance="misleading"),
    ]
    recipe = Recipe(
        seed=0, k=2, original_value=gold, intended_wrong_answer=wrong, substitution_type="entity",
        source_gold_chunk_id="a", source_sentence_id=0, gold_chunk_ids=["a"],
        misleading_chunk_id="q::misleading", distractor_chunk_ids=[], order=["a", "q::misleading"],
    )
    return Scenario(qid="q", question="Who designed the Eiffel Tower?", gold_answer=gold, chunks=chunks, recipe=recipe)


def test_correct_answer_is_labeled_by_exact_match():
    result = generate_and_judge(_Model("Gustave Eiffel"), _scenario())
    assert result.is_correct and result.judged_by == "exact"
    assert not result.matched_intended_wrong


def test_planted_wrong_value_is_flagged():
    result = generate_and_judge(_Model("Alexandre Bartholdi"), _scenario())
    assert not result.is_correct
    assert result.matched_intended_wrong


def test_llm_judge_recovers_a_phrasing_variant():
    result = generate_and_judge(
        _Model("the engineer Gustave Eiffel"), _scenario(), llm_judge=LLMJudge(_Model("yes"))
    )
    assert result.is_correct and result.judged_by == "judge"


def test_generations_round_trip_on_disk():
    result = generate_and_judge(_Model("Alexandre Bartholdi"), _scenario())
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "generations.jsonl")
        write_generations(path, [result])
        restored = read_generations(path)
    assert len(restored) == 1
    assert restored[0].qid == result.qid
    assert restored[0].is_correct == result.is_correct
    assert restored[0].matched_intended_wrong == result.matched_intended_wrong
