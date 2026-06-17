import dataclasses

import pytest

from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.schema import Chunk, Recipe, Scenario
from lineup.data.serialization import method_prediction_from_dict
from lineup.methods import ContextCite, LexicalSimilarity, LLMJudgeCulprit, run_method

RIGHT = "Gustave Eiffel"
WRONG = "Alexandre Bartholdi"


def _scenario() -> Scenario:
    gold = Chunk("g", "Eiffel Tower", f"The tower was designed by {RIGHT}.", ["..."], provenance="gold")
    misleading = Chunk("m", "Eiffel Tower", f"The tower was designed by {WRONG}.", ["..."], provenance="misleading")
    distractor = Chunk("d", "Paris", "Paris is the capital of France.", ["..."], provenance="distractor")
    recipe = Recipe(
        seed=0, k=3, original_value=RIGHT, intended_wrong_answer=WRONG, substitution_type="entity",
        source_gold_chunk_id="g", source_sentence_id=0, gold_chunk_ids=["g"],
        misleading_chunk_id="m", distractor_chunk_ids=["d"], order=["g", "m", "d"],
    )
    return Scenario(qid="q", question="Who designed the Eiffel Tower?", gold_answer=RIGHT, chunks=[gold, misleading, distractor], recipe=recipe)


class _PicksSecond(LanguageModel):
    def generate(self, messages, max_new_tokens=None):
        return Generation("Passage 2.", [1], [-0.1])

    def score(self, messages, response):
        return Scoring([], [], [])


class _SupportsMisleading(LanguageModel):
    """The answer's logprob is high only when the misleading chunk is in the context."""

    def generate(self, messages, max_new_tokens=None):
        return Generation(WRONG, [1], [-0.1])

    def score(self, messages, response):
        present = WRONG in messages[-1].content
        return Scoring([], [], [0.0] if present else [-5.0])


def test_lexical_similarity_blames_the_chunk_matching_the_answer():
    prediction = run_method(LexicalSimilarity(), None, _scenario(), WRONG)
    assert prediction.predicted_culprit_id == "m"   # the near-miss shares the most tokens with the answer


def test_llm_judge_culprit_reads_the_picked_number():
    prediction = run_method(LLMJudgeCulprit(), _PicksSecond(), _scenario(), WRONG)
    assert prediction.predicted_culprit_id == "m"   # passage 2 is the misleading chunk


def test_run_method_records_scores_for_every_chunk():
    prediction = run_method(LexicalSimilarity(), None, _scenario(), WRONG)
    assert [score.chunk_id for score in prediction.chunk_scores] == ["g", "m", "d"]


def test_prediction_round_trip_through_dict():
    prediction = run_method(LexicalSimilarity(), None, _scenario(), WRONG)
    restored = method_prediction_from_dict(dataclasses.asdict(prediction))
    assert restored.predicted_culprit_id == prediction.predicted_culprit_id
    assert [s.chunk_id for s in restored.chunk_scores] == [s.chunk_id for s in prediction.chunk_scores]


def test_contextcite_attributes_to_the_supporting_chunk():
    pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    prediction = run_method(ContextCite(n_ablations=64, seed=0), _SupportsMisleading(), _scenario(), WRONG)
    assert prediction.predicted_culprit_id == "m"   # only the misleading chunk lifts the answer's logprob
