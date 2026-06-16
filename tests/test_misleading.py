from random import Random

from lineup.data.misleading import (
    ValueSubstitutionBuilder,
    find_answer_location,
    substitution_check,
)
from lineup.data.schema import Chunk, QAExample


def _example(answer: str = "Gustave Eiffel") -> QAExample:
    gold = Chunk(
        "q::0",
        "Eiffel Tower",
        "The Eiffel Tower was designed by Gustave Eiffel. It opened in 1889.",
        ["The Eiffel Tower was designed by Gustave Eiffel.", " It opened in 1889."],
        provenance="gold",
        supporting_sentence_ids=(0,),
    )
    other = Chunk(
        "q::1", "Paris", "Paris is in France.", ["Paris is in France."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    return QAExample("q", "Who designed the Eiffel Tower?", answer, [gold, other], [])


def test_find_answer_location_prefers_supporting_sentence():
    chunk, sentence_id = find_answer_location(_example())
    assert chunk.title == "Eiffel Tower"
    assert sentence_id == 0


def test_substitution_check_flags_boolean_and_missing():
    assert substitution_check(_example("yes")) == "boolean answer"
    assert substitution_check(_example("Someone Absent")) == "answer not present in gold sentences"
    assert substitution_check(_example()) is None


def test_value_substitution_injects_a_wrong_value():
    pool = {"year": [], "number": [], "entity": ["Alexandre Bartholdi"]}
    built = ValueSubstitutionBuilder().build(_example(), pool, Random(0))
    assert built is not None
    chunk, recipe_bits = built
    assert chunk.provenance == "misleading"
    assert recipe_bits["intended_wrong_answer"] == "Alexandre Bartholdi"
    assert "Alexandre Bartholdi" in chunk.text
    assert "Gustave Eiffel" not in chunk.text
