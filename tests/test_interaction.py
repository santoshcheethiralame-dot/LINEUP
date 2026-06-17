from lineup.backends.base import Generation, LanguageModel, Scoring
from lineup.data.schema import Chunk, GenerationResult, Recipe, Scenario
from lineup.interaction import pairwise_interactions

RIGHT = "Gustave Eiffel"
WRONG = "Alexandre Bartholdi"


class OrCoalition(LanguageModel):
    """Answers wrongly while either marked chunk is present; only removing both restores gold."""

    def generate(self, messages, max_new_tokens=None):
        content = messages[-1].content
        wrong = "AAA" in content or "BBB" in content
        return Generation(WRONG if wrong else RIGHT, [1], [-0.1])

    def score(self, messages, response):
        return Scoring([], [], [-0.1])


def _scenario():
    chunks = [
        Chunk("c1", "A", f"Marker AAA: the tower was designed by {WRONG}."),
        Chunk("c2", "B", f"Marker BBB: the tower was designed by {WRONG}."),
        Chunk("c3", "C", "Iron is a common metal."),
    ]
    recipe = Recipe(0, 3, RIGHT, WRONG, "entity", "c1", 0, ["c1"], "c2", ["c3"], ["c1", "c2", "c3"])
    return Scenario("q1", "Who designed the tower?", RIGHT, chunks, recipe)


def _original():
    return GenerationResult("q1", "Who designed the tower?", RIGHT, WRONG, False, "exact", True, -0.1)


def test_detects_a_two_chunk_coalition():
    result = pairwise_interactions(OrCoalition(), _scenario(), _original())
    assert result.has_synergy
    synergy_pairs = {(pair.chunk_id_a, pair.chunk_id_b) for pair in result.pairs if pair.synergy}
    assert ("c1", "c2") in synergy_pairs


def test_no_false_synergy_for_an_independent_chunk():
    result = pairwise_interactions(OrCoalition(), _scenario(), _original())
    for pair in result.pairs:
        if "c3" in (pair.chunk_id_a, pair.chunk_id_b):
            assert not pair.synergy
