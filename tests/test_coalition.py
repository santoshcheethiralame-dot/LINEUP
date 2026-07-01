import os
import tempfile

from lineup.data.coalition import ChainSplitScenarioBuilder, DesignedCoalition, from_recipe, read_designed, write_designed
from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample

POOL = {"year": [], "number": [], "entity": ["Antwerp", "Rotterdam"]}
OR_POOL = {"year": [], "number": [], "entity": ["Alexandre Bartholdi", "Henri Banks"]}


def _bridge_example() -> QAExample:
    link = Chunk(
        "q::0", "Landmark Tower",
        "The Landmark Tower was designed by Otto Vane.",
        ["The Landmark Tower was designed by Otto Vane."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    value = Chunk(
        "q::1", "Otto Vane",
        "Otto Vane was born in Ghent.",
        ["Otto Vane was born in Ghent."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    distractors = [
        Chunk(f"q::{i}", f"Topic {chr(63 + i)} Hall", f"Notes on topic {i}.", [f"Notes on topic {i}."])
        for i in range(2, 8)
    ]
    return QAExample(
        "q", "Where was the designer of the Landmark Tower born?", "Ghent", [link, value], distractors
    )


def _or_example() -> QAExample:
    gold_a = Chunk(
        "q::0", "Eiffel Tower",
        "The Eiffel Tower was designed by Gustave Eiffel.",
        ["The Eiffel Tower was designed by Gustave Eiffel."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    gold_b = Chunk(
        "q::1", "Paris", "Paris is in France.", ["Paris is in France."],
        provenance="gold", supporting_sentence_ids=(0,),
    )
    distractors = [
        Chunk(f"q::{i}", f"D{i}", f"Topic {i} text.", [f"Topic {i} text."])
        for i in range(2, 8)
    ]
    return QAExample("q", "Who designed the Eiffel Tower?", "Gustave Eiffel", [gold_a, gold_b], distractors)


def test_chain_split_plants_an_and_pair():
    built = ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(_bridge_example())
    assert built is not None
    scenario, designed = built

    link = next(chunk for chunk in scenario.chunks if chunk.provenance == "link")
    value = next(chunk for chunk in scenario.chunks if chunk.provenance == "misleading")
    wrong = scenario.recipe.intended_wrong_answer
    fake = value.title

    assert designed.structure == "and"
    assert designed.threshold == 2
    assert set(designed.cover_chunk_ids) == {link.chunk_id, value.chunk_id}
    assert designed.designed_sufficient_sets() == [frozenset({link.chunk_id, value.chunk_id})]

    # The link passage points the bridge at the fabricated entity and carries no value at all.
    assert fake in link.text
    assert "Otto Vane" not in link.text
    assert wrong not in link.text
    assert "Ghent" not in link.text
    # The value passage is about the fabricated entity and carries only the wrong value.
    assert fake in value.text
    assert wrong in value.text
    assert "Ghent" not in value.text
    assert "Otto Vane" not in value.text
    # The wrong value and the fabricated entity appear nowhere else in the context.
    others = [chunk for chunk in scenario.chunks if chunk.chunk_id not in designed.cover_chunk_ids]
    assert all(wrong not in chunk.text for chunk in others)
    assert all(fake not in chunk.text for chunk in others)
    # Gold survives intact, so the true chain stays available and errors are organic.
    assert sum(chunk.provenance == "gold" for chunk in scenario.chunks) == 2


def test_chain_split_is_deterministic():
    first = ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(_bridge_example())
    second = ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(_bridge_example())
    assert first[0].recipe.order == second[0].recipe.order
    assert first[0].recipe.intended_wrong_answer == second[0].recipe.intended_wrong_answer
    assert first[1] == second[1]


def test_chain_split_needs_a_redirectable_bridge():
    named = _bridge_example()
    named.question = "Where was Otto Vane born?"
    assert ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(named) is None

    unlinked = _bridge_example()
    unlinked.gold_chunks[0].text = "The Landmark Tower is very tall."
    unlinked.gold_chunks[0].sentences = ["The Landmark Tower is very tall."]
    assert ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(unlinked) is None


def test_from_recipe_recovers_the_or_cover():
    scenario = ScenarioBuilder(answer_pool=OR_POOL, k=7, seed=0, n_decoys=2).build(_or_example())
    designed = from_recipe(scenario)
    assert designed.structure == "or"
    assert designed.threshold == 1
    assert len(designed.cover_chunk_ids) == 3  # the near-miss and both decoys
    assert designed.designed_sufficient_sets() == [frozenset({cid}) for cid in designed.cover_chunk_ids]
    assert designed.intended_wrong_answer == scenario.recipe.intended_wrong_answer


def test_from_recipe_is_none_when_nothing_was_planted():
    scenario = ScenarioBuilder(answer_pool=OR_POOL, k=6, seed=0, natural=True).build(_or_example())
    assert from_recipe(scenario) is None


def test_threshold_generalizes_to_k_of_n():
    designed = DesignedCoalition(
        qid="q", structure="or", cover_chunk_ids=["a", "b", "c"], threshold=2, intended_wrong_answer="x"
    )
    assert designed.designed_sufficient_sets() == [
        frozenset({"a", "b"}), frozenset({"a", "c"}), frozenset({"b", "c"})
    ]


def test_designed_round_trip_on_disk():
    built = ChainSplitScenarioBuilder(answer_pool=POOL, k=6, seed=0).build(_bridge_example())
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "designed.jsonl")
        write_designed(path, [built[1]])
        restored = read_designed(path)
    assert restored == [built[1]]
