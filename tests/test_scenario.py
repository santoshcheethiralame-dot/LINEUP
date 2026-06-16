import os
import tempfile

from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.serialization import read_scenarios, scenario_from_dict, scenario_to_dict, write_scenarios

POOL = {"year": [], "number": [], "entity": ["Alexandre Bartholdi", "Henri Banks"]}


def _example() -> QAExample:
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


def test_build_assembles_controlled_context():
    scenario = ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(_example())
    assert scenario is not None
    assert len(scenario.chunks) == 5
    provenance = [chunk.provenance for chunk in scenario.chunks]
    assert provenance.count("gold") == 2
    assert provenance.count("misleading") == 1
    assert provenance.count("distractor") == 2
    assert scenario.recipe.order == [chunk.chunk_id for chunk in scenario.chunks]
    assert scenario.recipe.intended_wrong_answer in POOL["entity"]


def test_build_is_deterministic():
    first = ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(_example())
    second = ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(_example())
    assert first.recipe.order == second.recipe.order
    assert first.recipe.intended_wrong_answer == second.recipe.intended_wrong_answer


def test_boolean_answer_is_skipped():
    example = _example()
    example.answer = "yes"
    assert ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(example) is None


def test_serialization_round_trip():
    scenario = ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(_example())
    restored = scenario_from_dict(scenario_to_dict(scenario))
    assert restored.recipe.order == scenario.recipe.order
    assert [c.chunk_id for c in restored.chunks] == [c.chunk_id for c in scenario.chunks]
    assert restored.chunks[0].supporting_sentence_ids == scenario.chunks[0].supporting_sentence_ids


def test_jsonl_round_trip_on_disk():
    scenario = ScenarioBuilder(answer_pool=POOL, k=5, seed=0).build(_example())
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "scenarios.jsonl")
        write_scenarios(path, [scenario])
        restored = read_scenarios(path)
    assert len(restored) == 1
    assert restored[0].recipe.order == scenario.recipe.order
    assert restored[0].chunks[0].supporting_sentence_ids == scenario.chunks[0].supporting_sentence_ids
