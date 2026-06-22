from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.substitution import build_answer_pool


def _example(qid, question, answer, gold_text):
    gold = Chunk(f"{qid}::g", "G", gold_text, [gold_text], provenance="gold", supporting_sentence_ids=(0,))
    distractors = [
        Chunk(f"{qid}::d{i}", f"P{i}", text, [text])
        for i, text in enumerate(["Paris is in France.", "Iron is a metal.", "A tower is tall."])
    ]
    return QAExample(qid, question, answer, [gold], distractors)


POOL = build_answer_pool([
    _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel", "The Eiffel Tower was designed by Gustave Eiffel."),
    _example("q2", "Who painted the Mona Lisa?", "Leonardo da Vinci", "The Mona Lisa was painted by Leonardo da Vinci."),
])


def test_natural_mode_plants_nothing():
    example = _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel", "The Eiffel Tower was designed by Gustave Eiffel.")
    scenario = ScenarioBuilder(answer_pool=POOL, k=4, seed=0, natural=True).build(example)
    assert scenario is not None
    assert all(chunk.provenance in ("gold", "distractor") for chunk in scenario.chunks)   # no misleading/decoy
    assert any(chunk.provenance == "gold" for chunk in scenario.chunks)
    assert scenario.recipe.substitution_type == "natural"
    assert scenario.recipe.misleading_chunk_id == "" and scenario.recipe.decoy_chunk_id == ""
