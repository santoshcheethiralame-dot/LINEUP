from random import Random

from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.substitution import _sample, perturb_value
from lineup.textnorm import contains_phrase


def test_perturb_avoids_values_already_present_in_context():
    for seed in range(20):
        out = perturb_value("1950", "year", {}, Random(seed), context="he was active in 1949 and 1951 and 1948")
        assert out not in {"1949", "1951", "1948", "1950"}


def test_sample_rejects_token_overlapping_entities():
    pool = {"entity": ["Theodore Roosevelt", "Franklin Roosevelt", "Gustave Eiffel"]}
    for seed in range(20):
        # every Roosevelt candidate shares the "Roosevelt" token with the answer
        assert _sample("Roosevelt", pool, "entity", Random(seed), "") == "Gustave Eiffel"


def test_near_miss_value_is_unique_to_the_misleading_chunk():
    gold = Chunk("g", "Bio", "She was born in 1950.", ["She was born in 1950."], provenance="gold", supporting_sentence_ids=(0,))
    distractors = [
        Chunk(f"d{year}", "Other", f"A different event happened in {year}.", [f"A different event happened in {year}."])
        for year in (1948, 1949, 1951, 1952, 1953)
    ]
    example = QAExample("q1", "When was she born?", "1950", [gold], distractors)
    scenario = ScenarioBuilder(answer_pool={}, k=6, seed=0).build(example)
    assert scenario is not None
    wrong = scenario.recipe.intended_wrong_answer
    others = [chunk for chunk in scenario.chunks if chunk.provenance != "misleading"]
    assert not any(contains_phrase(chunk.text, wrong) for chunk in others)
