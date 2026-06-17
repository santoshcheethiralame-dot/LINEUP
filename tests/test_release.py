from pathlib import Path

from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import Chunk, QAExample
from lineup.data.serialization import read_roles, read_scenarios
from lineup.data.substitution import build_answer_pool
from lineup.release import (
    case_records,
    chunk_records,
    data_card,
    dataset_statistics,
    role_counts,
)

SAMPLE = Path(__file__).resolve().parents[1] / "app" / "sample"


def _load():
    return read_scenarios(SAMPLE / "scenarios.jsonl"), read_roles(SAMPLE / "roles.jsonl")


def test_chunk_records_cover_every_passage():
    scenarios, cases = _load()
    records = chunk_records(scenarios, cases)
    assert len(records) == sum(len(scenario.chunks) for scenario in scenarios)
    assert {record["provenance"] for record in records} <= {"gold", "distractor", "misleading"}
    assert {record["role"] for record in records} <= {"culprit", "misleading", "silent", "inert", None}


def test_case_records_one_per_case():
    scenarios, cases = _load()
    records = case_records(scenarios, cases)
    assert len(records) == len(scenarios)
    for record in records:
        assert len(record["chunk_order"]) == record["num_chunks"]


def test_role_counts_match_the_labelled_rows():
    scenarios, cases = _load()
    counts = role_counts(cases)
    labelled = [record for record in chunk_records(scenarios, cases) if record["role"] is not None]
    assert sum(counts.values()) == len(labelled)


def test_data_card_has_frontmatter_and_sections():
    scenarios, cases = _load()
    card = data_card(dataset_statistics(scenarios, cases))
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "## Roles" in card
    assert "misleading-as-culprit" in card


def _example(qid, question, answer, gold_text):
    gold = Chunk(f"{qid}::g", "Gold", gold_text, [gold_text], provenance="gold", supporting_sentence_ids=(0,))
    distractors = [
        Chunk(f"{qid}::d{i}", f"Passage {i}", text, [text])
        for i, text in enumerate(
            [
                "Paris is the capital of France.",
                "Wrought iron is a malleable alloy.",
                "The Seine flows through Paris.",
                "Engineering shapes modern cities.",
                "A tower is a tall structure.",
            ]
        )
    ]
    return QAExample(qid, question, answer, [gold], distractors)


def test_scenario_construction_is_deterministic():
    examples = [
        _example("q1", "Who designed the Eiffel Tower?", "Gustave Eiffel", "The Eiffel Tower was designed by Gustave Eiffel."),
        _example("q2", "Who painted the Mona Lisa?", "Leonardo da Vinci", "The Mona Lisa was painted by Leonardo da Vinci."),
    ]
    pool = build_answer_pool(examples)
    first = [ScenarioBuilder(answer_pool=pool, k=4, seed=0).build(example) for example in examples]
    second = [ScenarioBuilder(answer_pool=pool, k=4, seed=0).build(example) for example in examples]
    assert all(scenario is not None for scenario in first)
    assert [scenario.recipe.order for scenario in first] == [scenario.recipe.order for scenario in second]
    assert [scenario.recipe.intended_wrong_answer for scenario in first] == [
        scenario.recipe.intended_wrong_answer for scenario in second
    ]
