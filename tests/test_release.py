from pathlib import Path

from lineup.data.scenario import ScenarioBuilder
from lineup.data.schema import CaseRoles, Chunk, QAExample
from lineup.data.serialization import read_roles, read_scenarios
from lineup.data.substitution import build_answer_pool
from lineup.release import (
    build_manifest,
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
    assert {record["role"] for record in records} <= {"culprit", "misleading", "silent", "inert", ""}


def test_case_records_one_per_case():
    scenarios, cases = _load()
    records = case_records(scenarios, cases)
    assert len(records) == len(scenarios)
    for record in records:
        assert len(record["chunk_order"]) == record["num_chunks"]


def test_role_counts_match_the_labelled_rows():
    scenarios, cases = _load()
    counts = role_counts(cases)
    labelled = [record for record in chunk_records(scenarios, cases) if record["role"]]
    assert sum(counts.values()) == len(labelled)


def test_data_card_has_frontmatter_and_sections():
    scenarios, cases = _load()
    stats = dataset_statistics(scenarios, cases)
    stats.update(model="Qwen/Qwen2.5-7B-Instruct", seed=0, source_dataset="hotpotqa/hotpot_qa", source_split="validation")
    card = data_card(stats)
    assert card.startswith("---\n")
    assert "license: mit" in card
    assert "## Roles" in card
    assert "misleading-as-culprit" in card
    assert "Qwen/Qwen2.5-7B-Instruct" in card   # the injected run metadata renders, not n/a


def test_release_columns_stay_typed_with_no_labelled_cases():
    scenarios, _ = _load()
    correct = [CaseRoles(s.qid, s.question, s.gold_answer, s.gold_answer, True, []) for s in scenarios]
    records = chunk_records(scenarios, correct)
    assert records
    assert all(record["role"] == "" for record in records)
    assert all(isinstance(record["is_causal"], bool) for record in records)
    assert all(isinstance(record["delta_logprob"], float) for record in records)


def test_build_manifest_carries_the_reproducibility_fields():
    manifest = build_manifest(
        lineup_version="0.0.1", model="m", seed=0, source_dataset="d", source_split="validation",
        created="2026-01-01T00:00:00", git_commit="abc", packages={"numpy": "2.0"}, statistics={"n_cases": 1},
    )
    assert manifest["model"] == "m" and manifest["seed"] == 0
    assert manifest["source"]["dataset"] == "d"
    assert manifest["git_commit"] == "abc" and manifest["packages"]["numpy"] == "2.0"


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
