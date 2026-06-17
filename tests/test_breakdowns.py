from pathlib import Path

from lineup.breakdowns import by_culprit_position, by_substitution_type
from lineup.data.serialization import read_predictions, read_roles, read_scenarios

SAMPLE = Path(__file__).resolve().parents[1] / "app" / "sample"


def _load():
    scenarios = read_scenarios(SAMPLE / "scenarios.jsonl")
    cases = [case for case in read_roles(SAMPLE / "roles.jsonl") if not case.original_correct]
    predictions = read_predictions(SAMPLE / "predictions.jsonl")
    return scenarios, cases, predictions


def test_substitution_breakdown_groups_and_scores():
    scenarios, cases, predictions = _load()
    grouped = by_substitution_type(scenarios, cases, predictions)
    assert grouped
    counted = sum(reports[0].n_cases for reports in grouped.values() if reports)
    assert counted == len(cases)   # the groups partition the cases
    for reports in grouped.values():
        assert {report.method for report in reports} <= {"contextcite", "lexical_similarity", "llm_judge", "single_chunk"}


def test_position_breakdown_is_keyed_by_integer_positions():
    scenarios, cases, predictions = _load()
    grouped = by_culprit_position(scenarios, cases, predictions)
    assert all(isinstance(position, int) for position in grouped)
