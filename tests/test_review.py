from pathlib import Path

from lineup.data.serialization import read_roles, read_scenarios
from lineup.review import read_review_sheet, review_agreement, sample_review_rows, write_review_sheet

SAMPLE = Path(__file__).resolve().parents[1] / "app" / "sample"


def test_sampled_rows_are_contested_roles_left_blank():
    scenarios = read_scenarios(SAMPLE / "scenarios.jsonl")
    cases = [case for case in read_roles(SAMPLE / "roles.jsonl") if not case.original_correct]
    rows = sample_review_rows(scenarios, cases, n=50, seed=0)
    assert rows
    assert all(row["oracle_role"] in ("culprit", "misleading") for row in rows)
    assert all(row["human_role"] == "" for row in rows)


def test_agreement_counts_only_filled_rows(tmp_path):
    rows = [
        {"qid": "q1", "chunk_id": "c1", "question": "", "model_answer": "", "chunk_text": "", "oracle_role": "culprit", "human_role": "culprit"},
        {"qid": "q2", "chunk_id": "c2", "question": "", "model_answer": "", "chunk_text": "", "oracle_role": "misleading", "human_role": "culprit"},
        {"qid": "q3", "chunk_id": "c3", "question": "", "model_answer": "", "chunk_text": "", "oracle_role": "culprit", "human_role": ""},
    ]
    path = tmp_path / "review.csv"
    write_review_sheet(path, rows)
    result = review_agreement(read_review_sheet(path))
    assert result["n"] == 2              # the unfilled row is ignored
    assert result["agreement"] == 0.5    # one of the two filled rows matches the oracle
