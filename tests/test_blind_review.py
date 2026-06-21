from types import SimpleNamespace

from lineup.review import sample_scenario_rows, score_scenario


def _sources():
    chunks = [SimpleNamespace(chunk_id=f"q1::{c}", text=f"text {c}") for c in "abcdef"]
    scenario = SimpleNamespace(qid="q1", chunks=chunks)
    with_culprit = SimpleNamespace(
        qid="q1", question="Q?", original_answer="wrong", original_correct=False,
        chunk_roles=[SimpleNamespace(chunk_id="q1::b", role="culprit"), SimpleNamespace(chunk_id="q1::a", role="inert")],
    )
    chunks2 = [SimpleNamespace(chunk_id=f"q2::{c}", text=f"t {c}") for c in "abc"]
    scenario2 = SimpleNamespace(qid="q2", chunks=chunks2)
    no_culprit = SimpleNamespace(
        qid="q2", question="Q2?", original_answer="x", original_correct=False,
        chunk_roles=[SimpleNamespace(chunk_id="q2::a", role="misleading")],
    )
    return [("hotpotqa/hardtraps/qwen", [scenario, scenario2], [with_culprit, no_culprit])]


def test_blank_sheet_is_blind_with_lettered_passages():
    blank, key = sample_scenario_rows(_sources(), n=10, seed=0)
    assert len(blank) == len(key) == 2
    for row in blank:
        assert set(row) == {"row_id", "question", "model_answer", "passages", "your_pick"}
        assert "answer" not in row and "qid" not in row  # nothing that leaks the culprit
    answer = {row["row_id"]: row["answer"] for row in key}
    culprit_row = next(row for row in blank if row["question"] == "Q?")
    assert "B)" in culprit_row["passages"]  # the second chunk is rendered as option B
    assert answer[culprit_row["row_id"]] == "B"
    none_row = next(row for row in blank if row["question"] == "Q2?")
    assert answer[none_row["row_id"]] == "none"  # no-culprit case -> correct answer is "none"


def test_score_scenario_agreement_majority_none_and_inter_rater():
    _, key = sample_scenario_rows(_sources(), n=10, seed=0)
    answer = {row["row_id"]: row["answer"] for row in key}
    ids = list(answer)
    perfect = [{"row_id": r, "your_pick": answer[r]} for r in ids]
    wrong_first = [{"row_id": r, "your_pick": ("Z" if r == ids[0] else answer[r])} for r in ids]

    result = score_scenario(key, {"A": perfect, "B": wrong_first, "C": perfect})
    assert result["per_reviewer"]["A"]["agreement"] == 1.0
    assert result["per_reviewer"]["B"]["agreement"] == 0.5
    assert result["majority_vs_oracle"] == 1.0
    assert result["none_recall"] == 1.0  # humans also say "none" on the no-culprit case
    assert abs(result["inter_rater"] - 4 / 6) < 1e-9


def test_unsure_is_excluded_from_agreement():
    _, key = sample_scenario_rows(_sources(), n=10, seed=0)
    answer = {row["row_id"]: row["answer"] for row in key}
    ids = list(answer)
    # one real (correct) pick, one "unsure" -> agreement is 1.0 over the single scored row
    sheet = [{"row_id": ids[0], "your_pick": answer[ids[0]]}, {"row_id": ids[1], "your_pick": "unsure"}]
    result = score_scenario(key, {"A": sheet})
    assert result["per_reviewer"]["A"] == {"n": 1, "agreement": 1.0, "unsure": 1}
