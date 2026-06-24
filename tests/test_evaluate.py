from lineup.data.schema import CaseRoles, ChunkRole
from lineup.evaluate import evaluate, leaderboard_markdown, no_culprit_rate, predictions_from_scores


def _role(cid, provenance, role, causal, salient):
    return ChunkRole(
        chunk_id=cid, provenance=provenance, role=role, causal=causal, salient=salient,
        now_correct=False, delta_logprob=0.0, answer_without="",
    )


def _cases():
    # case A: a clean single culprit (a1 is causal + salient)
    a = CaseRoles(
        qid="a", question="q", gold_answer="g", original_answer="w", original_correct=False,
        chunk_roles=[
            _role("a1", "misleading", "culprit", True, True),
            _role("a2", "distractor", "inert", False, False),
            _role("a3", "gold", "inert", False, False),
        ],
    )
    # case B: no single culprit (two salient-but-not-causal chunks share the blame)
    b = CaseRoles(
        qid="b", question="q", gold_answer="g", original_answer="w", original_correct=False,
        chunk_roles=[
            _role("b1", "misleading", "misleading", False, True),
            _role("b2", "decoy", "misleading", False, True),
            _role("b3", "gold", "inert", False, False),
        ],
    )
    return [a, b]


def test_no_culprit_rate():
    assert no_culprit_rate(_cases()) == 0.5  # one of two cases has no culprit


def test_predictions_from_scores_picks_argmax():
    preds = predictions_from_scores("m", {"a": {"a1": 9.0, "a2": 1.0}})
    assert preds[0].predicted_culprit_id == "a1"


def test_evaluate_perfect_method():
    cases = _cases()
    scores_by_qid = {"a": {"a1": 9.0, "a2": 1.0, "a3": 0.0}, "b": {"b1": 9.0, "b2": 8.0, "b3": 0.0}}
    preds = predictions_from_scores("perfect", scores_by_qid)
    [result] = evaluate(cases, preds)

    assert result.method == "perfect"
    assert result.n_wrong == 2
    assert result.n_with_culprit == 1
    assert result.top1_culprit_accuracy == 1.0          # finds the culprit when one exists
    assert result.recall_at_k >= result.recall_at_1     # the set never recovers less than a single pick
    assert "perfect" in leaderboard_markdown([result])
