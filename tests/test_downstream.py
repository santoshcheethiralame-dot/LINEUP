from lineup.data.schema import CaseRoles, ChunkRole, ChunkScore, GenerationResult, MethodPrediction
from lineup.downstream import _auroc, _oracle_confidence, _risk_coverage, evaluate_abstention


def _generation(qid: str, correct: bool, logprob: float) -> GenerationResult:
    return GenerationResult(qid, "q", "gold", "ans", correct, "exact", False, logprob)


def _prediction(qid: str, method: str, scores: dict) -> MethodPrediction:
    chunk_scores = [ChunkScore(cid, "?", value) for cid, value in scores.items()]
    best = max(chunk_scores, key=lambda score: score.score).chunk_id
    return MethodPrediction(qid, method, best, chunk_scores)


def test_auroc_ranks_confidence_against_correctness():
    assert _auroc([0.9, 0.1], [True, False]) == 1.0
    assert _auroc([0.1, 0.9], [True, False]) == 0.0
    assert _auroc([0.5, 0.5], [True, False]) == 0.5     # a constant signal is chance
    assert _auroc([0.9, 0.1], [True, True]) is None      # needs both classes


def test_risk_coverage_rewards_a_well_ordered_signal():
    aurc, selective = _risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False])
    assert selective == 1.0          # the confident half is all correct
    assert aurc < 0.3


def test_self_logprob_signal_separates_correct_from_wrong():
    generations = [_generation("q1", True, -0.1), _generation("q2", False, -2.0)]
    report = next(r for r in evaluate_abstention(generations, []) if r.signal == "self_logprob")
    assert report.n == 2 and report.n_correct == 1
    assert report.auroc == 1.0       # the correct answer was the more confident one


def test_method_margin_is_reported_per_method():
    generations = [_generation("q1", True, -0.1), _generation("q2", False, -0.1)]
    predictions = [
        _prediction("q1", "contextcite", {"a": 1.0, "b": 0.0}),   # decisive
        _prediction("q2", "contextcite", {"a": 0.5, "b": 0.4}),   # not decisive
    ]
    report = next(r for r in evaluate_abstention(generations, predictions) if r.signal == "contextcite_margin")
    assert report.n == 2
    assert report.auroc == 1.0       # here the decisive case happened to be the correct one


def test_oracle_flags_a_fixable_error():
    assert _oracle_confidence(CaseRoles("q", "q", "g", "g", True, [])) == 1.0
    fixable = CaseRoles("q", "q", "g", "w", False, [ChunkRole("m", "misleading", "misleading", False, True, True, 0.0, "")])
    assert _oracle_confidence(fixable) == 0.0
    unfixable = CaseRoles("q", "q", "g", "w", False, [ChunkRole("m", "misleading", "misleading", False, True, False, 0.0, "")])
    assert _oracle_confidence(unfixable) == 1.0
