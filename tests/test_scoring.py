from lineup.data.schema import CaseRoles, ChunkRole, ChunkScore, MethodPrediction
from lineup.scoring import score_predictions


def _case(qid: str, role_map: dict) -> CaseRoles:
    chunk_roles = [
        ChunkRole(cid, "?", role, role in ("culprit", "silent"), role in ("culprit", "misleading"), False, 0.0, "")
        for cid, role in role_map.items()
    ]
    return CaseRoles(qid, "q", "gold", "ans", False, chunk_roles)


def _prediction(qid: str, method: str, scores: dict) -> MethodPrediction:
    chunk_scores = [ChunkScore(cid, "?", value) for cid, value in scores.items()]
    best = max(chunk_scores, key=lambda score: score.score).chunk_id
    return MethodPrediction(qid, method, best, chunk_scores)


CASES = [_case("q1", {"g": "culprit", "m": "misleading", "d": "inert"})]


def test_method_that_picks_the_culprit_scores_perfectly():
    predictions = [_prediction("q1", "causal", {"g": 1.0, "m": 0.5, "d": 0.0})]
    report = score_predictions(CASES, predictions)[0]
    assert report.top1_culprit_accuracy == 1.0
    assert report.misleading_as_culprit_rate == 0.0
    assert report.culprit_over_misleading_winrate == 1.0


def test_salience_method_blames_the_misleading_chunk():
    predictions = [_prediction("q1", "salience", {"g": 0.4, "m": 1.0, "d": 0.0})]
    report = score_predictions(CASES, predictions)[0]
    assert report.top1_culprit_accuracy == 0.0
    assert report.misleading_as_culprit_rate == 1.0
    assert report.culprit_over_misleading_winrate == 0.0


def test_winrate_is_none_without_a_culprit_misleading_pair():
    cases = [_case("q1", {"g": "culprit", "d": "inert"})]   # no misleading chunk
    predictions = [_prediction("q1", "x", {"g": 1.0, "d": 0.0})]
    report = score_predictions(cases, predictions)[0]
    assert report.culprit_over_misleading_winrate is None
    assert report.n_pairs == 0


def test_predictions_for_unknown_cases_are_ignored():
    predictions = [_prediction("absent", "x", {"a": 1.0})]
    report = score_predictions(CASES, predictions)[0]
    assert report.n_cases == 0
    assert report.top1_culprit_accuracy == 0.0
