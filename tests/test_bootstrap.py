from lineup.data.schema import CaseRoles, ChunkRole, ChunkScore, MethodPrediction
from lineup.scoring import bootstrap_intervals, score_predictions

_ROLES = [("c1", "culprit"), ("c2", "misleading"), ("c3", "inert")]


def _role(chunk_id, role):
    return ChunkRole(chunk_id, "x", role, False, False, False, 0.0, "")


def _case(qid):
    return CaseRoles(qid, "q", "g", "a", False, [_role(cid, role) for cid, role in _ROLES])


def _prediction(qid, culprit, scores):
    return MethodPrediction(
        qid, "m", culprit, [ChunkScore(cid, "x", value) for cid, value in scores.items()]
    )


CASES = [_case("q1"), _case("q2"), _case("q3")]
PREDICTIONS = [
    _prediction("q1", "c2", {"c1": 0.1, "c2": 0.9, "c3": 0.0}),
    _prediction("q2", "c1", {"c1": 0.9, "c2": 0.1, "c3": 0.0}),
    _prediction("q3", "c1", {"c1": 0.9, "c2": 0.1, "c3": 0.0}),
]


def test_point_estimates_are_what_we_expect():
    report = score_predictions(CASES, PREDICTIONS)[0]
    assert abs(report.misleading_as_culprit_rate - 1 / 3) < 1e-9
    assert abs(report.culprit_over_misleading_winrate - 2 / 3) < 1e-9


def test_interval_brackets_the_point_estimate():
    report = score_predictions(CASES, PREDICTIONS)[0]
    interval = bootstrap_intervals(CASES, PREDICTIONS, n_boot=500, seed=0)["m"]
    low, high = interval["misleading_as_culprit_rate"]
    assert low is not None
    assert low <= report.misleading_as_culprit_rate <= high


def test_bootstrap_is_seed_reproducible():
    first = bootstrap_intervals(CASES, PREDICTIONS, n_boot=200, seed=1)
    second = bootstrap_intervals(CASES, PREDICTIONS, n_boot=200, seed=1)
    assert first == second
