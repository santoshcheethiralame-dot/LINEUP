from lineup.agreement import compare_models
from lineup.data.schema import CaseRoles, ChunkRole


def _role(chunk_id, role):
    return ChunkRole(chunk_id, "x", role, False, False, False, 0.0, "")


def _case(qid, correct, roles):
    return CaseRoles(qid, "q", "g", "a", correct, [_role(cid, role) for cid, role in roles])


def test_identical_runs_agree_fully():
    cases = [
        _case("q1", False, [("c1", "culprit"), ("c2", "misleading")]),
        _case("q2", False, [("c1", "culprit"), ("c2", "inert")]),
    ]
    report = compare_models(cases, cases)
    assert report.n_common == 2
    assert report.n_both_wrong == 2
    assert report.same_culprit_rate == 1.0
    assert report.culprit_jaccard == 1.0
    assert report.role_agreement == 1.0
    assert report.role_kappa == 1.0


def test_swapped_culprit_is_full_disagreement():
    a = [_case("q1", False, [("c1", "culprit"), ("c2", "misleading")])]
    b = [_case("q1", False, [("c1", "misleading"), ("c2", "culprit")])]
    report = compare_models(a, b)
    assert report.same_culprit_rate == 0.0
    assert report.culprit_jaccard == 0.0
    assert report.role_agreement == 0.0


def test_only_cases_wrong_in_both_are_compared():
    a = [_case("q1", True, [("c1", "inert")]), _case("q2", False, [("c1", "culprit")])]
    b = [_case("q1", False, [("c1", "culprit")]), _case("q2", False, [("c1", "culprit")])]
    report = compare_models(a, b)
    assert report.n_common == 2
    assert report.n_both_wrong == 1
    assert report.wrong_rate_a == 0.5
    assert report.wrong_rate_b == 1.0
    assert report.same_culprit_rate == 1.0
