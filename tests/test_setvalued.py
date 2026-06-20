import pytest

from lineup.data.schema import CaseRoles, ChunkRole, ChunkScore, MethodPrediction
from lineup.setvalued import attribution_recovery


def _role(chunk_id, provenance, role, causal):
    return ChunkRole(
        chunk_id=chunk_id, provenance=provenance, role=role, causal=causal,
        salient=(role in ("culprit", "misleading")), now_correct=False,
        delta_logprob=0.0, answer_without="",
    )


def _case(qid, roles, correct=False):
    return CaseRoles(qid, "q", "gold", "wrong", correct, roles)


def _pred(qid, scores):
    # scores: list of (chunk_id, provenance, score); predicted = the argmax
    chunk_scores = [ChunkScore(cid, prov, s) for cid, prov, s in scores]
    top = max(chunk_scores, key=lambda cs: cs.score)
    return MethodPrediction(qid, "good", top.chunk_id, chunk_scores)


# A: a clean single culprit (the near-miss alone drove the error)
CASE_A = _case("A", [
    _role("A::g", "gold", "inert", False),
    _role("A::m", "misleading", "culprit", True),
    _role("A::d", "distractor", "inert", False),
])
PRED_A = _pred("A", [("A::m", "misleading", 10.0), ("A::g", "gold", 1.0), ("A::d", "distractor", 0.0)])

# B: a redundant coalition — two chunks assert the wrong value, neither is individually causal
CASE_B = _case("B", [
    _role("B::g", "gold", "inert", False),
    _role("B::m", "misleading", "misleading", False),
    _role("B::de", "decoy", "misleading", False),
    _role("B::d", "distractor", "inert", False),
])
PRED_B = _pred("B", [("B::m", "misleading", 5.0), ("B::de", "decoy", 4.0), ("B::g", "gold", 1.0), ("B::d", "distractor", 0.0)])

# C: no single culprit, and the method fingers an innocent gold chunk
CASE_C = _case("C", [
    _role("C::g", "gold", "inert", False),
    _role("C::m", "misleading", "misleading", False),
    _role("C::de", "decoy", "misleading", False),
])
PRED_C = _pred("C", [("C::g", "gold", 9.0), ("C::m", "misleading", 2.0), ("C::de", "decoy", 1.0)])


def test_top1_cannot_cover_a_coalition():
    # On the coalition case alone, the top-1 pick covers half the responsible set; top-2 covers all.
    [report] = attribution_recovery([CASE_B], [PRED_B])
    assert report.recall_at_1 == 0.5
    assert report.recall_at_k == 1.0


def test_recovery_and_reliability_aggregate():
    cases = [CASE_A, CASE_B, CASE_C]
    preds = [PRED_A, PRED_B, PRED_C]
    [report] = attribution_recovery(cases, preds)

    assert report.n == 3
    assert report.mean_responsible == pytest.approx(5 / 3)        # |{m}|, |{m,de}|, |{m,de}|
    assert report.recall_at_1 == pytest.approx((1 + 0.5 + 0) / 3)
    assert report.recall_at_k == pytest.approx((1 + 1 + 0.5) / 3)
    assert report.recall_at_k > report.recall_at_1                # the set beats the single pick
    assert report.innocent_gold_rate == 0.5                       # 1 of the 2 no-culprit cases fingers gold
    # the margin is larger exactly when there is a single culprit and the pick is causal
    assert report.single_culprit_auroc == 1.0
    assert report.reliability_auroc == 1.0


def test_missing_responsible_set_is_skipped():
    bare = _case("D", [_role("D::g", "gold", "inert", False), _role("D::d", "distractor", "inert", False)])
    pred = _pred("D", [("D::g", "gold", 1.0), ("D::d", "distractor", 0.0)])
    assert attribution_recovery([bare], [pred]) == []
