from types import SimpleNamespace

from lineup.conformal import RankItem, build_ranks, calibrate_tau, coverage_and_size, top1_coverage


def _case(qid, culprit_ids):
    chunks = [f"{qid}::a", f"{qid}::b", f"{qid}::c"]
    roles = [SimpleNamespace(chunk_id=cid, role="culprit" if cid in culprit_ids else "inert") for cid in chunks]
    return SimpleNamespace(qid=qid, chunk_roles=roles)


def _pred(qid, order):
    scores = [SimpleNamespace(chunk_id=cid, score=float(len(order) - i)) for i, cid in enumerate(order)]
    return SimpleNamespace(qid=qid, method="contextcite", chunk_scores=scores, predicted_culprit_id=order[0])


def test_build_ranks_keeps_only_single_culprit_cases():
    cases = [_case("q1", {"q1::b"}), _case("q2", set()), _case("q3", {"q3::a", "q3::b"})]
    preds = [_pred("q1", ["q1::a", "q1::b", "q1::c"]), _pred("q2", ["q2::a", "q2::b", "q2::c"]), _pred("q3", ["q3::a", "q3::b", "q3::c"])]
    items = build_ranks(cases, preds, source="s")
    assert len(items) == 1                 # q2 (no culprit) and q3 (two culprits) excluded
    assert items[0].rank == 2              # culprit b is second by effect score


def test_calibrate_coverage_and_top1():
    items = [RankItem(f"k{i}", "baseline", rank, 3) for i, rank in enumerate([1, 1, 2, 2, 3])]
    assert calibrate_tau(items, 0.1) == 3  # conservative quantile reaches the largest rank
    coverage, size = coverage_and_size(items, 2)
    assert coverage == 0.8 and size == 2.0
    assert top1_coverage(items) == 0.4
