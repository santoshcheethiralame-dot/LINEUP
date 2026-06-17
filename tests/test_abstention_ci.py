from lineup.data.schema import GenerationResult
from lineup.downstream import bootstrap_abstention_auroc, evaluate_abstention


def _gen(qid, correct, logprob):
    return GenerationResult(qid, "q", "g", "a", correct, "exact", False, logprob)


GENERATIONS = [
    _gen("q1", True, -0.1),
    _gen("q2", True, -0.2),
    _gen("q3", True, -0.3),
    _gen("q4", False, -1.0),
    _gen("q5", False, -1.1),
    _gen("q6", False, -1.2),
]


def test_auroc_interval_brackets_the_point():
    reports = {report.signal: report for report in evaluate_abstention(GENERATIONS, [], None)}
    point = reports["self_logprob"].auroc
    low, high = bootstrap_abstention_auroc(GENERATIONS, [], None, n_boot=500, seed=0)["self_logprob"]
    assert point is not None and point >= 0.9   # the signal cleanly separates correct from wrong
    assert low is not None and low <= point <= high


def test_auroc_bootstrap_is_seed_reproducible():
    first = bootstrap_abstention_auroc(GENERATIONS, [], None, n_boot=200, seed=3)
    second = bootstrap_abstention_auroc(GENERATIONS, [], None, n_boot=200, seed=3)
    assert first == second
