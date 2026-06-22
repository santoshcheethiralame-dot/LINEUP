from types import SimpleNamespace

from lineup.selective import SIGNALS, best_signal, build_cases, risk_coverage, set_recall_on_abstained, split


def _score(chunk_id, value):
    return SimpleNamespace(chunk_id=chunk_id, score=value)


def _role(chunk_id, provenance, role, causal):
    return SimpleNamespace(chunk_id=chunk_id, provenance=provenance, role=role, causal=causal)


def _fixtures():
    # A: one culprit (chunk b), top-1 lands on it -> a hit, well-posed.
    case_a = SimpleNamespace(
        qid="q1",
        chunk_roles=[
            _role("q1::a", "gold", "inert", False),
            _role("q1::b", "misleading", "culprit", True),
            _role("q1::c", "distractor", "inert", False),
        ],
    )
    pred_a = SimpleNamespace(
        qid="q1", method="contextcite", predicted_culprit_id="q1::b",
        chunk_scores=[_score("q1::a", 0.1), _score("q1::b", 5.0), _score("q1::c", 0.2)],
    )
    # B: no single culprit (coalition), responsible set has two chunks -> not a hit, ill-posed.
    case_b = SimpleNamespace(
        qid="q2",
        chunk_roles=[
            _role("q2::x", "misleading", "misleading", False),
            _role("q2::y", "decoy", "misleading", False),
            _role("q2::z", "gold", "inert", False),
        ],
    )
    pred_b = SimpleNamespace(
        qid="q2", method="contextcite", predicted_culprit_id="q2::x",
        chunk_scores=[_score("q2::x", 1.0), _score("q2::y", 0.9), _score("q2::z", 0.8)],
    )
    return [case_a, case_b], [pred_a, pred_b]


def test_build_cases_labels_hit_and_responsible_set():
    cases, preds = _fixtures()
    built = {case.key: case for case in build_cases(cases, preds, source="s")}
    a, b = built["s/q1"], built["s/q2"]
    assert a.hit and a.well_posed and a.responsible == {"q1::b"}
    assert not b.hit and not b.well_posed and b.responsible == {"q2::x", "q2::y"}


def test_best_signal_and_risk_coverage():
    cases, preds = _fixtures()
    built = build_cases(cases, preds, source="s")
    assert best_signal(built) in SIGNALS
    assert risk_coverage(built, "margin", (1.0,))[1.0] == 0.5  # one of two cases is a hit
    # abstain on the low-margin half (case B): its effect-set covers the 2-chunk responsible set
    recall_1, recall_k = set_recall_on_abstained(built, "margin", 0.5)
    assert recall_1 == 0.5 and recall_k == 1.0


def test_split_is_deterministic():
    cases, preds = _fixtures()
    built = build_cases(cases, preds, source="s")
    assert split(built) == split(built)
