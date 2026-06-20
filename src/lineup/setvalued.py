from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles, MethodPrediction
from .stats import auroc

# Chunks planted to assert the wrong value. Under a redundant decoy the error is supported by
# more than one of these, so the responsible set has size > 1 and a single top-1 pick cannot
# cover it. This is the set we ask attribution to recover.
_RESPONSIBLE = ("misleading", "decoy")


@dataclass
class RecoveryReport:
    method: str
    n: int
    mean_responsible: float            # average |responsible set| per case (1 baseline, ~2 redundant)
    recall_at_1: float | None          # the top-1 pick's coverage of the responsible set
    recall_at_k: float | None          # top-|responsible| by score: coverage when the count is known
    innocent_gold_rate: float | None   # of cases with no single culprit, share whose top-1 fingers a gold chunk
    reliability_auroc: float | None    # does the score margin separate causal picks from non-causal ones?
    single_culprit_auroc: float | None # does the score margin flag cases that have a single culprit at all?


def _responsible_ids(case: CaseRoles) -> set:
    return {role.chunk_id for role in case.chunk_roles if role.provenance in _RESPONSIBLE}


def _ranked_ids(prediction: MethodPrediction) -> list:
    return [score.chunk_id for score in sorted(prediction.chunk_scores, key=lambda s: s.score, reverse=True)]


def _margin(prediction: MethodPrediction) -> float:
    scores = sorted((score.score for score in prediction.chunk_scores), reverse=True)
    return scores[0] - scores[1] if len(scores) >= 2 else 0.0


def attribution_recovery(
    cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction]
) -> list[RecoveryReport]:
    """Score attribution beyond top-1: how much of the responsible chunk set each method recovers,
    and whether its own score margin is a usable confidence signal.

    Joined by question id. ``recall_at_1`` vs ``recall_at_k`` exposes the structural ceiling of a
    single-chunk answer when the cause is a coalition; the two AUROCs test whether a method knows
    when it is right and when no single culprit exists.
    """
    cases_by_qid = {case.qid: case for case in cases}
    by_method: dict = {}
    for prediction in predictions:
        by_method.setdefault(prediction.method, []).append(prediction)

    reports = []
    for method in sorted(by_method):
        recall1, recallk, sizes = [], [], []
        margins, causal_hit, has_culprit = [], [], []
        gold_when_illposed = []
        for prediction in by_method[method]:
            case = cases_by_qid.get(prediction.qid)
            if case is None:
                continue
            responsible = _responsible_ids(case)
            if not responsible:
                continue
            ranked = _ranked_ids(prediction)
            k = len(responsible)
            recall1.append(len(set(ranked[:1]) & responsible) / k)
            recallk.append(len(set(ranked[:k]) & responsible) / k)
            sizes.append(k)

            causal = {role.chunk_id: role.causal for role in case.chunk_roles}
            provenance = {role.chunk_id: role.provenance for role in case.chunk_roles}
            margins.append(_margin(prediction))
            causal_hit.append(bool(causal.get(prediction.predicted_culprit_id, False)))
            case_has_culprit = any(role.role == "culprit" for role in case.chunk_roles)
            has_culprit.append(case_has_culprit)
            if not case_has_culprit:
                gold_when_illposed.append(provenance.get(prediction.predicted_culprit_id) == "gold")

        n = len(recall1)
        if n == 0:
            continue
        reports.append(
            RecoveryReport(
                method=method,
                n=n,
                mean_responsible=(sum(sizes) / n) if n else 0.0,
                recall_at_1=(sum(recall1) / n) if n else None,
                recall_at_k=(sum(recallk) / n) if n else None,
                innocent_gold_rate=(sum(gold_when_illposed) / len(gold_when_illposed)) if gold_when_illposed else None,
                reliability_auroc=auroc(margins, causal_hit),
                single_culprit_auroc=auroc(margins, has_culprit),
            )
        )
    return reports
