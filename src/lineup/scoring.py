from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles, MethodPrediction

_ROLES = ("culprit", "misleading", "silent", "inert")


@dataclass
class MethodReport:
    method: str
    n_cases: int
    top1_culprit_accuracy: float          # how often the argmax pick is a true culprit
    predicted_role_rate: dict             # role -> share of predictions landing on that true role
    misleading_as_culprit_rate: float     # the headline: predictions landing on a salient-but-innocent chunk
    culprit_over_misleading_winrate: float | None  # within-case: culprit scored above misleading
    n_pairs: int                          # number of within-case (culprit, misleading) pairs scored


def _winrate_pairs(roles: dict, score_by_chunk: dict):
    culprits = [cid for cid, role in roles.items() if role == "culprit"]
    misleading = [cid for cid, role in roles.items() if role == "misleading"]
    wins = 0.0
    total = 0
    for culprit in culprits:
        for decoy in misleading:
            culprit_score = score_by_chunk.get(culprit, 0.0)
            decoy_score = score_by_chunk.get(decoy, 0.0)
            wins += 1.0 if culprit_score > decoy_score else (0.5 if culprit_score == decoy_score else 0.0)
            total += 1
    return wins, total


def _score_one(method: str, predictions: list, role_by_case: dict) -> MethodReport:
    n = 0
    top1 = 0
    role_counts: Counter = Counter()
    wins = 0.0
    pairs = 0
    for prediction in predictions:
        roles = role_by_case.get(prediction.qid)
        if roles is None:
            continue
        n += 1
        predicted_role = roles.get(prediction.predicted_culprit_id, "inert")
        role_counts[predicted_role] += 1
        if predicted_role == "culprit":
            top1 += 1
        score_by_chunk = {score.chunk_id: score.score for score in prediction.chunk_scores}
        case_wins, case_pairs = _winrate_pairs(roles, score_by_chunk)
        wins += case_wins
        pairs += case_pairs

    role_rate = {role: (role_counts[role] / n if n else 0.0) for role in _ROLES}
    return MethodReport(
        method=method,
        n_cases=n,
        top1_culprit_accuracy=top1 / n if n else 0.0,
        predicted_role_rate=role_rate,
        misleading_as_culprit_rate=role_rate["misleading"],
        culprit_over_misleading_winrate=(wins / pairs) if pairs else None,
        n_pairs=pairs,
    )


def score_predictions(
    cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction]
) -> list[MethodReport]:
    """Grade every method's predictions against the oracle's role labels, joined by question id."""
    role_by_case = {
        case.qid: {chunk_role.chunk_id: chunk_role.role for chunk_role in case.chunk_roles}
        for case in cases
    }
    by_method: dict = defaultdict(list)
    for prediction in predictions:
        by_method[prediction.method].append(prediction)
    return [_score_one(method, by_method[method], role_by_case) for method in sorted(by_method)]
