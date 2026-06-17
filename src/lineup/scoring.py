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
    n_with_culprit: int                            # cases that actually have a findable culprit
    top1_culprit_accuracy: float | None            # over the cases with a culprit; None if none have one
    predicted_role_rate: dict                      # role -> share of predictions landing on that true role
    misleading_as_culprit_rate: float              # the headline: predictions landing on a salient-but-innocent chunk
    culprit_over_misleading_winrate: float | None  # within-case: culprit ranked above the near-miss (the hard pair)
    n_misleading_pairs: int
    culprit_over_rest_winrate: float | None        # within-case: culprit ranked above any non-culprit chunk
    n_rest_pairs: int


def _winrate(roles: dict, scores: dict, is_positive, is_negative):
    positives = [cid for cid, role in roles.items() if is_positive(role)]
    negatives = [cid for cid, role in roles.items() if is_negative(role)]
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            positive_score = scores.get(positive, 0.0)
            negative_score = scores.get(negative, 0.0)
            wins += 1.0 if positive_score > negative_score else (0.5 if positive_score == negative_score else 0.0)
            total += 1
    return wins, total


def _score_one(method: str, predictions: list, role_by_case: dict) -> MethodReport:
    n = 0
    n_with_culprit = 0
    top1 = 0
    role_counts: Counter = Counter()
    misleading_wins = 0.0
    misleading_pairs = 0
    rest_wins = 0.0
    rest_pairs = 0

    for prediction in predictions:
        roles = role_by_case.get(prediction.qid)
        if roles is None:
            continue
        n += 1
        predicted_role = roles.get(prediction.predicted_culprit_id, "inert")
        role_counts[predicted_role] += 1
        if any(role == "culprit" for role in roles.values()):
            n_with_culprit += 1
            if predicted_role == "culprit":
                top1 += 1

        scores = {score.chunk_id: score.score for score in prediction.chunk_scores}
        wins, pairs = _winrate(roles, scores, lambda r: r == "culprit", lambda r: r == "misleading")
        misleading_wins += wins
        misleading_pairs += pairs
        wins, pairs = _winrate(roles, scores, lambda r: r == "culprit", lambda r: r != "culprit")
        rest_wins += wins
        rest_pairs += pairs

    role_rate = {role: (role_counts[role] / n if n else 0.0) for role in _ROLES}
    return MethodReport(
        method=method,
        n_cases=n,
        n_with_culprit=n_with_culprit,
        top1_culprit_accuracy=(top1 / n_with_culprit) if n_with_culprit else None,
        predicted_role_rate=role_rate,
        misleading_as_culprit_rate=role_rate["misleading"],
        culprit_over_misleading_winrate=(misleading_wins / misleading_pairs) if misleading_pairs else None,
        n_misleading_pairs=misleading_pairs,
        culprit_over_rest_winrate=(rest_wins / rest_pairs) if rest_pairs else None,
        n_rest_pairs=rest_pairs,
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
