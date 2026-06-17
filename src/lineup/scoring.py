from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles, MethodPrediction
from .stats import percentile_interval

_ROLES = ("culprit", "misleading", "silent", "inert")
_INTERVAL_METRICS = (
    "top1_culprit_accuracy",
    "misleading_as_culprit_rate",
    "culprit_over_misleading_winrate",
    "culprit_over_rest_winrate",
)


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


@dataclass
class _CaseMetric:
    """The per-case primitives both the point estimate and the bootstrap aggregate over."""

    has_culprit: bool
    predicted_role: str
    misleading_wins: float
    misleading_pairs: int
    rest_wins: float
    rest_pairs: int


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


def _case_metric(prediction: MethodPrediction, roles: dict) -> _CaseMetric:
    scores = {score.chunk_id: score.score for score in prediction.chunk_scores}
    misleading_wins, misleading_pairs = _winrate(roles, scores, lambda r: r == "culprit", lambda r: r == "misleading")
    rest_wins, rest_pairs = _winrate(roles, scores, lambda r: r == "culprit", lambda r: r != "culprit")
    return _CaseMetric(
        has_culprit=any(role == "culprit" for role in roles.values()),
        predicted_role=roles.get(prediction.predicted_culprit_id, "inert"),
        misleading_wins=misleading_wins,
        misleading_pairs=misleading_pairs,
        rest_wins=rest_wins,
        rest_pairs=rest_pairs,
    )


def _aggregate(method: str, metrics: list) -> MethodReport:
    n = len(metrics)
    role_counts: Counter = Counter(metric.predicted_role for metric in metrics)
    n_with_culprit = sum(1 for metric in metrics if metric.has_culprit)
    top1 = sum(1 for metric in metrics if metric.has_culprit and metric.predicted_role == "culprit")
    misleading_wins = sum(metric.misleading_wins for metric in metrics)
    misleading_pairs = sum(metric.misleading_pairs for metric in metrics)
    rest_wins = sum(metric.rest_wins for metric in metrics)
    rest_pairs = sum(metric.rest_pairs for metric in metrics)

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


def _metrics_by_method(cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction]) -> dict:
    role_by_case = {
        case.qid: {chunk_role.chunk_id: chunk_role.role for chunk_role in case.chunk_roles}
        for case in cases
    }
    by_method: dict = {}
    for prediction in predictions:
        metrics = by_method.setdefault(prediction.method, [])
        roles = role_by_case.get(prediction.qid)
        if roles is not None:
            metrics.append(_case_metric(prediction, roles))
    return by_method


def score_predictions(
    cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction]
) -> list[MethodReport]:
    """Grade every method's predictions against the oracle's role labels, joined by question id."""
    by_method = _metrics_by_method(cases, predictions)
    return [_aggregate(method, by_method[method]) for method in sorted(by_method)]


def bootstrap_intervals(
    cases: Iterable[CaseRoles],
    predictions: Iterable[MethodPrediction],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Per-method confidence intervals for the headline metrics, by resampling cases with replacement.

    Returns {method: {metric: (low, high)}}. The case is the resampling unit, so the intervals
    reflect how much a different draw of questions could move each number.
    """
    by_method = _metrics_by_method(cases, predictions)
    rng = random.Random(seed)
    intervals: dict = {}
    for method in sorted(by_method):
        metrics = by_method[method]
        n = len(metrics)
        if n == 0:
            intervals[method] = {name: (None, None) for name in _INTERVAL_METRICS}
            continue
        samples: dict = {name: [] for name in _INTERVAL_METRICS}
        for _ in range(n_boot):
            resample = [metrics[rng.randrange(n)] for _ in range(n)]
            report = _aggregate(method, resample)
            for name in _INTERVAL_METRICS:
                samples[name].append(getattr(report, name))
        intervals[method] = {name: percentile_interval(samples[name], alpha) for name in _INTERVAL_METRICS}
    return intervals
