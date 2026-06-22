from __future__ import annotations

import math
from dataclasses import dataclass

from .data.schema import CaseRoles, MethodPrediction

# Split-conformal prediction sets for attribution. The label we cover is the single culprit, so we
# restrict to well-posed cases (exactly one culprit). The nonconformity score is the rank of the
# culprit in the effect ranking; calibrating a rank threshold tau yields a top-tau set that
# contains the culprit with probability >= 1 - alpha (marginal, on exchangeable data). It is
# scale-free, so effect scores from different models pool cleanly. Reporting the tau needed per
# condition shows how much larger attribution sets must be under redundancy for the same coverage.


@dataclass
class RankItem:
    key: str
    condition: str
    rank: int        # 1-indexed position of the single culprit in the effect ranking
    n_chunks: int


def _ranked(prediction: MethodPrediction) -> list:
    return [score.chunk_id for score in sorted(prediction.chunk_scores, key=lambda s: s.score, reverse=True)]


def build_ranks(cases, predictions, *, method: str = "contextcite", source: str = "", condition: str = "") -> list[RankItem]:
    by_qid = {case.qid: case for case in cases}
    items = []
    for prediction in predictions:
        if prediction.method != method:
            continue
        case = by_qid.get(prediction.qid)
        if case is None:
            continue
        culprits = [role.chunk_id for role in case.chunk_roles if role.role == "culprit"]
        if len(culprits) != 1:        # conformal target is the single culprit
            continue
        ranked = _ranked(prediction)
        if culprits[0] not in ranked:
            continue
        items.append(RankItem(f"{source}/{prediction.qid}", condition, ranked.index(culprits[0]) + 1, len(ranked)))
    return items


def calibrate_tau(calibration: list[RankItem], alpha: float) -> int | None:
    """The smallest set size tau whose empirical culprit-coverage on the calibration split meets
    the split-conformal 1 - alpha level."""
    ranks = sorted(item.rank for item in calibration)
    n = len(ranks)
    if n == 0:
        return None
    index = min(math.ceil((n + 1) * (1 - alpha)), n)   # 1-indexed order statistic
    return ranks[index - 1]


def coverage_and_size(test: list[RankItem], tau: int):
    """Empirical coverage of the top-tau set and its average size (capped per case at chunk count)."""
    if not test:
        return None, None
    covered = sum(1 for item in test if item.rank <= tau) / len(test)
    size = sum(min(tau, item.n_chunks) for item in test) / len(test)
    return covered, size


def top1_coverage(test: list[RankItem]) -> float | None:
    """Coverage of a single top-1 pick — the conformal set of size 1, for comparison."""
    if not test:
        return None
    return sum(1 for item in test if item.rank == 1) / len(test)
