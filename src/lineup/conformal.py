from __future__ import annotations

import math
from dataclasses import dataclass

from .data.schema import CaseRoles, MethodPrediction

# Split-conformal prediction sets for attribution. Two coverage targets share the same machinery.
# The well-posed target is the single culprit (exactly one causal-and-salient passage); the
# nonconformity score is the culprit's rank in the effect ranking. The general target is the
# responsible set -- every individually-causal passage -- covered either in part (mode 'any', best
# rank) or in full (mode 'all', worst rank), which carries the guarantee past the single-culprit
# subset to the over-determined and silent-driver cases. Cases with no causal passage at all (the
# no-single-cause coalitions) cannot be placed by single-removal ranking and are left to abstention.
# Calibrating a rank threshold tau yields a top-tau set covering the target with probability
# >= 1 - alpha (marginal, on exchangeable data). The score is scale-free, so effect scores from
# different models pool cleanly; the tau needed per condition shows how the set must grow under
# redundancy for the same coverage.


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


def responsible_ids(case) -> list:
    """The passages a correct attribution must cover: every chunk individually causal to the error.
    For a single-culprit case this is the culprit alone; for over-determined and silent-driver cases
    it is the whole causal set. The no-single-cause coalitions have an empty set here — single-removal
    ranking cannot place them, so they fall to the abstention path rather than the conformal set."""
    return [role.chunk_id for role in case.chunk_roles if role.causal]


def build_ranks_set(cases, predictions, *, method: str = "contextcite", mode: str = "any", source: str = "", condition: str = "") -> list[RankItem]:
    """Ranks for covering the responsible *set*, not only a single culprit.

    mode 'any' covers at least one responsible passage (nonconformity = best rank); mode 'all' covers
    the entire responsible set (nonconformity = worst rank), the honest target when blame is shared.
    Cases with no causal passage are skipped (responsible_ids), as are cases where a responsible
    passage never appears in the method's ranking.
    """
    by_qid = {case.qid: case for case in cases}
    items = []
    for prediction in predictions:
        if prediction.method != method:
            continue
        case = by_qid.get(prediction.qid)
        if case is None:
            continue
        responsible = responsible_ids(case)
        if not responsible:
            continue
        ranked = _ranked(prediction)
        ranks = [ranked.index(cid) + 1 for cid in responsible if cid in ranked]
        if len(ranks) != len(responsible):
            continue
        rank = max(ranks) if mode == "all" else min(ranks)
        items.append(RankItem(f"{source}/{prediction.qid}", condition, rank, len(ranked)))
    return items
