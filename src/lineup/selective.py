from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles, MethodPrediction
from .setvalued import _responsible_ids
from .stats import auroc

# Candidate confidence signals, all "higher = more sure there is one dominant culprit". They are
# scale-robust so they can be pooled across models whose effect scores live on different scales:
#   margin        top1 - top2 effect
#   gap_ratio     (top1 - top2) relative to |top1|
#   concentration top1's share of all positive effect (1.0 = one chunk owns the whole effect)
#   few_positives shrinks as more chunks carry positive effect (a coalition signal)
SIGNALS = ("margin", "gap_ratio", "concentration", "few_positives")


def _signals(prediction: MethodPrediction) -> dict:
    scores = sorted((score.score for score in prediction.chunk_scores), reverse=True)
    top1 = scores[0]
    top2 = scores[1] if len(scores) > 1 else 0.0
    positives = [score for score in scores if score > 0]
    positive_sum = sum(positives)
    return {
        "margin": top1 - top2,
        "gap_ratio": (top1 - top2) / (abs(top1) + 1e-6),
        "concentration": (max(top1, 0.0) / positive_sum) if positive_sum > 0 else 0.0,
        "few_positives": 1.0 / (1 + len(positives)),
    }


@dataclass
class SelectiveCase:
    key: str
    signals: dict
    hit: bool              # the case has exactly one culprit AND the top-1 pick is it
    well_posed: bool       # exactly one culprit exists
    responsible: set       # the planted wrong-value chunks (size > 1 under redundancy)
    ranked: list           # chunk ids by effect score, descending


def build_cases(cases: Iterable[CaseRoles], predictions: Iterable[MethodPrediction], *, method: str = "contextcite", source: str = "") -> list[SelectiveCase]:
    by_qid = {case.qid: case for case in cases}
    built = []
    for prediction in predictions:
        if prediction.method != method:
            continue
        case = by_qid.get(prediction.qid)
        if case is None:
            continue
        culprits = {role.chunk_id for role in case.chunk_roles if role.role == "culprit"}
        well_posed = len(culprits) == 1
        ranked = [score.chunk_id for score in sorted(prediction.chunk_scores, key=lambda s: s.score, reverse=True)]
        built.append(
            SelectiveCase(
                key=f"{source}/{prediction.qid}",
                signals=_signals(prediction),
                hit=well_posed and (prediction.predicted_culprit_id in culprits),
                well_posed=well_posed,
                responsible=_responsible_ids(case),
                ranked=ranked,
            )
        )
    return built


def split(cases: list[SelectiveCase]):
    """Deterministic 50/50 train/test split by a hash of the case key (no question leaks across)."""
    train, test = [], []
    for case in cases:
        digest = int(hashlib.sha256(case.key.encode()).hexdigest(), 16)
        (train if digest % 2 == 0 else test).append(case)
    return train, test


def signal_auroc(cases: list[SelectiveCase], signal: str):
    return auroc([case.signals[signal] for case in cases], [case.hit for case in cases])


def best_signal(cases: list[SelectiveCase]) -> str:
    return max(SIGNALS, key=lambda signal: signal_auroc(cases, signal) or 0.0)


def risk_coverage(cases: list[SelectiveCase], signal: str, coverages=(0.3, 0.5, 0.7, 1.0)) -> dict:
    """Selective accuracy at each coverage: attempt the most-confident fraction, abstain on the rest."""
    ordered = sorted(cases, key=lambda case: case.signals[signal], reverse=True)
    result = {}
    for coverage in coverages:
        k = max(1, round(coverage * len(ordered)))
        attempted = ordered[:k]
        result[coverage] = sum(case.hit for case in attempted) / len(attempted)
    return result


def set_recall_on_abstained(cases: list[SelectiveCase], signal: str, coverage: float):
    """For the low-confidence tail we abstain on, how much of the responsible set does the
    effect-set (top-|responsible| by score) recover, vs a single pick?"""
    ordered = sorted(cases, key=lambda case: case.signals[signal], reverse=True)
    k = max(1, round(coverage * len(ordered)))
    abstained = ordered[k:]
    recall_1, recall_k = [], []
    for case in abstained:
        if not case.responsible:
            continue
        size = len(case.responsible)
        recall_1.append(len(set(case.ranked[:1]) & case.responsible) / size)
        recall_k.append(len(set(case.ranked[:size]) & case.responsible) / size)
    return (
        (sum(recall_1) / len(recall_1)) if recall_1 else None,
        (sum(recall_k) / len(recall_k)) if recall_k else None,
    )
