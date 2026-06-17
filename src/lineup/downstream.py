from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .data.schema import CaseRoles, GenerationResult, MethodPrediction
from .stats import percentile_interval


@dataclass
class AbstentionReport:
    signal: str
    n: int
    n_correct: int
    auroc: float | None                       # confidence separates correct from wrong answers
    aurc: float | None                        # area under the risk-coverage curve (lower is better)
    selective_accuracy_at_50: float | None    # accuracy on the most-confident half


def _auroc(scores: list, labels: list) -> float | None:
    positives = sum(1 for label in labels if label)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    ranks = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average = (i + j - 1) / 2 + 1
        for index in range(i, j):
            ranks[index] = average
        i = j
    positive_rank_sum = sum(rank for rank, (_, label) in zip(ranks, ordered) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def risk_coverage_curve(scores: list, labels: list):
    """Answer the most-confident first; return coverage and the answered set's error rate."""
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    coverages = []
    risks = []
    correct = 0
    for answered, (_, label) in enumerate(ordered, start=1):
        correct += 1 if label else 0
        coverages.append(answered / len(ordered))
        risks.append(1 - correct / answered)
    return coverages, risks


def _risk_coverage(scores: list, labels: list):
    if not scores:
        return None, None
    _, risks = risk_coverage_curve(scores, labels)
    aurc = sum(risks) / len(risks)
    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    half = max(1, len(ordered) // 2)
    selective = sum(1 for _, label in ordered[:half] if label) / half
    return aurc, selective


def _margin(prediction: MethodPrediction) -> float:
    scores = sorted((chunk_score.score for chunk_score in prediction.chunk_scores), reverse=True)
    return scores[0] - scores[1] if len(scores) >= 2 else 0.0


def _oracle_confidence(case: CaseRoles) -> float:
    # A perfect causal account: a correct answer is trustworthy, and a wrong answer that some
    # chunk-removal turns correct is a detectable, fixable error to abstain on.
    if case.original_correct:
        return 1.0
    return 0.0 if any(chunk_role.now_correct for chunk_role in case.chunk_roles) else 1.0


def _collect_signals(generations, predictions, roles):
    generations = list(generations)
    qids = [generation.qid for generation in generations]
    correct = {generation.qid: generation.is_correct for generation in generations}

    signals: dict = {"self_logprob": {g.qid: g.answer_logprob for g in generations}}

    by_method: dict = defaultdict(dict)
    for prediction in predictions:
        by_method[prediction.method][prediction.qid] = prediction
    for method, predictions_by_qid in sorted(by_method.items()):
        signals[f"{method}_margin"] = {qid: _margin(p) for qid, p in predictions_by_qid.items()}

    if roles is not None:
        signals["oracle"] = {case.qid: _oracle_confidence(case) for case in roles}

    return qids, correct, signals


def evaluate_abstention(
    generations: Iterable[GenerationResult],
    predictions: Iterable[MethodPrediction],
    roles: Iterable[CaseRoles] | None = None,
) -> list[AbstentionReport]:
    """Selective-QA evaluation: can a confidence signal tell correct answers from wrong ones
    well enough to abstain on the rest? Compares the model's own confidence, each method's
    attribution decisiveness, and an oracle upper bound."""
    qids, correct, signals = _collect_signals(generations, predictions, roles)

    reports = []
    for name, confidence in signals.items():
        common = [qid for qid in qids if qid in confidence]
        scores = [confidence[qid] for qid in common]
        labels = [correct[qid] for qid in common]
        if any(labels) and not all(labels):
            aurc, selective = _risk_coverage(scores, labels)
        else:
            aurc, selective = None, None
        reports.append(
            AbstentionReport(
                signal=name,
                n=len(common),
                n_correct=sum(1 for label in labels if label),
                auroc=_auroc(scores, labels),
                aurc=aurc,
                selective_accuracy_at_50=selective,
            )
        )
    return reports


def bootstrap_abstention_auroc(
    generations: Iterable[GenerationResult],
    predictions: Iterable[MethodPrediction],
    roles: Iterable[CaseRoles] | None = None,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Per-signal confidence intervals for the abstention AUROC, by resampling cases.

    Returns {signal: (low, high)}. The case is the resampling unit, so the interval reflects
    how far a different draw of questions could move each signal's separating power.
    """
    qids, correct, signals = _collect_signals(generations, predictions, roles)
    n = len(qids)
    if n == 0:
        return {name: (None, None) for name in signals}
    rng = random.Random(seed)
    samples: dict = {name: [] for name in signals}
    for _ in range(n_boot):
        resample = [qids[rng.randrange(n)] for _ in range(n)]
        for name, confidence in signals.items():
            common = [qid for qid in resample if qid in confidence]
            scores = [confidence[qid] for qid in common]
            labels = [correct[qid] for qid in common]
            samples[name].append(_auroc(scores, labels))
    return {name: percentile_interval(samples[name], alpha) for name in signals}


def abstention_curves(generations, predictions, roles=None) -> dict:
    """The risk-coverage curve per signal, for plotting."""
    qids, correct, signals = _collect_signals(generations, predictions, roles)
    curves = {}
    for name, confidence in signals.items():
        common = [qid for qid in qids if qid in confidence]
        scores = [confidence[qid] for qid in common]
        labels = [correct[qid] for qid in common]
        if scores:
            curves[name] = risk_coverage_curve(scores, labels)
    return curves
