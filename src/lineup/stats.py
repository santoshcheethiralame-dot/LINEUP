from __future__ import annotations


def auroc(scores: list, labels: list) -> float | None:
    """Rank-based AUROC (Mann-Whitney): the chance a positive outranks a negative, ties at 0.5.

    Returns None when one class is absent, so a caller can tell "undefined" from "no separation".
    """
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


def percentile_interval(values, alpha: float = 0.05, min_defined: int = 20):
    """A two-sided percentile interval over bootstrap samples, ignoring undefined draws.

    Returns (low, high), or (None, None) when too few resamples produced a defined value for
    the interval to mean anything.
    """
    ordered = sorted(value for value in values if value is not None)
    n = len(ordered)
    if n < min_defined:
        return (None, None)

    def quantile(p: float):
        rank = max(1, min(n, round(p * (n + 1))))   # nearest-rank order statistic
        return ordered[rank - 1]

    return (quantile(alpha / 2), quantile(1 - alpha / 2))
