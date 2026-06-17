from __future__ import annotations


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
